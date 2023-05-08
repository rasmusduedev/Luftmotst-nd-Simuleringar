import matplotlib.pyplot as plt
import taichi as ti
import math
ti.init(ti.gpu)

# NOTE: Uppdelad i komponenter

v_init = 53.64480 # uppskjutningshastighet i m/s
start_vinkel = 16.3 # uppskjutningsvinkel i grader
RPM = 6699

pi = math.pi
dt = 0.01
kL = 0.00048
kM = 0.00002
rot_varv = RPM/60 # varv per sekund
m = 0.046 # massa i kg
radius = 0.043/2 # är endast 4.3 cm i diameter.
max_loops = 10000

ω = rot_varv * 2*pi
angle_rad = start_vinkel * (pi/180)
g = 9.82 # gravitationsacc[None]eleration i m/s²

pos = ti.Vector.field(2, ti.f32, ())
vel = ti.Vector.field(2, ti.f32, ())
acc = ti.Vector.field(2, ti.f32, ())

@ti.kernel
def init():
    pos[None] = [0., 0.]
    vel[None] = ti.Vector([ti.cos(angle_rad), ti.sin(angle_rad)]) * v_init
    acc[None] = [0., 0.]

@ti.func
def acceleration(v: ti.math.vec2) -> ti.math.vec2:
    a = ti.Vector([0., 0.])
    vinkel = ti.atan2(v.y, v.x)
    a.x = - (kL/m)*ti.cos(vinkel)*v.norm()**2 - (kM/m)*ω*v.norm()*ti.sin(vinkel)
    a.y = - (kL/m)*ti.sin(vinkel)*v.norm()**2 + (kM/m)*ω*v.norm()*ti.cos(vinkel) - g
    return a

@ti.kernel
def standard_euler(): # fixade fel - numera standard istället för semi-implicit
    pos[None] += dt * vel[None]
    vel[None] += dt * acceleration(vel[None]) # NOTE: DO NOT switch places between pos[None] and vel[None] updates

@ti.kernel
def RK2_and_carry() -> ti.f32: # Fixade fel
    vel_n, pos_n = vel[None], pos[None]
    vel_half = vel_n + (dt/2) * acceleration(vel_n)  # vel now stores n+½ values

    vel[None] = vel_n + dt * acceleration(vel_half) # goes back and updates old vel values using n+½ derivative
    pos[None] = pos_n + dt * vel_half # Updates old pos values using n+½ derivatives

    travel = pos[None] - pos_n
    return travel.norm()

init()
x_pos = []
y_pos = []
count, t, carry = 0, 0., 0.
while pos[None].y >= 0. and count <= max_loops:
    distance = RK2_and_carry()

    x_pos.append(pos[None].x)
    y_pos.append(pos[None].y)
    count += 1
    t += dt
    carry += distance
A = ti.Vector([x_pos[-2], y_pos[-2]])
B = ti.Vector([x_pos[-1], y_pos[-1]])
k = (B.y - A.y)/(B.x - A.x)
Nedslag_x = B.x - (B.y/k)


print(f'\n\n\n{count+1} iterations\nt = {t:.4} s\nNedslag = {Nedslag_x:.4} m\n\nMax höjd = {max(y_pos):.4} m\nCarry = {carry:.4} m\n\n')
A = 6  # Want figures to be A6
plt.rc('figure', figsize=[1.55*46.82 * .5**(.5 * A), 1.55*33.11 * .5**(.5 * A)])
plt.rc('text', usetex=True)
font = {'family':'normal', 'weight':'bold', 'size':22}
plt.rc('font', **font)

plt.plot(x_pos, y_pos, color = 'b', label='Projektilens bana', linewidth=4)
plt.xlabel('Längd längs med marken')
plt.ylabel('Altitud')
plt.legend()
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.grid()
plt.show()