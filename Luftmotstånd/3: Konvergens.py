import matplotlib.pyplot as plt
import taichi as ti
import math
ti.init(ti.gpu)

# NOTE: Uppdelad i komponenter

pi = math.pi
kL = 0.18
kM = 0.03
rot_varv = 10 # varv per sekund
start_vinkel = 45 # uppskjutningsvinkel i grader
m = 2 # massa i kg
v_init = 12 # uppskjutningshastighet i m/s
max_loops = 1000000

ω = rot_varv * 2*pi
angle_rad = start_vinkel * (pi/180)
g = 9.82 # gravitationsacc[None]eleration i m/s²

pos = ti.Vector.field(2, ti.f32, ())
vel = ti.Vector.field(2, ti.f32, ())
acc = ti.Vector.field(2, ti.f32, ())

@ti.kernel
def init():
    pos[None] = [0., 10.]
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
def standard_euler(dt: float): # fixade fel - numera standard istället för semi-implicit
    pos[None] += dt * vel[None]
    vel[None] += dt * acceleration(vel[None]) # NOTE: DO NOT switch places between pos[None] and vel[None] updates

@ti.kernel
def RK2(dt: float): # Fixade fel
    vel_n, pos_n = vel[None], pos[None]
    vel_half = vel_n + (dt/2) * acceleration(vel_n)  # vel now stores n+½ values

    vel[None] = vel_n + dt * acceleration(vel_half) # goes back and updates old vel values using n+½ derivative
    pos[None] = pos_n + dt * vel_half # Updates old pos values using n+½ derivatives



def calc_trajectory(dt: float):
    init()
    x_pos = []
    y_pos = []
    count, t = 0, 0.

    while pos[None].y > 0. and count <= max_loops:
        standard_euler(dt)
        x_pos.append(pos[None].x)
        y_pos.append(pos[None].y)
        count += 1
        t += dt
    A = ti.Vector([x_pos[-2], y_pos[-2]])
    B = ti.Vector([x_pos[-1], y_pos[-1]])
    k = (B.y - A.y)/(B.x - A.x)
    Nedslag_x = 0.
    if k == 0.: # Worse estimate
        Nedslag_x = (B.x - A.x)/2 + A.x
    else: # Better estimate
        Nedslag_x = B.x - (B.y/k)
    return Nedslag_x

minuslog10dt_line = []
Nx_line = []
base = 2
for i in range(13):
    dt = 1./(base**(i+1))
    N_x = calc_trajectory(dt)
    minuslog10dt_line.append(-math.log(dt, base))
    Nx_line.append(N_x)

A = 6  # Want figures to be A6
plt.rc('figure', figsize=[1.55*46.82 * .5**(.5 * A), 1.55*33.11 * .5**(.5 * A)])
plt.rc('text', usetex=True)
font = {'family':'normal', 'weight':'bold', 'size':22}
plt.rc('font', **font)

print(minuslog10dt_line)
plt.plot(minuslog10dt_line, Nx_line, color = 'b', label='Nedslagsplats - konvergens', linewidth=2)
plt.plot(minuslog10dt_line, Nx_line, 'o', markersize=10)
plt.xlabel(r'$-log_2(\Delta t)$')
plt.ylabel('Nedslagsplats')
plt.legend()
plt.grid()
plt.show()