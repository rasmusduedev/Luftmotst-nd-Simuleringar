import matplotlib.pyplot as plt
import taichi as ti
import math
ti.init(ti.gpu)

# NOTE: Uppdelad i komponenter

pi = math.pi
dt = 0.01
kL = 0.18
kM = 0.03
rot_varv = 10 # varv per sekund
start_vinkel = 45 # uppskjutningsvinkel i grader
m = 2 # massa i kg
v_init = 12 # uppskjutningshastighet i m/s
max_loops = 10000

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
def standard_euler(): # fixade fel - numera standard istället för semi-implicit
    pos[None] += dt * vel[None]
    vel[None] += dt * acceleration(vel[None]) # NOTE: DO NOT switch places between pos[None] and vel[None] updates


init()
x_pos = []
y_pos = []
count, t = 0, 0.
while pos[None].y > 0. and count <= max_loops:
    standard_euler()
    x_pos.append(pos[None].x)
    y_pos.append(pos[None].y)
    count += 1
    t += dt

print(f'\n\n\n{count+1} iterations\nt = {t} s\n\n\n')
A = 6  # Want figures to be A6
plt.rc('figure', figsize=[1.55*46.82 * .5**(.5 * A), 1.55*33.11 * .5**(.5 * A)])
plt.rc('text', usetex=True)
font = {'family':'normal', 'weight':'bold', 'size':22}
plt.rc('font', **font)

plt.plot(x_pos, y_pos, color = 'b', label='Projektilens bana', linewidth=4)
plt.xlabel('Längd längs med marken [m]')
plt.ylabel('Altitud [m]')
plt.legend()
#ax = plt.gca()
#ax.set_aspect('equal', adjustable='box')
#ax.scatter([0], [10], marker=(20,1), color="k", s=250)
plt.grid()
plt.show()