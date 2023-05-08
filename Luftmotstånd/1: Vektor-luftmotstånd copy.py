import matplotlib.pyplot as plt
import taichi as ti
import math
ti.init(ti.gpu)

# NOTE: Hanterar vektorer istället för enskilda komponenter

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
g = 9.82 # gravitationsacceleration i m/s²

pos = ti.Vector.field(2, ti.f32, ())
vel = ti.Vector.field(2, ti.f32, ())
acc = ti.Vector.field(2, ti.f32, ())

@ti.kernel
def init():
    pos[None] = [0., 10.]
    vel[None] = ti.Vector([ti.cos(angle_rad), ti.sin(angle_rad)]) * v_init
    acc[None] = [0., 0.]

@ti.func
def calc_Forces():
    v = vel[None].norm()
    F_luft = kL*v**2 * (-vel[None]/v)
    # Make 3d vectors in order to calculate cross product
    ω_3dvec = ti.Vector([0., 0., ω])
    v_3dvec = ti.Vector([vel[None].x, vel[None].y, 0.])
    F_M = kM*(ti.math.cross(ω_3dvec, v_3dvec))
    F_magnus = ti.Vector([F_M.x, F_M.y]) # only use the 2d projection of the resulting magnus force
    F_grav = ti.Vector([0., -g*m])

    F = F_luft + F_magnus + F_grav # combine the forces
    acc[None] = F/m

@ti.kernel
def standard_euler(): # fixade fel - numera standard Euler 
    calc_Forces()
    pos[None] += dt * vel[None]
    vel[None] += dt * acc[None]

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
plt.xlabel('Längd längs med marken')
plt.ylabel('Altitud')
plt.legend()
plt.grid()
plt.show()