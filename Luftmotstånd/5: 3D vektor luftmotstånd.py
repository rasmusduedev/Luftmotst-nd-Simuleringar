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

ω = rot_varv * 2*pi * ti.Vector([0., 0., 1.])
angle_rad = start_vinkel * (pi/180)
g = 9.82 # gravitationsacceleration i m/s²

pos = ti.Vector.field(3, ti.f32, ())
vel = ti.Vector.field(3, ti.f32, ())
acc = ti.Vector.field(3, ti.f32, ())

@ti.kernel
def init():
    pos[None] = [0., 10., 0.]
    vel[None] = ti.Vector([ti.cos(angle_rad), ti.sin(angle_rad), 0.]) * v_init
    acc[None] = [0., 0., 0.]

@ti.func
def calc_Forces():
    v = vel[None].norm()
    F_luft = kL*v**2 * (-vel[None]/v)
    F_magnus = kM*(ti.math.cross(ω, vel[None]))
    F_grav = ti.Vector([0., -g*m, 0.])

    F = F_luft + F_magnus + F_grav # combine the forces
    acc[None] = F/m

@ti.kernel
def simple_euler():
    calc_Forces() # fixade fel
    pos[None] += dt * vel[None]
    vel[None] += dt * acc[None]

init()
x_pos = []
y_pos = []
z_pos = []
count, t = 0, 0.
while pos[None].y > 0. and count <= max_loops:
    simple_euler()
    x_pos.append(pos[None].x)
    y_pos.append(pos[None].y)
    z_pos.append(pos[None].z)
    count += 1
    t += dt

print(f'\n\n\n{count+1} iterations\nt = {t} s\n\n\n')

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(x_pos, z_pos, y_pos, label='Projektilens bana', linewidth=3) # ändrar axlarna efter hur matplotlib visar datan

A = 6  # Want figures to be A6
plt.rc('text', usetex=True)
font = {'family':'normal', 'weight':'bold', 'size':22}
plt.rc('font', **font)

ax.set_xlabel('X axel')
ax.set_ylabel('Z axel') # ändrar axlarna så att det visas rätt
ax.set_zlabel('y axel')
plt.legend()
plt.show()