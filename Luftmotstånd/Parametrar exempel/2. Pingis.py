import matplotlib.pyplot as plt
import taichi as ti
import math
ti.init(ti.gpu)

# NOTE: Uppdelad i komponenter

pi = math.pi
dt = 0.001
kL = 3.8e-4
kM = 3.0e-5
rot_varv = -100. # varv per sekund
start_vinkel = -20 # uppskjutningsvinkel i grader
m = 0.0027 # massa i kg
v_init = 11.2 # uppskjutningshastighet i m/s
max_loops = 10000
percent_loss = 30

ω = rot_varv * 2*pi
angle_rad = start_vinkel * (pi/180)
g = 9.82 # gravitationsacc[None]eleration i m/s²

pos = ti.Vector.field(2, ti.f32, ())
vel = ti.Vector.field(2, ti.f32, ())
acc = ti.Vector.field(2, ti.f32, ())

@ti.kernel
def init():
    pos[None] = [0., 0.5]
    vel[None] = ti.Vector([ti.cos(angle_rad), ti.sin(angle_rad)]) * v_init
    acc[None] = [0., 0.]

@ti.func
def acceleration(v: ti.math.vec2) -> ti.math.vec2:
    a = ti.Vector([0., 0.])
    vinkel = ti.atan2(v.y, v.x)
    a.x = - (kL/m)*ti.cos(vinkel)*v.norm()**2 - (kM/m)*ω*v.norm()*ti.sin(vinkel)
    a.y = - (kL/m)*ti.sin(vinkel)*v.norm()**2 + (kM/m)*ω*v.norm()*ti.cos(vinkel) - g
    return a

@ti.func
def standard_euler(): # fixade fel - numera standard istället för semi-implicit
    pos[None] += dt * vel[None]
    vel[None] += dt * acceleration(vel[None]) # NOTE: DO NOT switch places between pos[None] and vel[None] updates

@ti.func
def RK2(): # Fixade fel
    vel_n, pos_n = vel[None], pos[None]
    vel_half = vel_n + (dt/2) * acceleration(vel_n)  # vel now stores n+½ values

    vel[None] = vel_n + dt * acceleration(vel_half) # goes back and updates old vel values using n+½ derivative
    pos[None] = pos_n + dt * vel_half # Updates old pos values using n+½ derivatives

@ti.kernel
def update():
    if pos[None].y <= 0. and pos[None].x <= 2.74:
        new_kinetic = (1-(percent_loss/100)) * (m*vel[None].norm()**2)/2  #förlora lite rörelseenergi för varje studs
        vinkel_studs = ti.atan2(ti.abs(vel[None].y), vel[None].x) # absolutvärdet av y-komponenten, då den nya studshastigheten annars får negativ y-komposant (alltså den fortsätter färdas ned i marken)
        new_velocity = ti.sqrt(2*new_kinetic/m) # ny hastighet efter att rörelseenergin minskat
        vel[None] = ti.Vector([ti.cos(vinkel_studs), ti.sin(vinkel_studs)]) * new_velocity
        # ^ ny hastighetsvektor
    RK2()

init()
x_pos = []
y_pos = []
count, t = 0, 0.
while vel[None] > 0.5 and count <= max_loops and pos[None].y >=-0.5:
    update()

    x_pos.append(pos[None].x)
    y_pos.append(pos[None].y)
    count += 1
    t += dt
A = ti.Vector([x_pos[-2], y_pos[-2]])
B = ti.Vector([x_pos[-1], y_pos[-1]])
if B.x == A.x: Nedslag_x = 'Okänt'
else: k = (B.y - A.y)/(B.x - A.x); Nedslag_x = B.x - (B.y/k)

print(f'\n\n\n{count+1} iterations\nt = {t:.4} s\nNedslag = {Nedslag_x:.4} m\nMax höjd = {max(y_pos):.4} m\n\n\n')
A = 6  # Want figures to be A6
plt.rc('figure', figsize=[1.55*46.82 * .5**(.5 * A), 1.55*33.11 * .5**(.5 * A)])
plt.rc('text', usetex=True)
font = {'family':'normal', 'weight':'bold', 'size':22}
plt.rc('font', **font)

plt.plot(x_pos, y_pos, color = 'b', label='Projektilens bana', linewidth=4)
plt.xlabel('Längd längs med marken [m]')
plt.ylabel('Altitud [m]')
#plt.legend()
plt.vlines(x = 1.37, ymin = 0, ymax = 0.1525, color = 'k', linewidth=7)
plt.hlines(y=0, xmin = 0., xmax = 2.74, color='k', linewidth=7)
plt.grid()
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.show()