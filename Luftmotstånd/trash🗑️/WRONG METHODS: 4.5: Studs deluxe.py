import matplotlib.pyplot as plt
import taichi as ti
import math
ti.init(ti.gpu)

# NOTE: Uppdelad i komponenter

pi = math.pi
dt_standard = 0.01
kL = 0.18
kM = 0.03
rot_varv = 10 # varv per sekund
start_vinkel = 45 # uppskjutningsvinkel i grader
m = 2 # massa i kg
v_init = 12 # uppskjutningshastighet i m/s
max_loops = 10000
percent_loss = 10.
radius = 0.1
tolerance_under_zero = -0.001
new_dt = ti.field(ti.f32, ())
new_dt[None] = dt_standard
tolerans_längd = 0.01
t = ti.field(ti.f32, ())

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

#@ti.func
def simple_euler(dt: float):
    vinkel = ti.atan2(vel[None].y, vel[None].x)
    acc[None].x = - (kL/m)*ti.cos(vinkel)*vel[None].norm()**2 - (kM/m)*ω*vel[None].norm()*ti.sin(vinkel)
    acc[None].y = - (kL/m)*ti.sin(vinkel)*vel[None].norm()**2 + (kM/m)*ω*vel[None].norm()*ti.cos(vinkel) - g

    vel[None] += dt * acc[None]
    pos[None] += dt * vel[None]

#@ti.func
def update_for_impact(pos_old, vel_old):
    new_dt[None] = dt_standard
    sub_count = 0
    pos[None] = pos_old
    vel[None] = vel_old
    new_dt[None] /= 20
    # fortsätt updatera tills en viss längd från marken, with old (before latest update) values
    while (pos[None].y - radius) >= tolerans_längd and sub_count <= 50:
        simple_euler(dt=new_dt[None])
        t[None] += new_dt[None]
        sub_count += 1
    print(sub_count)

#@ti.kernel
def update():
    if (pos[None].y - radius) <= 0.:
        new_kinetic = (1-(percent_loss/100)) * (m*vel[None].norm()**2)/2  #förlora lite rörelseenergi för varje studs
        vinkel_studs = ti.atan2(ti.abs(vel[None].y), vel[None].x) # absolutvärdet av y-komponenten, då den nya studshastigheten annars får negativ y-komposant (alltså den fortsätter färdas ned i marken)
        new_velocity = ti.sqrt(2*new_kinetic/m) # ny hastighet efter att rörelseenergin minskat
        vel[None] = ti.Vector([ti.cos(vinkel_studs), ti.sin(vinkel_studs)]) * new_velocity
        # ^ ny hastighetsvektor
    
    pos_old = pos[None]
    vel_old = vel[None]
    simple_euler(dt=dt_standard)
    t[None] += dt_standard

    if (pos[None].y - radius) <= 0.: # if update brings projectile under the ground (illegal)
        update_for_impact(pos_old, vel_old)
    else:
        t[None] += dt_standard

init()
x_pos = []
y_pos = []
count = 0
t[None] = 0.
while vel[None] > 0.5 and count <= max_loops and pos[None].y >=-0.3:# and (pos[None].y + radius) >= tolerance_under_zero:
    update()
    x_pos.append(pos[None].x)
    y_pos.append(pos[None].y)
    count += 1
    #print(pos[None].y)
A = ti.Vector([x_pos[-2], y_pos[-2]])
B = ti.Vector([x_pos[-1], y_pos[-1]])
Nedslag_x = 0.
if B.x != A.x:
    k = (B.y - A.y)/(B.x - A.x)
    Nedslag_x = B.x - (B.y/k)

print(f'\n\n\n{count+1} iterations\nt = {t[None]} s\nNedslag = {Nedslag_x} m\n\n\n')
A = 6  # Want figures to be A6
plt.rc('figure', figsize=[1.55*46.82 * .5**(.5 * A), 1.55*33.11 * .5**(.5 * A)])
plt.rc('text', usetex=True)
font = {'family':'normal', 'weight':'bold', 'size':22}
plt.rc('font', **font)

plt.plot(x_pos, y_pos, color = 'b', label='Projektilens bana', linewidth=4)
plt.plot(x_pos, y_pos, marker='o')
plt.axhline(y=radius, color='r', linestyle='-')
plt.xlabel('Längd längs med marken')
plt.ylabel('Altitud')
plt.legend()
plt.grid()
plt.show()