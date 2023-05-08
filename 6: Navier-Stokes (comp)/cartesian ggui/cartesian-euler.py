import cartesian_flux as flu
from matplotlib import cm
import taichi as ti
import numpy as np
import matplotlib
import precomp
ti.init(ti.gpu)

# NOTE: Compressible Navier-Stokes equations with constant dynamic viscosity μ
# 2nd order in space, 2nd order in time.

# Constants
C = 0.4
gamma = 1.4
μ = 0.01#1.0e-3 # dynamic viscosity, here assumed to be constant
LENGTH = 801.#2. #1.
Rad_Cylinder = 25#0.07#0.05

init_rho = 1.0
init_vel = 40
init_pressure = 1000

# Grid
Nx = 801#500 # number of x-axis cells
Ny = 201#300 # number of y-axis cells

grid = (Nx, Ny)
q = ti.Vector.field(4, ti.f32, grid) # rho, u, w, p
U = ti.Vector.field(4, ti.f32, grid) # rho, rho*u, rho*w, rho*E

Δq_x = ti.Vector.field(4, ti.f32, grid) # slope along x
Δq_y = ti.Vector.field(4, ti.f32, grid) # slope along y 

x_L = ti.Vector.field(4, ti.f32, grid) # left face along x interface i-½
x_R = ti.Vector.field(4, ti.f32, grid) # right face along x interface i-½
y_L = ti.Vector.field(4, ti.f32, grid) # left face along y interface j-½
y_R = ti.Vector.field(4, ti.f32, grid) # right face along y interface j-½

x_flux = ti.Vector.field(4, ti.f32, grid) # x flux terms for i-½
y_flux = ti.Vector.field(4, ti.f32, grid) # y flux terms for j-½

# Quality of life thing to store the x and y terms in the divergence 
x_terms = ti.field(ti.f32, 4)
y_terms = ti.field(ti.f32, 4)

U_half_timestep = ti.Vector.field(4, ti.f32, grid)
div_t = ti.Vector.field(4, ti.f32, grid) # time-derivative for each conserved variable
dt = ti.field(ti.f32, shape=()) # For timestep variation, since velocity u now varies

# Slopes, Laplaces and Gradients for the viscosity force terms
slope_u_x = ti.field(ti.f32, grid)
slope_u_y = ti.field(ti.f32, grid)
slope_v_x = ti.field(ti.f32, grid)
slope_v_y = ti.field(ti.f32, grid)
lap = ti.Vector.field(2, ti.f32, grid) # vector laplacian ∇² of vector u == vector of scalar laplacians of u and v 
grad_div = ti.Vector.field(2, ti.f32, grid) # Gradient vector ∇(∇·u) where u is the total velocity vector.

##### Dimensions #####
dx = LENGTH/Nx # x width of each cell
dy = dx
cyl_pos = ti.Vector([0.2*LENGTH, 0.5*Ny*dy])

###### 2D-Display ######
width = 1300
colour_res = 100
height = int(Ny*width/Nx)
pixels = ti.field(ti.f32, (width, height))
cmap_name = 'viridis'#'rainbow'  # python colormap
cmap = cm.get_cmap(cmap_name)

image = ti.Vector.field(3, ti.f32, (width, height))
cmap_field = ti.Vector.field(3, ti.f32, colour_res)

colors = [(1, 1, 0), (0.953, 0.490, 0.016), (0, 0, 0), (0.176, 0.976, 0.529), (0, 1, 1)]
my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap', colors)
show_vorticity = False


@ti.kernel
def init():
    for i, j in q:
        q[i, j] = [init_rho, init_vel, 0, init_pressure]#[0.5, 3, 0, 3]
        #z = ti.Vector([i*dx-0.1, j*dy-0.3])
        #if z.norm() < 0.07:
        #    q[i, j] = [0.85, 3, 0, 2]


@ti.kernel
def cons_to_prim(U_field: ti.template()): # Either U_half_timestep or U to be converted...
    for i, j in U:
        rho, rho_u, rho_w, rho_E = U_field[i, j][0], U_field[i, j][1], U_field[i, j][2], U_field[i, j][3]
        q[i, j][0] = rho
        q[i, j][1] = rho_u / rho
        q[i, j][2] = rho_w / rho
        e = (rho_E / rho) - (q[i, j][1]**2 + q[i, j][2]**2)/2
        q[i, j][3] = (gamma-1.) * rho * e

@ti.kernel
def prim_to_cons():
    for i, j in q:
        rho, u, w, p = q[i, j][0], q[i, j][1], q[i, j][2], q[i, j][3]
        U[i, j][0] = rho
        U[i, j][1] = rho * u
        U[i, j][2] = rho * w
        e = p / (rho * (gamma-1.))
        E = e + (u**2 + w**2)/2
        U[i, j][3] = rho * E


@ti.func
def boundary_c(b, N): 
    R = b+1 # go right
    if R > N-1: R = N-1 # dont step over the grid
    L = b-1 # go left
    if L < 0: L = 0
    
    return R, L

@ti.func
def in_cylinder(i, j, radius):
    boolean_ = False
    z = ti.Vector([i*dx, j*dy]) - cyl_pos
    if z.norm() <= radius:
        boolean_ = True
    return boolean_


"""
@ti.func
def fetch_cell_p(i, j):
    momentum = ti.Vector([0., 0.])
    if in_cylinder(i, j):
        momentum = ti.Vector([0., 0.])
    elif in_cylinder(i, j) == False:
        momentum = ti.Vector([U[i, j][1], U[i, j][2]])
    return momentum

@ti.func
def reflected_momentum(i, j, vekk: ti.math.vec4):
    p = ti.Vector([vekk[1], vekk[2]])
    radius_vec = ti.Vector([i*dx, i*dy]) - cyl_pos
    norm = radius_vec / radius_vec.norm()
    radius_comp_p = norm * p.dot(norm)
    return p - 2*radius_comp_p
"""

@ti.kernel
def set_BC():
    for i, j in U:
        # Outflow at x_axis:
        U[0, j] = U[2, j]
        U[1, j] = U[2, j]
        U[Nx-1, j] = U[Nx-3, j]
        U[Nx-2, j] = U[Nx-3, j]

        # Outflow at y axis
        U[i, 0] = U[i, 2]
        U[i, 1] = U[i, 2]
        U[i, Ny-1] = U[i, Ny-3]
        U[i, Ny-2] = U[i, Ny-3]

        if in_cylinder(i, j, Rad_Cylinder):
            """
            vec_four = ti.Vector([1., 0., 0., 1.])
            outermost, outer, inner, innermost = vec_four, vec_four, vec_four, vec_four
            if in_cylinder(i, j, Rad_Cylinder-dx):
                if in_cylinder(i, j, Rad_Cylinder-2*dx):
                    if in_cylinder(i, j, Rad_Cylinder-3*dx):
                        if in_cylinder(i, j, Rad_Cylinder-4*dx) == False:
                            innermost = U[i, j]
                    else:
                        inner = U[i, j]
                else: 
                    outer = U[i, j]
            else:
                outermost = U[i, j]

            innermost = outermost
            inner = outer

            innermost[1] = - outermost[1]
            innermost[2] = - outermost[2]
            inner[1] = - outer[1]
            inner[2] = - outer[2]
            if in_cylinder(i, j, Rad_Cylinder-dx):
                if in_cylinder(i, j, Rad_Cylinder-2*dx):
                    if in_cylinder(i, j, Rad_Cylinder-3*dx):
                        if in_cylinder(i, j, Rad_Cylinder-4*dx) == False:
                            U[i, j] = innermost
                        else:
                            U[i, j] = [np.NAN, 0., 0., 0.]
                    else:
                        U[i, j] = inner
                else:
                    U[i, j] = outer
            else: 
                U[i, j] = outermost
            """
            #"""
            if in_cylinder(i, j, Rad_Cylinder-dx) == False: # <------ NOTE: Please fix the incorrect and non-working BC around the sphere!!!
                if i*dx <= cyl_pos.x:
                    U[i, j] = U[i-3, j]
                    U[i-1, j] = U[i-2, j]

                    U[i, j][1] = -U[i-3, j][1]
                    U[i-1, j][1] = -U[i-2, j][1]

                    U[i, j][2] = -U[i-3, j][2]
                    U[i-1, j][2] = -U[i-2, j][2]
                if i*dx > cyl_pos.x: # <------ NOTE: Please fix the incorrect and non-working BC around the sphere!!!
                    U[i, j] = U[i+3, j]
                    U[i+1, j] = U[i+2, j]

                    U[i, j][1] = -U[i+3, j][1]
                    U[i+1, j][1] = -U[i+2, j][1]

                    U[i, j][2] = -U[i+3, j][2]
                    U[i+1, j][2] = -U[i+2, j][2]
            #"""


            """
            p = fetch_cell_p(i-1, j-1) + fetch_cell_p(i, j-1) + fetch_cell_p(i+1, j-1) \
              + fetch_cell_p(i-1, j)   + fetch_cell_p(i, j)   + fetch_cell_p(i+1, j) \
              + fetch_cell_p(i-1, j+1) + fetch_cell_p(i, j+1) + fetch_cell_p(i+1, j+1)
            live_cells = 0.
            for c in ti.static(range(3)):
                for d in ti.static(range(3)):
                    x = i + (c-1)
                    y = j + (d-1)
                    ti.atomic_add(live_cells, in_cylinder(x, y))
            tot_p = p / live_cells
            radius_vec = ti.Vector([i*dx, i*dy]) - cyl_pos
            norm = radius_vec / radius_vec.norm()
            radius_comp_p = norm * tot_p.dot(norm)
            """

            #U[i, j][1] = 0. #tot_p.x - 2*radius_comp_p.x #-p.x/live_cells   <------ NOTE: Please fix the incorrect and non-working BC around the sphere!!!
            #U[i, j][2] = 0. #tot_p.y - 2*radius_comp_p.y #-p.y/live_cells   <-------|

@ti.kernel
def calc_dt():
    dt[None] = 1.0e5 # arbitrarily large number
    for i, j in q:
        rho, u, w, p = q[i, j][0], q[i, j][1], q[i, j][2], q[i, j][3]
        c = ti.sqrt(gamma * p / rho)
        welp_x = C * dx / (ti.abs(u) + c)
        welp_y = C * dy / (ti.abs(w) + c)
        welp = ti.min(welp_x, welp_y)
        ti.atomic_min(dt[None], welp)
    # Stores new timestep value in dt[None]

@ti.func
def scalar_minmod(å, ö):
    Δ = 0.0
    if ti.abs(å) < ti.abs(ö) and å*ö > 0:
        Δ = å
    if ti.abs(å) > ti.abs(ö) and å*ö > 0:
        Δ = ö
    return Δ

@ti.func
def vector_minmod(a, b, out_type): # Minmode slope limiter!
    svar = out_type
    for k in ti.static(range(4)):
        å = a[k]
        ö = b[k]
        svar[k] = scalar_minmod(å, ö)
    return svar

@ti.func
def scalar_slope(Axis, f, i, j):
    slope = 0.
    if Axis == 0:
        R, L = boundary_c(i, Nx)
        a, b = f[i, j] - f[L, j], f[R, j] - f[i, j]
        slope = scalar_minmod(a, b)/dx
    if Axis == 1: 
        R, L = boundary_c(j, Ny)
        a, b = f[i, j] - f[i, L], f[i, R] - f[i, j]
        slope = scalar_minmod(a, b)/dy
    return slope

@ti.func
def scalar_laplacian(slope_f_x: ti.template(), slope_f_y: ti.template(), i, j): # Specific to cartesian coordinates -- laplacian of a scalar quantity f
    second_div_x = scalar_slope(0, slope_f_x, i, j)
    second_div_y = scalar_slope(1, slope_f_y, i, j)
    scalar_lap_f = second_div_x + second_div_y
    return scalar_lap_f

@ti.func
def laplace():
    for i, j in q: # Initial spatial derivatives
        slope_u_x[i, j] = Δq_x[i, j][1]
        slope_u_y[i, j] = Δq_y[i, j][1]
        slope_v_x[i, j] = Δq_x[i, j][2]
        slope_v_y[i, j] = Δq_y[i, j][2]
    for i, j in q: # Laplacian using said derivatives
        lap_u = scalar_laplacian(slope_u_x, slope_u_y, i, j)
        lap_v = scalar_laplacian(slope_v_x, slope_v_y, i, j)
        lap[i, j] = [lap_u, lap_v] # <----- Slope limited vector laplacian
        

@ti.func
def gradients_and_divergence():
    for i, j in q:
        second_derivative_u_x = scalar_slope(0, slope_u_x, i, j)
        second_derivative_v_y = scalar_slope(1, slope_v_y, i, j)
        derivative_x_of_v_y = scalar_slope(0, slope_v_y, i, j)
        derivative_y_of_u_x = scalar_slope(1, slope_u_x, i, j)
        
        grad_div_x = second_derivative_u_x + derivative_x_of_v_y
        grad_div_y = derivative_y_of_u_x + second_derivative_v_y
    
        grad_div[i, j] = [grad_div_x, grad_div_y]


@ti.kernel
def all(result_div: ti.template()):
    for i, j in q:
        vec_four = ti.Vector([0., 0., 0., 0.])
        R, L = boundary_c(i, Nx)
        a, b = q[i, j] - q[L, j], q[R, j] - q[i, j]
        Δq_x[i, j] = vector_minmod(a, b, vec_four)/dx

        R, L = boundary_c(j, Ny)
        a, b = q[i, j] - q[i, L], q[i, R] - q[i, j]
        Δq_y[i, j] = vector_minmod(a, b, vec_four)/dy

    laplace()
    gradients_and_divergence()
    for i, j in q:
        dt = dt[None]
        # calculate faces (left and right)
        R, L = boundary_c(i, Nx)
        x_L[i, j] = q[L, j] + (dx/2)*Δq_x[L, j]
        x_R[i, j] = q[i, j] - (dx/2)*Δq_x[i, j]
        R, L = boundary_c(j, Ny)
        y_L[i, j] = q[i, L] + (dy/2)*Δq_y[i, L]
        y_R[i, j] = q[i, j] - (dy/2)*Δq_y[i, j]
    
    for i, j in q:
        # Left and right states along each axis
        X_Left, X_Right = x_L[i, j], x_R[i, j]
        Y_Left, Y_Right = y_L[i, j], y_R[i, j]

        # radius axis flux and azimuth axis/plane flux:
        x_flux[i, j] = flu.rusanov_flux(0, gamma, X_Left, X_Right)
        y_flux[i, j] = flu.rusanov_flux(1, gamma, Y_Left, Y_Right)
        
    for i, j in q:
        # x terms for the divergence:
        R, L = boundary_c(i, Nx)
        x_terms = (x_flux[R, j] - x_flux[i, j]) / dx # <---- is actually a vector with 4 elements

        # y terms for the divergence:
        R, L = boundary_c(j, Ny)
        y_terms = (y_flux[i, R] - y_flux[i, j]) / dy # <---- 4-vector of y-terms for the divergence

        viscous_force_x = μ*lap[i, j].x + (μ/3)*grad_div[i, j].x
        viscous_force_y = μ*lap[i, j].y + (μ/3)*grad_div[i, j].y

        # Complete time derivatives for conserved variables
        result_div[i, j][0] = - (x_terms[0] + y_terms[0])
        result_div[i, j][1] = - (x_terms[1] + y_terms[1]) + viscous_force_x
        result_div[i, j][2] = - (x_terms[2] + y_terms[2]) + viscous_force_y
        result_div[i, j][3] = - (x_terms[3] + y_terms[3])


@ti.kernel
def euler_add(a_timestep: ti.template(), b: ti.template(), timestep: float, dev_n: ti.template()):
    for i, j in b:
        a_timestep[i, j] = b[i, j] + timestep * dev_n[i, j]

@ti.kernel
def paint(divide_by: float):
    for i, j in pixels:
        x = ti.round(Nx * i/width, dtype=int)
        y = ti.round(Ny * j/height, dtype=int)

        if show_vorticity:
            vorticity = Δq_x[x, y][2] - Δq_y[x, y][1]
            pixels[i, j] = vorticity/divide_by 
        else: 
            vel_magnitude = ti.sqrt(q[x, y][1]**2 + q[x, y][2]**2)
            density = q[x, y][0]
            pixels[i, j] = density/divide_by

@ti.kernel
def what():
    for i, j in image:
        d = ti.round(pixels[i, j] * colour_res, dtype=ti.i32)
        if d >= colour_res: d = colour_res-1 
        # because regions with value 1 is just painted black, reason being that d is 
        # then d=colour_res, but there is only indices up to (colour_res-1) in the cmap_field
        image[i, j] = cmap_field[d]

copy_vortic = ti.Vector.field(4, ti.f32, (width, height))

@ti.kernel
def vortic_to_field():
    for i, j in copy_vortic:
        image[i, j] = [copy_vortic[i, j][0], copy_vortic[i, j][1], copy_vortic[i, j][2]]


window = ti.ui.Window('Compressible, viscous', (width, height), pos = (150, 150))
canvas = window.get_canvas()
gui = window.get_gui()

init()
precomp.set_cmap_field(colour_res, cmap, cmap_field)
prim_to_cons()
t = 0.
old_value = init_rho

while window.running:
    global z_value
    set_BC()

    # NOTE: Integrate for half timestep!
    cons_to_prim(U) # From U^n to q^n
    calc_dt()
    all(div_t) # get time-slope at U^n [using q^n]
    euler_add(U_half_timestep, U, dt[None]/2, div_t) # Euler step to half timestep

    # NOTE: Integrate for whole timestep!
    cons_to_prim(U_half_timestep) # From U^n+½ to q^n+½
    all(div_t) # get slope at U^n+½ (half timestep) [using q^n+½]
    euler_add(U, U, dt[None], div_t) # Back to U^n and take Euler step to full timestep (with slope from n+½)

    with gui.sub_window('Menu', x=0.2, y=0.2, width=0.2, height=0.2):
        gui.text(f't = {t:.3}')
        is_clicked = gui.button('name')
        z_value = gui.slider_float('Display tolerance', old_value, minimum=0.5, maximum=20)
        old_value = z_value

    paint(z_value)
    
    if show_vorticity:
        # Vorticity display method taken from https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/karman_vortex_street.py
        vor_img = cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=-6.02, vmax=6.02), cmap=my_cmap).to_rgba(pixels.to_numpy())
        copy_vortic.from_numpy(vor_img)
        vortic_to_field()
        canvas.set_image(image)
    else:
        what()
        canvas.set_image(image)

    window.show()
    t += dt[None]