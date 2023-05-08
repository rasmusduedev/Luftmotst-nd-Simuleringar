import flux_2d_sphere as flu
from matplotlib import cm
import matplotlib
import taichi as ti
import precomp
import numpy as np
ti.init(ti.gpu)

# NOTE: fixed missing (1/r) factors in the gradient of the divergence
# 2D Euler equations simulation in a spherical coordinate system with 2nd order in time, 2nd order in space.
# BC [Radial]: Reflective at sphere surface and outflow at outer grid boundary
# BC [Azimuth]: Periodic at 360 degree_view


# Constants
C = 0.4e-1
gamma = 1.4
RADIUS = 0.3     #0.1
Rad_ball = 0.05  #0.04267/2 # golf ball radius
ω = 2*3.141592   # Angular velocity in rad/s
grip_const = 0.#500.#100. #50000000. # Arbitrary grip constant in 1/s (I just invented it)
μ = 1.0e-3 # dynamic viscosity

init_rho = 0.5   #1.293 # usual air density
init_fluid = -3  #-91.19616 # fastest golf hit
init_pressure = 3  #101325 # atmospheric pressure (1 bar)

# Grid
N_r = 150 # radius cells
N_φ = 200 # polar angle cells
degree_view = 360 # span of the circle, in degrees [do notice that the angle is with respect to the upper vertical line, going clockwise]

grid = (N_r, N_φ)
q = ti.Vector.field(4, ti.f32, grid) # rho, u, w, p
U = ti.Vector.field(4, ti.f32, grid) # rho, rho*u, rho*w, rho*E

Δq_r = ti.Vector.field(4, ti.f32, grid) # slope along radius
Δq_φ = ti.Vector.field(4, ti.f32, grid) # slope along phi (polar angle)

r_L = ti.Vector.field(4, ti.f32, grid) # left face along radius interface i-½
r_R = ti.Vector.field(4, ti.f32, grid) # right face along radius interface i-½
φ_L = ti.Vector.field(4, ti.f32, grid) # left face along phi interface j-½
φ_R = ti.Vector.field(4, ti.f32, grid) # right face along phi interface j-½

rad_flux = ti.Vector.field(4, ti.f32, grid) # radius flux terms for i-½
phi_flux = ti.Vector.field(4, ti.f32, grid) # phi flux terms for j-½
div_t = ti.Vector.field(4, ti.f32, grid) # time-derivative for each conserved variable

# Quality of life to make the code sligthly smaller, it stores all the terms for the final divergence equations
rad_terms = ti.field(ti.f32, 4)
phi_terms = ti.field(ti.f32, 4)

U_half_timestep = ti.Vector.field(4, ti.f32, grid)
div_t = ti.Vector.field(4, ti.f32, grid) # time-derivative for each conserved variable
dt = ti.field(ti.f32, shape=()) # For timestep variation, since velocity u now varies

#### Viscous force terms fields:
slope_u_r = ti.field(ti.f32, grid)
slope_u_φ = ti.field(ti.f32, grid)
slope_w_r = ti.field(ti.f32, grid)
slope_w_φ = ti.field(ti.f32, grid)
grad_u_term1 = ti.field(ti.f32, grid)
grad_u_term2 = ti.field(ti.f32, grid)
grad_w_term1 = ti.field(ti.f32, grid)
grad_w_term2 = ti.field(ti.f32, grid)
lap = ti.Vector.field(2, ti.f32, grid) # vector laplacian ∇² of vector u == vector of scalar laplacians of u and v 
grad_div = ti.Vector.field(2, ti.f32, grid) # Gradient vector ∇(∇·u) where u is the total velocity vector.

##### Dimensions #####
# Conversion from degrees to radians (saves on computation for later):
pi = 3.14159265359
dr = RADIUS/N_r                       # radius length of each cell
dφ = (degree_view * pi/180) / N_φ # angle φ length of each cell
fov_rad = degree_view * pi/180 # span of entire angle grid, in radians
rad_index = ti.round(N_r * (Rad_ball/RADIUS), dtype=int)  # Any index i equal to or less than 
                                                          # the rad_index is inside the sphere

###### 2D-Display ######
res = 800
colour_res = 100
zoom = 0.5 
center = ti.Vector([0.5, 0.5])
index_grid = ti.Vector.field(2, int, (res, res))
pixels = ti.field(ti.f32, (res, res))
cmap_name = 'viridis'  # python colormap
cmap = cm.get_cmap(cmap_name)
image = ti.Vector.field(3, ti.f32, (res, res))
cmap_field = ti.Vector.field(3, ti.f32, colour_res)

colors = [(1, 1, 0), (0.953, 0.490, 0.016), (0, 0, 0), (0.176, 0.976, 0.529), (0, 1, 1)]
my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap', colors)
show_vorticity = True


@ti.kernel
def init():
    for i, j in q:
        r = i/N_r
        φ = j/N_φ
    
        phi = dφ * j
        w = ti.cos(phi-(pi/2))*init_fluid
        u = ti.sin(phi-(pi/2))*init_fluid

        q[i, j] = [init_rho, u, w, init_pressure]

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
    
    if degree_view == 360 and N == N_φ: # make periodic when the grid spans 360 degrees
        R = b+1
        if R > N-1: R = 0
        L = b-1
        if L < 0: L = N-1
    return R, L

@ti.kernel
def set_BC(): # reflective boundary conditions at sphere boundary
    for i, j in U:
        r = get_radius(i)
        phi = dφ * j
        # Continue giving the left front init conditions, as if the 
        # central ball is traveling through air.
        if phi > pi/2 and phi <= (2*pi - (pi/2)):
            if r >= get_radius(N_r-5):
                rho = init_rho
                u = ti.sin(phi-(pi/2))*init_fluid
                w = ti.cos(phi-(pi/2))*init_fluid
                p = init_pressure
                e = p / (rho * (gamma-1.))
                E = e + (u**2 + w**2)/2
                
                U[i, j][0] = init_rho
                U[i, j][1] = init_rho*u
                U[i, j][2] = init_rho*w
                U[i, j][3] = rho * E

        # Make the central ball nan-valued and give it
        # reflective boundary conditions.
        
        ri = rad_index
        if i <= ri:
            U[i, j] = [np.NAN, 0., 0., 0.]

        U[ri+0, j] = U[ri+3, j]
        U[ri+1, j] = U[ri+2, j]

        # Let momentum go towards zero when it approaches the sphere 
        U[ri+0, j][1] = -U[ri+3, j][1]
        U[ri+1, j][1] = -U[ri+2, j][1]
        
        # Outflow at surface:
        U[N_r-1, j] = U[N_r-3, j]
        U[N_r-2, j] = U[N_r-3, j]

@ti.kernel
def calc_dt():
    dt[None] = 1.0e5 # arbitrarily large number
    for i, j in q:
        rho, u, w, p = q[i, j][0], q[i, j][1], q[i, j][2], q[i, j][3]
        r = get_radius(i)
        c = ti.sqrt(gamma * p / rho)
        welp_rad = C * dr / (ti.abs(u) + c)
        welp_phi = C * dφ*r / (ti.abs(w) + c)
        welp = ti.min(welp_rad, welp_phi)
        ti.atomic_min(dt[None], welp)
    # Stores new timestep value in dt[None]

@ti.func
def get_radius(i):
    return (dr/2) + (i*dr) # dont return zero for gods sake


@ti.func
def scalar_minmod(å, ö):
    Δ = 0.0
    if ti.abs(å) < ti.abs(ö) and å*ö > 0:
        Δ = å
    if ti.abs(å) > ti.abs(ö) and å*ö > 0:
        Δ = ö
    return Δ

@ti.func
def vector_minmod(a, b): # Minmode slope limiter!
    svar = ti.Vector([0., 0., 0., 0.])
    for k in ti.static(range(4)):
        å = a[k]
        ö = b[k]
        svar[k] = scalar_minmod(å, ö)
    return svar

@ti.func
def scalar_slope(Axis, f, i, j):
    slope = 0.
    if Axis == 0:
        R, L = boundary_c(i, N_r)
        a, b = f[i, j] - f[L, j], f[R, j] - f[i, j]
        slope = scalar_minmod(a, b)/dr
    if Axis == 1: 
        R, L = boundary_c(j, N_φ)
        a, b = f[i, j] - f[i, L], f[i, R] - f[i, j]
        slope = scalar_minmod(a, b)/dφ
    return slope

@ti.func
def new_laplace_and_divergence():
    for i, j in q: # Initial spatial derivatives
        slope_u_r[i, j] = Δq_r[i, j][1]
        slope_u_φ[i, j] = Δq_φ[i, j][1]
        slope_w_r[i, j] = Δq_r[i, j][2]
        slope_w_φ[i, j] = Δq_φ[i, j][2]
    for i, j in q:
        r = get_radius(i)
        u, w = q[i, j][1], q[i, j][2]

        # Second derivatives
        second_u_r = scalar_slope(0, slope_u_r, i, j)
        second_u_φ = scalar_slope(1, slope_u_φ, i, j)
        second_w_r = scalar_slope(0, slope_w_r, i, j)
        second_w_φ = scalar_slope(1, slope_w_φ, i, j)

        # Derivatives of derivatives - mixed dimensions (i.e. x-derivative of y-derivative kinda stuff)
        derivative_r_of_w_φ = scalar_slope(0, slope_w_φ, i, j)
        derivative_φ_of_u_r = scalar_slope(1, slope_u_r, i, j)
        
        #NOTE: Scalar laplacians!
        # laplacian of radial velocity:
        scalar_lap_u = second_u_r + (2/r)*slope_u_r[i, j] + (1/r**2)*second_u_φ
        # laplacian of azimuth velocity:
        scalar_lap_w = second_w_r + (2/r)*slope_w_r[i, j] + (1/r**2)*second_w_φ
        lap[i, j] = [scalar_lap_u, scalar_lap_w]

        #NOTE: Gradient of divergence of total velocity vector u!
        grad_div[i, j][0] = second_u_r + (2/r)*slope_u_r[i, j] - (2*u/(r**2)) - (1/r**2)*slope_w_φ[i, j] + (1/r)*derivative_r_of_w_φ
        grad_div[i, j][1] = (1/r**2)*second_w_φ + (2/(r**2))*slope_u_φ[i, j] + (1/r)*derivative_φ_of_u_r 



@ti.kernel
def all():
    for i, j in q:
        R, L = boundary_c(i, N_r)
        Δq_r[i, j] = vector_minmod(q[i, j] - q[L, j], q[R, j] - q[i, j])/dr # Uses minmode slope limiter
        R, L = boundary_c(j, N_φ)
        Δq_φ[i, j] = vector_minmod(q[i, j] - q[i, L], q[i, R] - q[i, j])/dφ

    #laplace()
    #gradients_and_divergence()
    new_laplace_and_divergence()
    
    for i, j in q:
        dt = dt[None]
        # calculate faces (left and right)
        R, L = boundary_c(i, N_r)
        r_L[i, j] = q[L, j] + (dr/2)*Δq_r[L, j]
        r_R[i, j] = q[i, j] - (dr/2)*Δq_r[i, j]
        R, L = boundary_c(j, N_φ)
        φ_L[i, j] = q[i, L] + (dφ/2)*Δq_φ[i, L]
        φ_R[i, j] = q[i, j] - (dφ/2)*Δq_φ[i, j]
    for i, j in q:
        # Left and right states along each axis
        radius_L, radius_R = r_L[i, j], r_R[i, j]
        phi_L ,  phi_R = φ_L[i, j], φ_R[i, j]

        # radius axis flux and azimuth axis/plane flux:
        rad_flux[i, j] = flu.riemann_flux(0, radius_L, radius_R, gamma)
        phi_flux[i, j] = flu.riemann_flux(1,    phi_L,    phi_R, gamma)

    for i, j in q:
        # REMEMBER: the flux at i-½ is represented by flux[i, j], NOT by flux[L, j]
        r = get_radius(i)
        r_minus = get_radius(i-0.5) # radius at i-½
        r_plus = get_radius(i+0.5)  # radius at i+½
        
        # Radius terms:
        R, L = boundary_c(i, N_r)
        for k in ti.static(range(4)):
            rad_terms[k]= (1/r**2) * ((r_plus**2 * rad_flux[R, j][k]) - (r_minus**2 * rad_flux[i, j][k])) / dr

        # Phi terms:
        R, L = boundary_c(j, N_φ)
        for k in ti.static(range(4)):
            phi_terms[k] = (1/r) * (phi_flux[i, R][k] - phi_flux[i, j][k]) / dφ

        rho, u, w, p = q[i, j][0], q[i, j][1], q[i, j][2], q[i, j][3]
        R, L = boundary_c(i, N_r)
        dP_dr = (q[R, j][3] - q[i, j][3])/dr # extra pressure derivative term appearing in the radial momentum

        viscous_force_r = μ*lap[i, j][0] + (μ/3)*grad_div[i, j][0]
        viscous_force_φ = μ*lap[i, j][1] + (μ/3)*grad_div[i, j][1]

        # Complete time derivatives for conserved variables
        div_t[i, j][0] = - (rad_terms[0] + phi_terms[0])
        div_t[i, j][1] = - (rad_terms[1] + phi_terms[1]) + (rho*w**2)/r - dP_dr + viscous_force_r
        div_t[i, j][2] = - (rad_terms[2] + phi_terms[2]) - (rho*u*w)/r          + viscous_force_φ
        div_t[i, j][3] = - (rad_terms[3] + phi_terms[3])

        glurn = rad_index + 3
        if i <= glurn: # Sphere drags on air closest to sphere boundary
            div_t[i, j][2] = - (rad_terms[2] + phi_terms[2]) - (rho*u*w)/r + viscous_force_φ - rho*grip_const*ω*Rad_ball


@ti.kernel
def euler_add(a_timestep: ti.template(), b: ti.template(), timestep: float, dev_n: ti.template()):
    for i, j in b:
        a_timestep[i, j] = b[i, j] + timestep * dev_n[i, j]


@ti.kernel
def paint(divide_by: float):
    for i, j in pixels:
        z = (ti.Vector([i/res, j/res]) - center) / zoom # Center the vector and scale everything
        if z.norm() * zoom < 0.5:
            r, t = index_grid[i, j] # use precomputed cell indexes

            if show_vorticity:
                radius = get_radius(r)
                vorticity = (1/radius)*Δq_φ[r, t][1] - (Δq_r[r, t][2] + q[r, t][2]/radius)
                pixels[i, j] = vorticity/divide_by
            else: 
                vel_magnitude = ti.sqrt(q[r, t][1]**2 + q[r, t][2]**2)
                density = q[r, t][0]
                pixels[i, j] = density/divide_by
        else:
            pixels[i, j] = 0 

@ti.kernel
def what():
    for i, j in image:
        d = ti.round(pixels[i, j] * colour_res, dtype=ti.i32)
        if d >= colour_res: d = colour_res-1 
        # because regions with value 1 is just painted black, reason being that d is 
        # then d=colour_res, but there is only indices up to (colour_res-1) in the cmap_field
        image[i, j] = cmap_field[d]

copy_vortic = ti.Vector.field(4, ti.f32, (res, res))

@ti.kernel
def vortic_to_field():
    for i, j in copy_vortic:
        image[i, j] = [copy_vortic[i, j][0], copy_vortic[i, j][1], copy_vortic[i, j][2]]


window = ti.ui.Window('Compressible, viscous', (res, res), pos = (150, 150))
canvas = window.get_canvas()
gui = window.get_gui()

init()
precomp.set_cmap_field(colour_res, cmap, cmap_field)
precomp.assign_grid_indexes(index_grid, center, zoom, res, N_r, N_φ, dφ, fov_rad)
prim_to_cons()
t = 0.
old_value = init_rho

while window.running:
    global z_value
    set_BC()

    # NOTE: Integrate for half timestep!
    cons_to_prim(U) # From U^n to q^n
    calc_dt()
    all() # get time-slope at U^n [using q^n]
    euler_add(U_half_timestep, U, dt[None]/2, div_t) # Euler step to half timestep

    # NOTE: Integrate for whole timestep!
    cons_to_prim(U_half_timestep) # From U^n+½ to q^n+½
    all() # get slope at U^n+½ (half timestep) [using q^n+½]
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