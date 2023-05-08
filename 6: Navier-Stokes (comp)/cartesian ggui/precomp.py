import taichi as ti

# Dont make it a ti.kernel; cmap doesn't work in taichi scope
def set_cmap_field(colour_res, cmap, cmap_field: ti.template()):
    for i in range(colour_res):
        scale = i / colour_res
        g = cmap(scale)
        f = ti.Vector([g[0], g[1], g[2]]) # we only want the RGB in the RGBA that cmap outputs
        cmap_field[i] = f