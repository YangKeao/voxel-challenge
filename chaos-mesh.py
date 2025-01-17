from scene import Scene
import taichi as ti
from taichi.math import *

dark_mode = True
scene = Scene(voxel_edges=0.04 if not dark_mode else 0, exposure=1.5)
scene.set_floor(-60, (1.0, 1.0, 1.0))
if dark_mode:
    scene.set_background_color(vec3(25, 41, 79)/256)
else:
    scene.set_background_color(vec3(1.0, 1.0, 1.0))
scene.set_directional_light((1, 1, 1), 0.1, (1, 1, 1))
icon_center = vec3(0, 0, 0)
icon_size = 128
line_color = vec3(23, 45, 114)/256 if not dark_mode else vec3(1, 1, 1)
line_radius = 3
shs, svs = 8.0, 6.0
pink_sphere_color = vec3(242, 92, 124) / 256 if not dark_mode else vec3(224, 105, 127) / 256
blue_sphere_color = vec3(16, 166, 250) / 256 if not dark_mode else vec3(60, 160, 241)/256
whilte_sphere_color = vec3(1, 1, 1)
margin = 24
sphere_radius, big_sphere_radius, disc_extra_radius = 7,10,1.5
disc_z_range = 4

@ti.func
def draw_sphere(center,radius,color,empty=False):
    sophistication = 2
    big_radius = int(sophistication * radius)
    for i, j, k in ti.ndrange((-big_radius, big_radius), (-big_radius, big_radius), (-big_radius, big_radius)):
        x = vec3(i, j, k) / sophistication
        if x.dot(x) < radius * radius:
            if not empty:
                scene.set_voxel(center + x, 1, color)
            else:
                scene.set_voxel(center + x, 0, color)

@ti.func
def draw_disc(center, radius, color, z_range):
    sophistication = 2
    big_radius = int(sophistication * radius)
    big_z = sophistication * z_range
    for i, j, k in ti.ndrange((-big_radius, big_radius), (-big_radius, big_radius), (-big_z, big_z)):
        x = vec3(i, j, k) / sophistication
        if x.dot(x) < radius * radius:
            scene.set_voxel(center + x, 1, color)

@ti.func
def draw_sin_curve(translation,r_axis,r_angle,color,period,phase,length,scale,radius,skip=vec2(0.0, 0.0)):
    sophistication = 50
    for big_x in range(length * sophistication):
        x = big_x / sophistication
        if not (x > skip[0] and x < skip[1]):
            y = scale * ti.sin(x / period * 2 * pi + phase)
            rotated = rotate3d(vec3(x, y, 0), r_axis, r_angle)
            draw_sphere(translation + rotated, radius, color)

@ti.kernel
def initialize_voxels():
    for line_y in ti.static(range(3)):
        translation=vec3(-icon_size/2,line_y*icon_size/2-icon_size/2+shs/2-(line_y-1)*margin,0)+icon_center
        skip=vec2(0.0, 0.0)
        if line_y == 2:
            skip=vec2(icon_size/2-30, icon_size/2-20)
        if line_y == 0:
            skip=vec2(icon_size/2+24, icon_size/2+34)
        draw_sin_curve(translation, vec3(1, 0, 0), 0.0, line_color,
                       icon_size * 2, pi, icon_size, shs, line_radius,skip)
    for line_x in ti.static(range(3)):
        translation=vec3(line_x*icon_size/2-icon_size/2+svs/2-(line_x-1)*margin,-icon_size/2,0)+icon_center
        draw_sin_curve(translation, vec3(0, 0, 1), pi / 2, line_color,
                       icon_size, pi, icon_size, svs, line_radius)
    for x,y in ti.static(ti.ndrange((0,3), (0,3))):
        pos_x = -icon_size/2+margin+x*(icon_size-2*margin)/2
        pos_x -= svs*ti.sin((y*(icon_size-2*margin)/2+margin)/icon_size*2*pi+pi)-svs/2
        pos_y = -icon_size / 2 + margin + y * (icon_size - 2*margin)/2
        pos_y += shs*ti.sin((x*(icon_size-2*margin)/2+margin)/icon_size*pi+pi)+shs/2
        if x == 1 and y == 1:
            draw_disc(vec3(pos_x, pos_y, 0),big_sphere_radius+disc_extra_radius,line_color,disc_z_range)
            draw_sphere(vec3(pos_x, pos_y, 0),big_sphere_radius, whilte_sphere_color,empty=dark_mode)
        elif x == 1 or y == 1:
            draw_disc(vec3(pos_x, pos_y, 0),sphere_radius+disc_extra_radius,line_color,disc_z_range)
            draw_sphere(vec3(pos_x, pos_y, 0),sphere_radius,blue_sphere_color)
        else:
            draw_disc(vec3(pos_x, pos_y, 0),sphere_radius+disc_extra_radius,line_color,disc_z_range)
            draw_sphere(vec3(pos_x, pos_y, 0),sphere_radius,pink_sphere_color)

initialize_voxels()
scene.finish()
