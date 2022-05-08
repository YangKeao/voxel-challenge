from functools import total_ordering
import taichi as ti
from taichi.lang.kernel_impl import kernel
from taichi.math import *

ti.init(arch=ti.cpu)

radius = 512
sphere_bias = ti.sqrt(3)

point_num = 1024 * 1024 * 4
point = ti.Vector.field(3, dtype=ti.f32, shape=point_num)
step_length = 0.05

pixels = ti.field(dtype=float, shape=(radius*2, radius*2))

total_period = 120
step_period = total_period / 4

@ti.func
def step(p, sigma, time):
    if time >= 0 and time < step_period:
        diff = time

        x_ = -(sigma * p.x * p.y * p.y) * step_length
        y_ = (sigma * p.x * p.x * p.y) * step_length
        p.x += x_
        p.y += y_
        p = p.normalized()

    if time >= step_period and time < step_period * 2:
        diff = time - step_period

        p = rotate3d(p, vec3(0, 0, 1), (p.z - (-ti.sqrt(2)/2)) / ti.sqrt(2) * pi * diff / step_period)

    if time >= step_period * 2 and time < step_period * 3:
        diff = time - step_period * 2

        y_ = sigma * p.y * p.z * p.z * step_length
        z_ = -sigma * p.y * p.y * p.z * step_length
        p.y += y_
        p.z += z_
        p = p.normalized()

    if time >= step_period * 3 and time < step_period * 4:
        diff = time - step_period * 3
        p = rotate3d(p, vec3(1, 0, 0), (p.x - (-ti.sqrt(2)/2)) / ti.sqrt(2) * pi * diff / step_period)
    
    return p

@ti.kernel
def random_points():
    for i in range(point_num):
        point[i].x = (ti.random() - 0.5) * 2
        point[i].y = (ti.random() - 0.5) * 2
        point[i].z = (ti.random() - 0.5) * 2
        point[i] = point[i].normalized()

@ti.func
def render_point(p):
    p_3d = p * radius
    p_3d = rotate3d(p_3d, vec3(0, 0, 1), -pi / 4)
    p_3d = rotate3d(p_3d, vec3(0, 1, 0), pi / 4)

    p_3d.yz += radius
    if p_3d.x < 0:
        pixels[p_3d.y, p_3d.z] = 0.6
    else:
        pixels[p_3d.y, p_3d.z] = 1

@ti.kernel
def paint(time: ti.i32):
    for i, j in ti.ndrange((-radius, radius), (-radius, radius)):
        pixels[i + radius, j + radius] = 0

    for i in range(point_num):
        p = step(point[i], 1.0, time % total_period)
        if time % step_period == step_period - 1 or time % (2 * step_period) < step_period:
            point[i] = p
        render_point(p)

random_points()

gui = ti.GUI("Plykin Attractors", res=(radius*2, radius*2))

i = 0
for _ in range(2000):
    paint(i)
    i = i + 1
    if i%10 == 0:
        print("precalculate: ", i)

while gui.running:
    paint(i)
    gui.set_image(pixels)
    gui.show()
    i = i + 1