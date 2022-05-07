import taichi as ti
from taichi.math import *

ti.init(arch=ti.gpu)

radius = 520
sphere_bias = ti.sqrt(3)

pixels = ti.field(dtype=float, shape=(radius*2, radius*2))
attract_steps = 720

@ti.func
def f(p, sign, sigma):
    x = sign * p.z
    y = ti.sqrt(p.xy.dot(p.xy))*(
        p.y * ti.exp(sigma / 2 * (p.xy.dot(p.xy))) *
        ti.cos(pi / 2 * (p.z * ti.sqrt(2) + 1))
        + sign * p.x * ti.exp(-sigma/2 * (p.xy.dot(p.xy))) *
        ti.sin(pi / 2 * (p.z * ti.sqrt(2) + 1))
    ) / ti.sqrt(
        p.x * p.x * ti.exp(-sigma * p.xy.dot(p.xy)) + p.y *
        p.y * ti.exp(sigma * p.xy.dot(p.xy))
    )
    z = ti.sqrt(p.xy.dot(p.xy))*(
        p.y * ti.exp(sigma / 2 * (p.xy.dot(p.xy))) *
        ti.sin(pi / 2 * (p.z * ti.sqrt(2) + 1))
        - sign * p.x * ti.exp(-sigma/2 * (p.xy.dot(p.xy))) *
        ti.cos(pi / 2 * (p.z * ti.sqrt(2) + 1))
    ) / ti.sqrt(
        p.x * p.x * ti.exp(-sigma * p.xy.dot(p.xy)) + p.y *
        p.y * ti.exp(sigma * p.xy.dot(p.xy))
    )
    return vec3(x, y, z)


@ti.func
def step(p, sigma):
    negative = f(p, -1.0, sigma)
    positive = f(negative, 1.0, sigma)
    return positive

@ti.kernel
def paint():
    for i, j in ti.ndrange((-radius, radius), (-radius, radius)):
        pixels[i+radius, j+radius] = 0
        if i * i + j * j < radius * radius:
            k = ti.sqrt(radius * radius - i * i - j * j)
            p_3d = vec3(i, j, k)
            p = vec3(i, j, k)/radius
            for _ in range(attract_steps):
                p = step(p, 1.0)
            p_3d = p.normalized() * radius
            p_3d.xy += radius
            if p_3d.z < 0:
                pixels[p_3d.x, p_3d.y] = 0.2
            else:
                pixels[p_3d.x, p_3d.y] = 1

gui = ti.GUI("Plykin Attractors", res=(radius*2, radius*2))

i = 0
while gui.running:
    paint()
    gui.set_image(pixels)
    gui.show()
    i = i + 1