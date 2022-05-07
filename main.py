from scene import Scene
import taichi as ti
from taichi.math import *

scene = Scene(voxel_edges=0, exposure=1.3)
scene.set_background_color(vec3(1.0, 1.0, 1.0))
scene.set_directional_light((1, 1, 1), 0.1, (1, 1, 1))
scene.set_floor(-0.5, vec3(255, 211, 186) / 256)
scene.set_floor(-0.5, vec3(1, 1, 1))

radius = 32
color = vec3(22, 152, 115) / 256
inner_radius = 26
inner_color = vec3(87, 98, 213) / 256
sphere_bias = ti.sqrt(3)
sigma = 1.0
attractor_attempt = 4096
max_count = 100
gc_radius = 1
garbage_threshold = 2

@ti.func
def f(p, sign, sigma):
    x = sign * p.z
    y = ti.sqrt(p.xy.dot(p.xy))*(
        p.y * ti.exp(sigma / 2 * (p.xy.dot(p.xy))) * ti.cos(pi / 2 * (p.z * ti.sqrt(2) + 1))
        + sign * p.x * ti.exp(-sigma/2 * (p.xy.dot(p.xy))) * ti.sin(pi / 2 * (p.z * ti.sqrt(2) + 1))
    ) / ti.sqrt(
        p.x * p.x * ti.exp(-sigma * p.xy.dot(p.xy)) + p.y * p.y * ti.exp(sigma * p.xy.dot(p.xy))
    )
    z = ti.sqrt(p.xy.dot(p.xy))*(
        p.y * ti.exp(sigma / 2 * (p.xy.dot(p.xy))) * ti.sin(pi / 2 * (p.z * ti.sqrt(2) + 1))
        - sign * p.x * ti.exp(-sigma/2 * (p.xy.dot(p.xy))) * ti.cos(pi / 2 * (p.z * ti.sqrt(2) + 1))
    ) / ti.sqrt(
        p.x * p.x * ti.exp(-sigma * p.xy.dot(p.xy)) + p.y * p.y * ti.exp(sigma * p.xy.dot(p.xy))
    )
    return vec3(x, y, z)

@ti.func
def step(p, sigma):
    negative = f(p, -1.0, sigma)
    positive = f(negative, 1.0, sigma)
    return positive

@ti.kernel
def initialize_voxels():
    for i, j, k in ti.ndrange((-radius, radius), (-radius, radius), (-radius, radius)):
        big_p = vec3(i,j,k)
        p = vec3(i/radius, j/radius, k/radius)
        if ti.sqrt(big_p.dot(big_p)) - radius > -sphere_bias and ti.sqrt(big_p.dot(big_p)) - radius < 0:
            ti.loop_config(serialize=True)
            for _ in range(attractor_attempt):
                p = step(p, sigma)

            scene.set_voxel(p.normalized() * radius, 1, color)

    for i, j, k in ti.ndrange((-inner_radius, inner_radius), (-inner_radius, inner_radius), (-inner_radius, inner_radius)):
        big_p = vec3(i,j,k)
        p = vec3(i/radius, j/radius, k/radius)
        if ti.sqrt(big_p.dot(big_p)) - inner_radius > -sphere_bias and ti.sqrt(big_p.dot(big_p)) - inner_radius < 0:
            scene.set_voxel(big_p, 1, inner_color)


initialize_voxels()
scene.finish()
