import warp as wp
from warpower import _bvh
from warpower._site_bvh import nearest_site
wp.set_module_options({"enable_backward":False,"fast_math":False})

@wp.func(inline=True)
def contribution(tree: _bvh.Tree, points: wp.array[wp.vec3], active: wp.array[int], occupied: wp.array[int], samples: wp.array[wp.vec3], base_weights: wp.array[wp.float64], sizing: wp.array[float], dimension: int, owners: wp.array[int], mass: wp.array[wp.float64], moment: wp.array[wp.vec3d], energies: wp.array[wp.float64], total: wp.array[wp.float64], maximum: wp.array[wp.float64], component: int, error: wp.array[int], s: int):
    owners[s] = -1
    if error[0] != 0 or base_weights[s] == wp.float64(0.0):
        return wp.vec3d(0.0)
    h = wp.float64(sizing[s])
    if h <= wp.float64(0.0) or not wp.isfinite(h):
        wp.atomic_or(error, 0, 8)
        return wp.vec3d(0.0)
    denominator = h * h * h
    if dimension == 2:
        denominator *= h
    weight = base_weights[s] / denominator
    sample = wp.vec3d(samples[s])
    hit = nearest_site(tree, points, sample, active, occupied, False)
    if hit.index < 0 or not wp.isfinite(weight):
        wp.atomic_or(error, 0, 16)
        return wp.vec3d(0.0)
    owners[s] = hit.index
    delta = sample - wp.vec3d(points[hit.index])
    wp.atomic_add(mass, hit.index, weight)
    wp.atomic_add(moment, hit.index, weight * delta)
    return wp.vec3d(weight * hit.distance2, weight, wp.sqrt(hit.distance2))

@wp.kernel
def accumulate_reduced(tree: _bvh.Tree, points: wp.array[wp.vec3], active: wp.array[int], occupied: wp.array[int], samples: wp.array[wp.vec3], base_weights: wp.array[wp.float64], sizing: wp.array[float], dimension: int, owners: wp.array[int], mass: wp.array[wp.float64], moment: wp.array[wp.vec3d], energies: wp.array[wp.float64], total: wp.array[wp.float64], maximum: wp.array[wp.float64], component: int, error: wp.array[int]):
    s = wp.tid()
    values = contribution(tree, points, active, occupied, samples, base_weights, sizing, dimension, owners, mass, moment, energies, total, maximum, component, error, s)
    energy_tile = wp.tile_sum(wp.tile(values[0]))
    weight_tile = wp.tile_sum(wp.tile(values[1]))
    maximum_tile = wp.tile_max(wp.tile(values[2]))
    maximum_value = maximum_tile[0]
    wp.tile_atomic_add(energies, energy_tile, offset=(component,))
    wp.tile_atomic_add(total, weight_tile, offset=(component,))
    if s % 32 == 0:
        wp.atomic_max(maximum, component, maximum_value)
