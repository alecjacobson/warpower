"""UNTESTED private #154 experiment: block-reduce scalar diagnostics only.

Launch with a sample dimension rounded UP to the block size, so every lane
reaches wp.tile. Preserve all original per-site atomics and nearest-site logic.
This is not imported by the library or part of PR153.
"""
import warp as wp
from warpower import _bvh
from warpower._site_bvh import nearest_site
wp.set_module_options({'enable_backward': False, 'fast_math': False})

@wp.kernel
def accumulate_samples_reduced(
    tree: _bvh.Tree,
    points: wp.array[wp.vec3],
    active: wp.array[int],
    occupied: wp.array[int],
    samples: wp.array[wp.vec3],
    base_weights: wp.array[wp.float64],
    sizing: wp.array[float],
    dimension: int,
    owners: wp.array[int],
    mass: wp.array[wp.float64],
    moment: wp.array[wp.vec3d],
    energies: wp.array[wp.float64],
    total: wp.array[wp.float64],
    maximum: wp.array[wp.float64],
    component: int,
    error: wp.array[int],
):
    s = wp.tid()
    energy_value = wp.float64(0.0)
    total_value = wp.float64(0.0)
    distance_value = wp.float64(0.0)
    if s < samples.shape[0]:
        owners[s] = -1
        if error[0] == 0 and base_weights[s] != wp.float64(0.0):
            h = wp.float64(sizing[s])
            if h <= wp.float64(0.0) or not wp.isfinite(h):
                wp.atomic_or(error, 0, 8)
            else:
                denominator = h * h * h
                if dimension == 2:
                    denominator *= h
                weight = base_weights[s] / denominator
                sample = wp.vec3d(samples[s])
                hit = nearest_site(tree, points, sample, active, occupied, False)
                if hit.index < 0 or not wp.isfinite(weight):
                    wp.atomic_or(error, 0, 16)
                else:
                    owners[s] = hit.index
                    delta = sample - wp.vec3d(points[hit.index])
                    wp.atomic_add(mass, hit.index, weight)
                    wp.atomic_add(moment, hit.index, weight * delta)
                    energy_value = weight * hit.distance2
                    total_value = weight
                    distance_value = wp.sqrt(hit.distance2)
    energy_sum = wp.tile_sum(wp.tile(energy_value))[0]
    total_sum = wp.tile_sum(wp.tile(total_value))[0]
    distance_max = wp.tile_max(wp.tile(distance_value))[0]
    if s % wp.block_dim() == 0:
        wp.atomic_add(energies, component, energy_sum)
        wp.atomic_add(total, component, total_sum)
        wp.atomic_max(maximum, component, distance_max)
