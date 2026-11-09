"""Opt-in FP32 nearest-site queries; moments, energy and topology stay FP64."""

import warp as wp

from . import _bvh

wp.set_module_options({"enable_backward": False, "fast_math": False})


@wp.struct
class SiteHit32:
    index: int
    distance2: wp.float32


@wp.func(inline=True)
def nearest_site32(
    tree: _bvh.Tree,
    points: wp.array[wp.vec3],
    p: wp.vec3,
    active: wp.array[int],
    occupied: wp.array[int],
    only_free: bool,
):
    result = SiteHit32()
    result.index = -1
    result.distance2 = wp.float32(3.4028234663852886e38)
    stack = wp.zeros(shape=128, dtype=int)
    stack[0] = 1
    size = int(1)
    while size > 0:
        size -= 1
        node = stack[size]
        delta = wp.max(
            wp.max(wp.vec3(tree.lower[node]) - p, p - wp.vec3(tree.upper[node])), wp.vec3(0.0)
        )
        if wp.dot(delta, delta) > result.distance2:
            continue
        if tree.count[node] > 0:
            for k in range(tree.start[node], tree.start[node] + tree.count[node]):
                i = tree.ids[k]
                if i >= active[0]:
                    continue
                if only_free and occupied[i] >= 0:
                    continue
                distance = wp.length_sq(wp.vec3(points[i]) - p)
                if distance < result.distance2 or (
                    distance == result.distance2 and i < result.index
                ):
                    result.index = i
                    result.distance2 = distance
        else:
            left, right = int(tree.left[node]), int(tree.right[node])
            dl = wp.max(
                wp.max(wp.vec3(tree.lower[left]) - p, p - wp.vec3(tree.upper[left])),
                wp.vec3(0.0),
            )
            dr = wp.max(
                wp.max(wp.vec3(tree.lower[right]) - p, p - wp.vec3(tree.upper[right])),
                wp.vec3(0.0),
            )
            if wp.dot(dl, dl) < wp.dot(dr, dr):
                left, right = right, left
            stack[size] = left
            stack[size + 1] = right
            size += 2
    return result


@wp.kernel
def accumulate_samples32(
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
    owners[s] = -1
    if error[0] != 0 or base_weights[s] == wp.float64(0.0):
        return
    h = wp.float64(sizing[s])
    if h <= wp.float64(0.0) or not wp.isfinite(h):
        wp.atomic_or(error, 0, 8)
        return
    denominator = h * h * h
    if dimension == 2:
        denominator *= h
    weight = base_weights[s] / denominator
    sample = wp.vec3d(samples[s])
    hit = nearest_site32(tree, points, samples[s], active, occupied, False)
    if hit.index < 0 or not wp.isfinite(weight):
        wp.atomic_or(error, 0, 16)
        return
    owners[s] = hit.index
    delta = sample - wp.vec3d(points[hit.index])
    # Recompute selected-owner distances in FP64 for acceptance diagnostics.
    distance2 = wp.dot(delta, delta)
    wp.atomic_add(mass, hit.index, weight)
    wp.atomic_add(moment, hit.index, weight * delta)
    wp.atomic_add(energies, component, weight * distance2)
    wp.atomic_add(total, component, weight)
    wp.atomic_max(maximum, component, wp.sqrt(distance2))
