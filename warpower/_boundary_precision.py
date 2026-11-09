"""Opt-in query and boundary accumulation precision; topology stays unchanged."""

from functools import lru_cache

import warp as wp

from . import _bvh
from ._site_bvh import nearest_site

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


@lru_cache(maxsize=None)
def make_accumulate(query_precision, accumulation_precision):
    query = nearest_site32 if query_precision == "float32" else nearest_site
    query_vector = wp.vec3 if query_precision == "float32" else wp.vec3d
    scalar = wp.float32 if accumulation_precision == "float32" else wp.float64
    vector = wp.vec3 if accumulation_precision == "float32" else wp.vec3d

    @wp.kernel(enable_backward=False, module="unique", module_options={"fast_math": False})
    def accumulate(
        tree: _bvh.Tree,
        points: wp.array[wp.vec3],
        active: wp.array[int],
        occupied: wp.array[int],
        samples: wp.array[wp.vec3],
        base_weights: wp.array[wp.float64],
        sizing: wp.array[float],
        dimension: int,
        owners: wp.array[int],
        mass: wp.array[scalar],
        moment: wp.array[vector],
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
        hit = query(tree, points, query_vector(samples[s]), active, occupied, False)
        if hit.index < 0 or not wp.isfinite(weight):
            wp.atomic_or(error, 0, 16)
            return
        local_weight = scalar(weight)
        local_moment = local_weight * (vector(samples[s]) - vector(points[hit.index]))
        if local_weight <= scalar(0.0) or not wp.isfinite(local_weight):
            wp.atomic_or(error, 0, 16)
            return
        for k in range(3):
            if not wp.isfinite(local_moment[k]):
                wp.atomic_or(error, 0, 16)
                return
        owners[s] = hit.index
        wp.atomic_add(mass, hit.index, local_weight)
        wp.atomic_add(moment, hit.index, local_moment)
        # Keep acceptance diagnostics in FP64 even for FP32 nearest-site queries.
        # The selected owner may differ; no reference query is hidden here.
        delta = wp.vec3d(samples[s]) - wp.vec3d(points[hit.index])
        distance2 = wp.dot(delta, delta)
        wp.atomic_add(energies, component, weight * distance2)
        wp.atomic_add(total, component, weight)
        wp.atomic_max(maximum, component, wp.sqrt(distance2))

    return accumulate


@wp.kernel
def publish(
    mass: wp.array[float],
    moment: wp.array[wp.vec3],
    output_mass: wp.array[wp.float64],
    output_moment: wp.array[wp.vec3d],
    error: wp.array[int],
):
    i = wp.tid()
    m, v = mass[i], moment[i]
    if (
        not wp.isfinite(m)
        or not wp.isfinite(v[0])
        or not wp.isfinite(v[1])
        or not wp.isfinite(v[2])
    ):
        wp.atomic_or(error, 0, 16)
    output_mass[i] = wp.float64(m)
    output_moment[i] = wp.vec3d(v)
