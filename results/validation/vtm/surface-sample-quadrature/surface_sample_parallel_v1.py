"""Private per-sample quadrature and instrumented arithmetic references; not adopted."""
import warp as wp
from warpower._surface_bvh import SurfaceTree, Closest, closest_point
wp.set_module_options({"enable_backward": False, "fast_math": False})

@wp.kernel
def surface_error_samples(tree: SurfaceTree, points: wp.array[wp.vec3], faces: wp.array2d[int], count: wp.array[int], integral: wp.array[wp.float64], maximum: wp.array[wp.float64], error: wp.array[int]):
    i = wp.tid() // 3
    k = wp.tid() % 3
    if i >= count[0] or error[0] != 0:
        return
    a = wp.vec3d(points[faces[i, 0]])
    b = wp.vec3d(points[faces[i, 1]])
    c = wp.vec3d(points[faces[i, 2]])
    weight = wp.length(wp.cross(b - a, c - a)) / wp.float64(6.0)
    dominant = a
    if k == 1:
        dominant = b
    elif k == 2:
        dominant = c
    sample = (a + b + c) / wp.float64(6.0) + dominant / wp.float64(2.0)
    hit = closest_point(tree, sample, wp.float64(1e+150))
    if hit.face < 0 or not wp.isfinite(hit.distance2):
        wp.atomic_or(error, 0, 16)
    else:
        wp.atomic_add(integral, 0, weight * hit.distance2)
        wp.atomic_max(maximum, 0, wp.sqrt(hit.distance2))

@wp.kernel
def reference_observed(tree: SurfaceTree, points: wp.array[wp.vec3], faces: wp.array2d[int], count: wp.array[int], integral: wp.array[wp.float64], maximum: wp.array[wp.float64], error: wp.array[int], hits: wp.array[Closest], weighted: wp.array[wp.float64]):
    i = wp.tid()
    if i >= count[0] or error[0] != 0:
        return
    a = wp.vec3d(points[faces[i, 0]])
    b = wp.vec3d(points[faces[i, 1]])
    c = wp.vec3d(points[faces[i, 2]])
    weight = wp.length(wp.cross(b - a, c - a)) / wp.float64(6.0)
    for k in range(3):
        dominant = a
        if k == 1:
            dominant = b
        elif k == 2:
            dominant = c
        sample = (a + b + c) / wp.float64(6.0) + dominant / wp.float64(2.0)
        hit = closest_point(tree, sample, wp.float64(1e+150))
        hits[3 * i + k] = hit
        weighted[3 * i + k] = weight * hit.distance2
        if hit.face < 0 or not wp.isfinite(hit.distance2):
            wp.atomic_or(error, 0, 16)
        else:
            wp.atomic_add(integral, 0, weight * hit.distance2)
            wp.atomic_max(maximum, 0, wp.sqrt(hit.distance2))

@wp.kernel
def samples_observed(tree: SurfaceTree, points: wp.array[wp.vec3], faces: wp.array2d[int], count: wp.array[int], integral: wp.array[wp.float64], maximum: wp.array[wp.float64], error: wp.array[int], hits: wp.array[Closest], weighted: wp.array[wp.float64]):
    i = wp.tid() // 3
    k = wp.tid() % 3
    if i >= count[0] or error[0] != 0:
        return
    a = wp.vec3d(points[faces[i, 0]])
    b = wp.vec3d(points[faces[i, 1]])
    c = wp.vec3d(points[faces[i, 2]])
    weight = wp.length(wp.cross(b - a, c - a)) / wp.float64(6.0)
    dominant = a
    if k == 1:
        dominant = b
    elif k == 2:
        dominant = c
    sample = (a + b + c) / wp.float64(6.0) + dominant / wp.float64(2.0)
    hit = closest_point(tree, sample, wp.float64(1e+150))
    hits[3 * i + k] = hit
    weighted[3 * i + k] = weight * hit.distance2
    if hit.face < 0 or not wp.isfinite(hit.distance2):
        wp.atomic_or(error, 0, 16)
    else:
        wp.atomic_add(integral, 0, weight * hit.distance2)
        wp.atomic_max(maximum, 0, wp.sqrt(hit.distance2))
