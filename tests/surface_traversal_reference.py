"""Balanced-heap traversal from 18b18f5, retained as a regression reference.

Independent solid-angle and brute-force distance oracles accompany comparisons.
"""

import warp as wp

from warpower._surface_bvh import Closest, SurfaceTree, box_distance, closest_triangle, ray_box


@wp.func(inline=True)
def reference_closest_point(tree: SurfaceTree, p: wp.vec3d, limit: wp.float64):
    result = Closest()
    result.point = wp.vec3d(0.0)
    result.face = -1
    result.distance2 = limit * limit
    stack = wp.zeros(shape=64, dtype=int)
    stack[0] = 1
    size = int(1)
    while size > 0:
        size -= 1
        node = stack[size]
        if box_distance(tree, node, p) >= result.distance2:
            continue
        if node >= tree.leaf_base:
            first = 4 * (node - tree.leaf_base)
            for j in range(first, wp.min(first + 4, tree.n)):
                f = tree.ids[j]
                a = wp.vec3d(tree.points[tree.indices[3 * f]])
                b = wp.vec3d(tree.points[tree.indices[3 * f + 1]])
                c = wp.vec3d(tree.points[tree.indices[3 * f + 2]])
                q = closest_triangle(p, a, b, c)
                distance = wp.length_sq(p - q)
                if distance < result.distance2:
                    result.point = q
                    result.face = f
                    result.distance2 = distance
        else:
            left, right = 2 * node, 2 * node + 1
            if box_distance(tree, left, p) < box_distance(tree, right, p):
                left, right = right, left
            stack[size] = left
            stack[size + 1] = right
            size += 2
    return result


@wp.func(inline=True)
def reference_ray_count(
    tree: SurfaceTree, origin: wp.vec3d, direction: wp.vec3d, excluded_face: int = -1
):
    # Return signed intersections and an ambiguity flag. Shared-edge hits are
    # retried with another direction instead of double-counting adjacent faces.
    crossings, ambiguous = int(0), int(0)
    if excluded_face >= 0:
        a = wp.vec3d(tree.points[tree.indices[3 * excluded_face]])
        b = wp.vec3d(tree.points[tree.indices[3 * excluded_face + 1]]) - a
        c = wp.vec3d(tree.points[tree.indices[3 * excluded_face + 2]]) - a
        crossings = -int(wp.dot(direction, wp.cross(b, c)) < wp.float64(0.0))
    inverse = wp.vec3d(
        wp.float64(1.0) / direction[0],
        wp.float64(1.0) / direction[1],
        wp.float64(1.0) / direction[2],
    )
    stack = wp.zeros(shape=64, dtype=int)
    stack[0] = 1
    size = int(1)
    while size > 0:
        size -= 1
        node = stack[size]
        if not ray_box(tree, node, origin, inverse):
            continue
        if node >= tree.leaf_base:
            first = 4 * (node - tree.leaf_base)
            for j in range(first, wp.min(first + 4, tree.n)):
                f = tree.ids[j]
                if f == excluded_face:
                    continue
                a = wp.vec3d(tree.points[tree.indices[3 * f]])
                b = wp.vec3d(tree.points[tree.indices[3 * f + 1]]) - a
                c = wp.vec3d(tree.points[tree.indices[3 * f + 2]]) - a
                normal = wp.cross(b, c)
                det = wp.dot(direction, normal)
                scale = wp.length(normal) * wp.length(direction)
                if wp.abs(det) <= wp.float64(1.0e-14) * scale:
                    continue
                delta = origin - a
                t = -wp.dot(delta, normal) / det
                if t <= wp.float64(0.0):
                    continue
                q = delta + t * direction
                inverse_area = wp.float64(1.0) / wp.dot(normal, normal)
                u = wp.dot(wp.cross(q, c), normal) * inverse_area
                v = wp.dot(wp.cross(b, q), normal) * inverse_area
                tolerance = wp.float64(2.0e-12)
                if u >= -tolerance and v >= -tolerance and u + v <= wp.float64(1.0) + tolerance:
                    if u <= tolerance or v <= tolerance or u + v >= wp.float64(1.0) - tolerance:
                        ambiguous = 1
                    crossings += wp.where(det > wp.float64(0.0), 1, -1)
        else:
            stack[size] = 2 * node
            stack[size + 1] = 2 * node + 1
            size += 2
    return wp.vec2i(crossings, ambiguous)


@wp.func(inline=True)
def reference_material_side(tree: SurfaceTree, p: wp.vec3d, excluded_face: int = -1):
    # A generic ray suffices off triangle boundaries. Three fixed, non-axis
    # directions are available for shared-edge/vertex ties, all on device.
    result = reference_ray_count(
        tree, p, wp.vec3d(0.8170638283521119, 0.33141623424177, 0.4713172342319), excluded_face
    )
    if result[1] != 0:
        result = reference_ray_count(
            tree, p, wp.vec3d(-0.213173497731, 0.93127172131, 0.44721714319), excluded_face
        )
    if result[1] != 0:
        result = reference_ray_count(
            tree, p, wp.vec3d(0.63719234173, -0.72819341297, 0.27191345121), excluded_face
        )
    if result[1] != 0:
        return 128
    return result[0]
