"""Unmodified radix nearest traversal from18ecb03, for numerical regressions.

This is a regression reference, not an independent distance oracle. The
existing surface-radix tests separately check brute-force triangle distances.
"""

import warp as wp

from warpower._surface_bvh import Closest, SurfaceTree, box_distance, closest_triangle


@wp.func(inline=True)
def uncached_closest_point(tree: SurfaceTree, p: wp.vec3d, limit: wp.float64):
    result = Closest()
    result.point = wp.vec3d(0.0)
    result.face = -1
    result.distance2 = limit * limit
    # 63 Morton bits plus at most 31 index bits bound the radix depth below 96.
    stack = wp.zeros(shape=128, dtype=int)
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
            left, right = int(tree.left[node]), int(tree.right[node])
            if box_distance(tree, left, p) < box_distance(tree, right, p):
                left, right = right, left
            stack[size] = left
            stack[size + 1] = right
            size += 2
    return result
