"""Private exact nearest traversal: omit children already rejected by the pop test."""
import warp as wp
from warpower._surface_bvh import SurfaceTree, Closest, box_distance, closest_triangle, closest_point
wp.set_module_options({"enable_backward": False, "fast_math": False})

@wp.func(inline=True)
def pruned_closest_point(tree: SurfaceTree, p: wp.vec3d, limit: wp.float64):
    result = Closest()
    result.point = wp.vec3d(0.0)
    result.face = -1
    result.distance2 = limit * limit
    # 63 Morton bits plus at most 31 index bits bound the radix depth below 96.
    stack = wp.zeros(shape=128, dtype=int)
    # Child distances are already computed for ordering. The point and tree
    # stay fixed during this query, so retain those rounded scalar values and
    # compare against the current best distance when each node is popped.
    bounds = wp.zeros(shape=128, dtype=wp.float64)
    stack[0] = 1
    bounds[0] = box_distance(tree, 1, p)
    size = int(1)
    while size > 0:
        size -= 1
        node = stack[size]
        if bounds[size] >= result.distance2:
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
            left_distance = box_distance(tree, left, p)
            right_distance = box_distance(tree, right, p)
            if left_distance < right_distance:
                left, right = right, left
                left_distance, right_distance = right_distance, left_distance
            # Keep the original rejection comparison, including unordered NaNs.
            # The best distance only decreases; a rejected child cannot revive.
            if not (left_distance >= result.distance2):
                stack[size] = left
                bounds[size] = left_distance
                size += 1
            if not (right_distance >= result.distance2):
                stack[size] = right
                bounds[size] = right_distance
                size += 1
    return result

@wp.kernel
def reference(tree: SurfaceTree, points: wp.array[wp.vec3d], limits: wp.array[wp.float64], hits: wp.array[Closest]):
    i = wp.tid()
    hits[i] = closest_point(tree, points[i], limits[i])


@wp.kernel
def candidate(tree: SurfaceTree, points: wp.array[wp.vec3d], limits: wp.array[wp.float64], hits: wp.array[Closest]):
    i = wp.tid()
    hits[i] = pruned_closest_point(tree, points[i], limits[i])
