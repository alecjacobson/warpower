"""Private #197 partial-ray experiment; no production replacement.

The leaf floating expressions are copied verbatim from the current scalar ray.
Integer sums implement both crossing reduction and ambiguity OR (>0). Every
thread reaches each tile collective, including inactive final-block lanes.
"""
import warp as wp
from warpower._surface_bvh import SurfaceTree, ray_box, ray_count, material_side, closest_point
wp.set_module_options({'enable_backward': False, 'fast_math': False})

@wp.kernel
def reference(tree: SurfaceTree, points: wp.array[wp.vec3d], excluded: wp.array[int], out: wp.array2d[int]):
    i = wp.tid()
    r = ray_count(tree, points[i], wp.vec3d(1.0, 1.0, 1.0), excluded[i])
    out[i, 0] = r[0]
    out[i, 1] = r[1]
    out[i, 2] = material_side(tree, points[i], excluded[i])

@wp.kernel
def reference_side(tree: SurfaceTree, points: wp.array[wp.vec3d], excluded: wp.array[int], out: wp.array2d[int]):
    i = wp.tid()
    out[i, 2] = material_side(tree, points[i], excluded[i])

@wp.kernel
def reference_nearest(tree: SurfaceTree, points: wp.array[wp.vec3d], distances: wp.array[wp.float64]):
    i = wp.tid()
    hit = closest_point(tree, points[i], wp.float64(1.0e150))
    distances[i] = hit.distance2

@wp.func(inline=True)
def partial_ray_count(tree: SurfaceTree, origin: wp.vec3d, direction: wp.vec3d, excluded_face: int, lane: int, width: int, partition: int):
    # Return signed intersections and an ambiguity flag. Shared-edge hits are
    # retried with another direction instead of double-counting adjacent faces.
    crossings, ambiguous = int(0), int(0)
    if excluded_face >= 0 and lane == 0:
        a = wp.vec3d(tree.points[tree.indices[3 * excluded_face]])
        b = wp.vec3d(tree.points[tree.indices[3 * excluded_face + 1]]) - a
        c = wp.vec3d(tree.points[tree.indices[3 * excluded_face + 2]]) - a
        crossings = -int(wp.dot(direction, wp.cross(b, c)) < wp.float64(0.0))
    inverse = wp.vec3d(
        wp.float64(1.0) / direction[0],
        wp.float64(1.0) / direction[1],
        wp.float64(1.0) / direction[2],
    )
    node = int(1)
    end_node = int(0)
    offset = lane
    stride = width
    if partition == 1:
        # Lanes own disjoint subtrees. Test every ancestor exactly as the scalar
        # traversal does. If a leaf is reached early, only its canonical lane
        # owns it. The excluded-face correction is still applied exactly once.
        bit = width // 2
        while bit > 0:
            if not ray_box(tree, node, origin, inverse):
                return wp.vec2i(crossings, ambiguous)
            if node >= tree.leaf_base:
                if (lane & (2 * bit - 1)) != 0:
                    return wp.vec2i(crossings, ambiguous)
                break
            if (lane & bit) == 0:
                node = tree.right[node]
            else:
                node = tree.left[node]
            bit //= 2
        end_node = tree.escape[node]
        offset = 0
        stride = 1
    while node != end_node:
        if not ray_box(tree, node, origin, inverse):
            node = tree.escape[node]
            continue
        if node >= tree.leaf_base:
            first = 4 * (node - tree.leaf_base)
            for j in range(first + offset, wp.min(first + 4, tree.n), stride):
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
            node = tree.escape[node]
        else:
            node = tree.right[node]
    return wp.vec2i(crossings, ambiguous)


@wp.kernel
def leaf_1(tree: SurfaceTree, points: wp.array[wp.vec3d], excluded: wp.array[int], out: wp.array2d[int]):
    block, lane = wp.tid()
    row = lane // 1
    part = lane % 1
    i = block * 32 + row
    p = wp.vec3d(0.0)
    ex = int(-1)
    if i < points.shape[0]:
        p = points[i]
        ex = excluded[i]
    raw_part = wp.vec2i(0, 0)
    if i < points.shape[0]:
        raw_part = partial_ray_count(tree, p, wp.vec3d(1.0, 1.0, 1.0), ex, part, 1, 0)
    raw = raw_part
    first_part = wp.vec2i(0, 0)
    if i < points.shape[0]:
        first_part = partial_ray_count(tree, p, wp.vec3d(0.8170638283521119, 0.33141623424177, 0.4713172342319), ex, part, 1, 0)
    first = first_part
    second_part = wp.vec2i(0, 0)
    if i < points.shape[0] and first[1] != 0:
        second_part = partial_ray_count(tree, p, wp.vec3d(-0.213173497731, 0.93127172131, 0.44721714319), ex, part, 1, 0)
    second = second_part
    third_part = wp.vec2i(0, 0)
    if i < points.shape[0] and first[1] != 0 and second[1] != 0:
        third_part = partial_ray_count(tree, p, wp.vec3d(0.63719234173, -0.72819341297, 0.27191345121), ex, part, 1, 0)
    third = third_part
    side = first[0]
    if first[1] != 0:
        side = second[0]
        if second[1] != 0:
            side = third[0]
            if third[1] != 0:
                side = 128
    if part == 0 and i < points.shape[0]:
        out[i, 2] = side
        out[i, 0] = raw[0]
        out[i, 1] = int(raw[1] != 0)


@wp.kernel
def leaf_1_side(tree: SurfaceTree, points: wp.array[wp.vec3d], excluded: wp.array[int], out: wp.array2d[int]):
    block, lane = wp.tid()
    row = lane // 1
    part = lane % 1
    i = block * 32 + row
    p = wp.vec3d(0.0)
    ex = int(-1)
    if i < points.shape[0]:
        p = points[i]
        ex = excluded[i]
    first_part = wp.vec2i(0, 0)
    if i < points.shape[0]:
        first_part = partial_ray_count(tree, p, wp.vec3d(0.8170638283521119, 0.33141623424177, 0.4713172342319), ex, part, 1, 0)
    first = first_part
    second_part = wp.vec2i(0, 0)
    if i < points.shape[0] and first[1] != 0:
        second_part = partial_ray_count(tree, p, wp.vec3d(-0.213173497731, 0.93127172131, 0.44721714319), ex, part, 1, 0)
    second = second_part
    third_part = wp.vec2i(0, 0)
    if i < points.shape[0] and first[1] != 0 and second[1] != 0:
        third_part = partial_ray_count(tree, p, wp.vec3d(0.63719234173, -0.72819341297, 0.27191345121), ex, part, 1, 0)
    third = third_part
    side = first[0]
    if first[1] != 0:
        side = second[0]
        if second[1] != 0:
            side = third[0]
            if third[1] != 0:
                side = 128
    if part == 0 and i < points.shape[0]:
        out[i, 2] = side


@wp.kernel
def leaf_2(tree: SurfaceTree, points: wp.array[wp.vec3d], excluded: wp.array[int], out: wp.array2d[int]):
    block, lane = wp.tid()
    row = lane // 2
    part = lane % 2
    i = block * 16 + row
    p = wp.vec3d(0.0)
    ex = int(-1)
    if i < points.shape[0]:
        p = points[i]
        ex = excluded[i]
    raw_part = wp.vec2i(0, 0)
    if i < points.shape[0]:
        raw_part = partial_ray_count(tree, p, wp.vec3d(1.0, 1.0, 1.0), ex, part, 2, 0)
    raw_tile = wp.tile_reshape(wp.tile(raw_part), shape=(16, 2))
    raw_sums = wp.tile_sum(raw_tile, axis=1)
    raw = raw_sums[row]
    first_part = wp.vec2i(0, 0)
    if i < points.shape[0]:
        first_part = partial_ray_count(tree, p, wp.vec3d(0.8170638283521119, 0.33141623424177, 0.4713172342319), ex, part, 2, 0)
    first_tile = wp.tile_reshape(wp.tile(first_part), shape=(16, 2))
    first_sums = wp.tile_sum(first_tile, axis=1)
    first = first_sums[row]
    second_part = wp.vec2i(0, 0)
    if i < points.shape[0] and first[1] != 0:
        second_part = partial_ray_count(tree, p, wp.vec3d(-0.213173497731, 0.93127172131, 0.44721714319), ex, part, 2, 0)
    second_tile = wp.tile_reshape(wp.tile(second_part), shape=(16, 2))
    second_sums = wp.tile_sum(second_tile, axis=1)
    second = second_sums[row]
    third_part = wp.vec2i(0, 0)
    if i < points.shape[0] and first[1] != 0 and second[1] != 0:
        third_part = partial_ray_count(tree, p, wp.vec3d(0.63719234173, -0.72819341297, 0.27191345121), ex, part, 2, 0)
    third_tile = wp.tile_reshape(wp.tile(third_part), shape=(16, 2))
    third_sums = wp.tile_sum(third_tile, axis=1)
    third = third_sums[row]
    side = first[0]
    if first[1] != 0:
        side = second[0]
        if second[1] != 0:
            side = third[0]
            if third[1] != 0:
                side = 128
    if part == 0 and i < points.shape[0]:
        out[i, 2] = side
        out[i, 0] = raw[0]
        out[i, 1] = int(raw[1] != 0)


@wp.kernel
def leaf_2_side(tree: SurfaceTree, points: wp.array[wp.vec3d], excluded: wp.array[int], out: wp.array2d[int]):
    block, lane = wp.tid()
    row = lane // 2
    part = lane % 2
    i = block * 16 + row
    p = wp.vec3d(0.0)
    ex = int(-1)
    if i < points.shape[0]:
        p = points[i]
        ex = excluded[i]
    first_part = wp.vec2i(0, 0)
    if i < points.shape[0]:
        first_part = partial_ray_count(tree, p, wp.vec3d(0.8170638283521119, 0.33141623424177, 0.4713172342319), ex, part, 2, 0)
    first_tile = wp.tile_reshape(wp.tile(first_part), shape=(16, 2))
    first_sums = wp.tile_sum(first_tile, axis=1)
    first = first_sums[row]
    second_part = wp.vec2i(0, 0)
    if i < points.shape[0] and first[1] != 0:
        second_part = partial_ray_count(tree, p, wp.vec3d(-0.213173497731, 0.93127172131, 0.44721714319), ex, part, 2, 0)
    second_tile = wp.tile_reshape(wp.tile(second_part), shape=(16, 2))
    second_sums = wp.tile_sum(second_tile, axis=1)
    second = second_sums[row]
    third_part = wp.vec2i(0, 0)
    if i < points.shape[0] and first[1] != 0 and second[1] != 0:
        third_part = partial_ray_count(tree, p, wp.vec3d(0.63719234173, -0.72819341297, 0.27191345121), ex, part, 2, 0)
    third_tile = wp.tile_reshape(wp.tile(third_part), shape=(16, 2))
    third_sums = wp.tile_sum(third_tile, axis=1)
    third = third_sums[row]
    side = first[0]
    if first[1] != 0:
        side = second[0]
        if second[1] != 0:
            side = third[0]
            if third[1] != 0:
                side = 128
    if part == 0 and i < points.shape[0]:
        out[i, 2] = side


@wp.kernel
def leaf_4(tree: SurfaceTree, points: wp.array[wp.vec3d], excluded: wp.array[int], out: wp.array2d[int]):
    block, lane = wp.tid()
    row = lane // 4
    part = lane % 4
    i = block * 8 + row
    p = wp.vec3d(0.0)
    ex = int(-1)
    if i < points.shape[0]:
        p = points[i]
        ex = excluded[i]
    raw_part = wp.vec2i(0, 0)
    if i < points.shape[0]:
        raw_part = partial_ray_count(tree, p, wp.vec3d(1.0, 1.0, 1.0), ex, part, 4, 0)
    raw_tile = wp.tile_reshape(wp.tile(raw_part), shape=(8, 4))
    raw_sums = wp.tile_sum(raw_tile, axis=1)
    raw = raw_sums[row]
    first_part = wp.vec2i(0, 0)
    if i < points.shape[0]:
        first_part = partial_ray_count(tree, p, wp.vec3d(0.8170638283521119, 0.33141623424177, 0.4713172342319), ex, part, 4, 0)
    first_tile = wp.tile_reshape(wp.tile(first_part), shape=(8, 4))
    first_sums = wp.tile_sum(first_tile, axis=1)
    first = first_sums[row]
    second_part = wp.vec2i(0, 0)
    if i < points.shape[0] and first[1] != 0:
        second_part = partial_ray_count(tree, p, wp.vec3d(-0.213173497731, 0.93127172131, 0.44721714319), ex, part, 4, 0)
    second_tile = wp.tile_reshape(wp.tile(second_part), shape=(8, 4))
    second_sums = wp.tile_sum(second_tile, axis=1)
    second = second_sums[row]
    third_part = wp.vec2i(0, 0)
    if i < points.shape[0] and first[1] != 0 and second[1] != 0:
        third_part = partial_ray_count(tree, p, wp.vec3d(0.63719234173, -0.72819341297, 0.27191345121), ex, part, 4, 0)
    third_tile = wp.tile_reshape(wp.tile(third_part), shape=(8, 4))
    third_sums = wp.tile_sum(third_tile, axis=1)
    third = third_sums[row]
    side = first[0]
    if first[1] != 0:
        side = second[0]
        if second[1] != 0:
            side = third[0]
            if third[1] != 0:
                side = 128
    if part == 0 and i < points.shape[0]:
        out[i, 2] = side
        out[i, 0] = raw[0]
        out[i, 1] = int(raw[1] != 0)


@wp.kernel
def leaf_4_side(tree: SurfaceTree, points: wp.array[wp.vec3d], excluded: wp.array[int], out: wp.array2d[int]):
    block, lane = wp.tid()
    row = lane // 4
    part = lane % 4
    i = block * 8 + row
    p = wp.vec3d(0.0)
    ex = int(-1)
    if i < points.shape[0]:
        p = points[i]
        ex = excluded[i]
    first_part = wp.vec2i(0, 0)
    if i < points.shape[0]:
        first_part = partial_ray_count(tree, p, wp.vec3d(0.8170638283521119, 0.33141623424177, 0.4713172342319), ex, part, 4, 0)
    first_tile = wp.tile_reshape(wp.tile(first_part), shape=(8, 4))
    first_sums = wp.tile_sum(first_tile, axis=1)
    first = first_sums[row]
    second_part = wp.vec2i(0, 0)
    if i < points.shape[0] and first[1] != 0:
        second_part = partial_ray_count(tree, p, wp.vec3d(-0.213173497731, 0.93127172131, 0.44721714319), ex, part, 4, 0)
    second_tile = wp.tile_reshape(wp.tile(second_part), shape=(8, 4))
    second_sums = wp.tile_sum(second_tile, axis=1)
    second = second_sums[row]
    third_part = wp.vec2i(0, 0)
    if i < points.shape[0] and first[1] != 0 and second[1] != 0:
        third_part = partial_ray_count(tree, p, wp.vec3d(0.63719234173, -0.72819341297, 0.27191345121), ex, part, 4, 0)
    third_tile = wp.tile_reshape(wp.tile(third_part), shape=(8, 4))
    third_sums = wp.tile_sum(third_tile, axis=1)
    third = third_sums[row]
    side = first[0]
    if first[1] != 0:
        side = second[0]
        if second[1] != 0:
            side = third[0]
            if third[1] != 0:
                side = 128
    if part == 0 and i < points.shape[0]:
        out[i, 2] = side


@wp.kernel
def subtree_2(tree: SurfaceTree, points: wp.array[wp.vec3d], excluded: wp.array[int], out: wp.array2d[int]):
    block, lane = wp.tid()
    row = lane // 2
    part = lane % 2
    i = block * 16 + row
    p = wp.vec3d(0.0)
    ex = int(-1)
    if i < points.shape[0]:
        p = points[i]
        ex = excluded[i]
    raw_part = wp.vec2i(0, 0)
    if i < points.shape[0]:
        raw_part = partial_ray_count(tree, p, wp.vec3d(1.0, 1.0, 1.0), ex, part, 2, 1)
    raw_tile = wp.tile_reshape(wp.tile(raw_part), shape=(16, 2))
    raw_sums = wp.tile_sum(raw_tile, axis=1)
    raw = raw_sums[row]
    first_part = wp.vec2i(0, 0)
    if i < points.shape[0]:
        first_part = partial_ray_count(tree, p, wp.vec3d(0.8170638283521119, 0.33141623424177, 0.4713172342319), ex, part, 2, 1)
    first_tile = wp.tile_reshape(wp.tile(first_part), shape=(16, 2))
    first_sums = wp.tile_sum(first_tile, axis=1)
    first = first_sums[row]
    second_part = wp.vec2i(0, 0)
    if i < points.shape[0] and first[1] != 0:
        second_part = partial_ray_count(tree, p, wp.vec3d(-0.213173497731, 0.93127172131, 0.44721714319), ex, part, 2, 1)
    second_tile = wp.tile_reshape(wp.tile(second_part), shape=(16, 2))
    second_sums = wp.tile_sum(second_tile, axis=1)
    second = second_sums[row]
    third_part = wp.vec2i(0, 0)
    if i < points.shape[0] and first[1] != 0 and second[1] != 0:
        third_part = partial_ray_count(tree, p, wp.vec3d(0.63719234173, -0.72819341297, 0.27191345121), ex, part, 2, 1)
    third_tile = wp.tile_reshape(wp.tile(third_part), shape=(16, 2))
    third_sums = wp.tile_sum(third_tile, axis=1)
    third = third_sums[row]
    side = first[0]
    if first[1] != 0:
        side = second[0]
        if second[1] != 0:
            side = third[0]
            if third[1] != 0:
                side = 128
    if part == 0 and i < points.shape[0]:
        out[i, 2] = side
        out[i, 0] = raw[0]
        out[i, 1] = int(raw[1] != 0)


@wp.kernel
def subtree_2_side(tree: SurfaceTree, points: wp.array[wp.vec3d], excluded: wp.array[int], out: wp.array2d[int]):
    block, lane = wp.tid()
    row = lane // 2
    part = lane % 2
    i = block * 16 + row
    p = wp.vec3d(0.0)
    ex = int(-1)
    if i < points.shape[0]:
        p = points[i]
        ex = excluded[i]
    first_part = wp.vec2i(0, 0)
    if i < points.shape[0]:
        first_part = partial_ray_count(tree, p, wp.vec3d(0.8170638283521119, 0.33141623424177, 0.4713172342319), ex, part, 2, 1)
    first_tile = wp.tile_reshape(wp.tile(first_part), shape=(16, 2))
    first_sums = wp.tile_sum(first_tile, axis=1)
    first = first_sums[row]
    second_part = wp.vec2i(0, 0)
    if i < points.shape[0] and first[1] != 0:
        second_part = partial_ray_count(tree, p, wp.vec3d(-0.213173497731, 0.93127172131, 0.44721714319), ex, part, 2, 1)
    second_tile = wp.tile_reshape(wp.tile(second_part), shape=(16, 2))
    second_sums = wp.tile_sum(second_tile, axis=1)
    second = second_sums[row]
    third_part = wp.vec2i(0, 0)
    if i < points.shape[0] and first[1] != 0 and second[1] != 0:
        third_part = partial_ray_count(tree, p, wp.vec3d(0.63719234173, -0.72819341297, 0.27191345121), ex, part, 2, 1)
    third_tile = wp.tile_reshape(wp.tile(third_part), shape=(16, 2))
    third_sums = wp.tile_sum(third_tile, axis=1)
    third = third_sums[row]
    side = first[0]
    if first[1] != 0:
        side = second[0]
        if second[1] != 0:
            side = third[0]
            if third[1] != 0:
                side = 128
    if part == 0 and i < points.shape[0]:
        out[i, 2] = side


@wp.kernel
def subtree_4(tree: SurfaceTree, points: wp.array[wp.vec3d], excluded: wp.array[int], out: wp.array2d[int]):
    block, lane = wp.tid()
    row = lane // 4
    part = lane % 4
    i = block * 8 + row
    p = wp.vec3d(0.0)
    ex = int(-1)
    if i < points.shape[0]:
        p = points[i]
        ex = excluded[i]
    raw_part = wp.vec2i(0, 0)
    if i < points.shape[0]:
        raw_part = partial_ray_count(tree, p, wp.vec3d(1.0, 1.0, 1.0), ex, part, 4, 1)
    raw_tile = wp.tile_reshape(wp.tile(raw_part), shape=(8, 4))
    raw_sums = wp.tile_sum(raw_tile, axis=1)
    raw = raw_sums[row]
    first_part = wp.vec2i(0, 0)
    if i < points.shape[0]:
        first_part = partial_ray_count(tree, p, wp.vec3d(0.8170638283521119, 0.33141623424177, 0.4713172342319), ex, part, 4, 1)
    first_tile = wp.tile_reshape(wp.tile(first_part), shape=(8, 4))
    first_sums = wp.tile_sum(first_tile, axis=1)
    first = first_sums[row]
    second_part = wp.vec2i(0, 0)
    if i < points.shape[0] and first[1] != 0:
        second_part = partial_ray_count(tree, p, wp.vec3d(-0.213173497731, 0.93127172131, 0.44721714319), ex, part, 4, 1)
    second_tile = wp.tile_reshape(wp.tile(second_part), shape=(8, 4))
    second_sums = wp.tile_sum(second_tile, axis=1)
    second = second_sums[row]
    third_part = wp.vec2i(0, 0)
    if i < points.shape[0] and first[1] != 0 and second[1] != 0:
        third_part = partial_ray_count(tree, p, wp.vec3d(0.63719234173, -0.72819341297, 0.27191345121), ex, part, 4, 1)
    third_tile = wp.tile_reshape(wp.tile(third_part), shape=(8, 4))
    third_sums = wp.tile_sum(third_tile, axis=1)
    third = third_sums[row]
    side = first[0]
    if first[1] != 0:
        side = second[0]
        if second[1] != 0:
            side = third[0]
            if third[1] != 0:
                side = 128
    if part == 0 and i < points.shape[0]:
        out[i, 2] = side
        out[i, 0] = raw[0]
        out[i, 1] = int(raw[1] != 0)


@wp.kernel
def subtree_4_side(tree: SurfaceTree, points: wp.array[wp.vec3d], excluded: wp.array[int], out: wp.array2d[int]):
    block, lane = wp.tid()
    row = lane // 4
    part = lane % 4
    i = block * 8 + row
    p = wp.vec3d(0.0)
    ex = int(-1)
    if i < points.shape[0]:
        p = points[i]
        ex = excluded[i]
    first_part = wp.vec2i(0, 0)
    if i < points.shape[0]:
        first_part = partial_ray_count(tree, p, wp.vec3d(0.8170638283521119, 0.33141623424177, 0.4713172342319), ex, part, 4, 1)
    first_tile = wp.tile_reshape(wp.tile(first_part), shape=(8, 4))
    first_sums = wp.tile_sum(first_tile, axis=1)
    first = first_sums[row]
    second_part = wp.vec2i(0, 0)
    if i < points.shape[0] and first[1] != 0:
        second_part = partial_ray_count(tree, p, wp.vec3d(-0.213173497731, 0.93127172131, 0.44721714319), ex, part, 4, 1)
    second_tile = wp.tile_reshape(wp.tile(second_part), shape=(8, 4))
    second_sums = wp.tile_sum(second_tile, axis=1)
    second = second_sums[row]
    third_part = wp.vec2i(0, 0)
    if i < points.shape[0] and first[1] != 0 and second[1] != 0:
        third_part = partial_ray_count(tree, p, wp.vec3d(0.63719234173, -0.72819341297, 0.27191345121), ex, part, 4, 1)
    third_tile = wp.tile_reshape(wp.tile(third_part), shape=(8, 4))
    third_sums = wp.tile_sum(third_tile, axis=1)
    third = third_sums[row]
    side = first[0]
    if first[1] != 0:
        side = second[0]
        if second[1] != 0:
            side = third[0]
            if third[1] != 0:
                side = 128
    if part == 0 and i < points.shape[0]:
        out[i, 2] = side


@wp.kernel
def subtree_8(tree: SurfaceTree, points: wp.array[wp.vec3d], excluded: wp.array[int], out: wp.array2d[int]):
    block, lane = wp.tid()
    row = lane // 8
    part = lane % 8
    i = block * 4 + row
    p = wp.vec3d(0.0)
    ex = int(-1)
    if i < points.shape[0]:
        p = points[i]
        ex = excluded[i]
    raw_part = wp.vec2i(0, 0)
    if i < points.shape[0]:
        raw_part = partial_ray_count(tree, p, wp.vec3d(1.0, 1.0, 1.0), ex, part, 8, 1)
    raw_tile = wp.tile_reshape(wp.tile(raw_part), shape=(4, 8))
    raw_sums = wp.tile_sum(raw_tile, axis=1)
    raw = raw_sums[row]
    first_part = wp.vec2i(0, 0)
    if i < points.shape[0]:
        first_part = partial_ray_count(tree, p, wp.vec3d(0.8170638283521119, 0.33141623424177, 0.4713172342319), ex, part, 8, 1)
    first_tile = wp.tile_reshape(wp.tile(first_part), shape=(4, 8))
    first_sums = wp.tile_sum(first_tile, axis=1)
    first = first_sums[row]
    second_part = wp.vec2i(0, 0)
    if i < points.shape[0] and first[1] != 0:
        second_part = partial_ray_count(tree, p, wp.vec3d(-0.213173497731, 0.93127172131, 0.44721714319), ex, part, 8, 1)
    second_tile = wp.tile_reshape(wp.tile(second_part), shape=(4, 8))
    second_sums = wp.tile_sum(second_tile, axis=1)
    second = second_sums[row]
    third_part = wp.vec2i(0, 0)
    if i < points.shape[0] and first[1] != 0 and second[1] != 0:
        third_part = partial_ray_count(tree, p, wp.vec3d(0.63719234173, -0.72819341297, 0.27191345121), ex, part, 8, 1)
    third_tile = wp.tile_reshape(wp.tile(third_part), shape=(4, 8))
    third_sums = wp.tile_sum(third_tile, axis=1)
    third = third_sums[row]
    side = first[0]
    if first[1] != 0:
        side = second[0]
        if second[1] != 0:
            side = third[0]
            if third[1] != 0:
                side = 128
    if part == 0 and i < points.shape[0]:
        out[i, 2] = side
        out[i, 0] = raw[0]
        out[i, 1] = int(raw[1] != 0)


@wp.kernel
def subtree_8_side(tree: SurfaceTree, points: wp.array[wp.vec3d], excluded: wp.array[int], out: wp.array2d[int]):
    block, lane = wp.tid()
    row = lane // 8
    part = lane % 8
    i = block * 4 + row
    p = wp.vec3d(0.0)
    ex = int(-1)
    if i < points.shape[0]:
        p = points[i]
        ex = excluded[i]
    first_part = wp.vec2i(0, 0)
    if i < points.shape[0]:
        first_part = partial_ray_count(tree, p, wp.vec3d(0.8170638283521119, 0.33141623424177, 0.4713172342319), ex, part, 8, 1)
    first_tile = wp.tile_reshape(wp.tile(first_part), shape=(4, 8))
    first_sums = wp.tile_sum(first_tile, axis=1)
    first = first_sums[row]
    second_part = wp.vec2i(0, 0)
    if i < points.shape[0] and first[1] != 0:
        second_part = partial_ray_count(tree, p, wp.vec3d(-0.213173497731, 0.93127172131, 0.44721714319), ex, part, 8, 1)
    second_tile = wp.tile_reshape(wp.tile(second_part), shape=(4, 8))
    second_sums = wp.tile_sum(second_tile, axis=1)
    second = second_sums[row]
    third_part = wp.vec2i(0, 0)
    if i < points.shape[0] and first[1] != 0 and second[1] != 0:
        third_part = partial_ray_count(tree, p, wp.vec3d(0.63719234173, -0.72819341297, 0.27191345121), ex, part, 8, 1)
    third_tile = wp.tile_reshape(wp.tile(third_part), shape=(4, 8))
    third_sums = wp.tile_sum(third_tile, axis=1)
    third = third_sums[row]
    side = first[0]
    if first[1] != 0:
        side = second[0]
        if second[1] != 0:
            side = third[0]
            if third[1] != 0:
                side = 128
    if part == 0 and i < points.shape[0]:
        out[i, 2] = side


@wp.kernel
def subtree_16(tree: SurfaceTree, points: wp.array[wp.vec3d], excluded: wp.array[int], out: wp.array2d[int]):
    block, lane = wp.tid()
    row = lane // 16
    part = lane % 16
    i = block * 2 + row
    p = wp.vec3d(0.0)
    ex = int(-1)
    if i < points.shape[0]:
        p = points[i]
        ex = excluded[i]
    raw_part = wp.vec2i(0, 0)
    if i < points.shape[0]:
        raw_part = partial_ray_count(tree, p, wp.vec3d(1.0, 1.0, 1.0), ex, part, 16, 1)
    raw_tile = wp.tile_reshape(wp.tile(raw_part), shape=(2, 16))
    raw_sums = wp.tile_sum(raw_tile, axis=1)
    raw = raw_sums[row]
    first_part = wp.vec2i(0, 0)
    if i < points.shape[0]:
        first_part = partial_ray_count(tree, p, wp.vec3d(0.8170638283521119, 0.33141623424177, 0.4713172342319), ex, part, 16, 1)
    first_tile = wp.tile_reshape(wp.tile(first_part), shape=(2, 16))
    first_sums = wp.tile_sum(first_tile, axis=1)
    first = first_sums[row]
    second_part = wp.vec2i(0, 0)
    if i < points.shape[0] and first[1] != 0:
        second_part = partial_ray_count(tree, p, wp.vec3d(-0.213173497731, 0.93127172131, 0.44721714319), ex, part, 16, 1)
    second_tile = wp.tile_reshape(wp.tile(second_part), shape=(2, 16))
    second_sums = wp.tile_sum(second_tile, axis=1)
    second = second_sums[row]
    third_part = wp.vec2i(0, 0)
    if i < points.shape[0] and first[1] != 0 and second[1] != 0:
        third_part = partial_ray_count(tree, p, wp.vec3d(0.63719234173, -0.72819341297, 0.27191345121), ex, part, 16, 1)
    third_tile = wp.tile_reshape(wp.tile(third_part), shape=(2, 16))
    third_sums = wp.tile_sum(third_tile, axis=1)
    third = third_sums[row]
    side = first[0]
    if first[1] != 0:
        side = second[0]
        if second[1] != 0:
            side = third[0]
            if third[1] != 0:
                side = 128
    if part == 0 and i < points.shape[0]:
        out[i, 2] = side
        out[i, 0] = raw[0]
        out[i, 1] = int(raw[1] != 0)


@wp.kernel
def subtree_16_side(tree: SurfaceTree, points: wp.array[wp.vec3d], excluded: wp.array[int], out: wp.array2d[int]):
    block, lane = wp.tid()
    row = lane // 16
    part = lane % 16
    i = block * 2 + row
    p = wp.vec3d(0.0)
    ex = int(-1)
    if i < points.shape[0]:
        p = points[i]
        ex = excluded[i]
    first_part = wp.vec2i(0, 0)
    if i < points.shape[0]:
        first_part = partial_ray_count(tree, p, wp.vec3d(0.8170638283521119, 0.33141623424177, 0.4713172342319), ex, part, 16, 1)
    first_tile = wp.tile_reshape(wp.tile(first_part), shape=(2, 16))
    first_sums = wp.tile_sum(first_tile, axis=1)
    first = first_sums[row]
    second_part = wp.vec2i(0, 0)
    if i < points.shape[0] and first[1] != 0:
        second_part = partial_ray_count(tree, p, wp.vec3d(-0.213173497731, 0.93127172131, 0.44721714319), ex, part, 16, 1)
    second_tile = wp.tile_reshape(wp.tile(second_part), shape=(2, 16))
    second_sums = wp.tile_sum(second_tile, axis=1)
    second = second_sums[row]
    third_part = wp.vec2i(0, 0)
    if i < points.shape[0] and first[1] != 0 and second[1] != 0:
        third_part = partial_ray_count(tree, p, wp.vec3d(0.63719234173, -0.72819341297, 0.27191345121), ex, part, 16, 1)
    third_tile = wp.tile_reshape(wp.tile(third_part), shape=(2, 16))
    third_sums = wp.tile_sum(third_tile, axis=1)
    third = third_sums[row]
    side = first[0]
    if first[1] != 0:
        side = second[0]
        if second[1] != 0:
            side = third[0]
            if third[1] != 0:
                side = 128
    if part == 0 and i < points.shape[0]:
        out[i, 2] = side


@wp.kernel
def subtree_32(tree: SurfaceTree, points: wp.array[wp.vec3d], excluded: wp.array[int], out: wp.array2d[int]):
    block, lane = wp.tid()
    row = lane // 32
    part = lane % 32
    i = block * 1 + row
    p = wp.vec3d(0.0)
    ex = int(-1)
    if i < points.shape[0]:
        p = points[i]
        ex = excluded[i]
    raw_part = wp.vec2i(0, 0)
    if i < points.shape[0]:
        raw_part = partial_ray_count(tree, p, wp.vec3d(1.0, 1.0, 1.0), ex, part, 32, 1)
    raw_tile = wp.tile_reshape(wp.tile(raw_part), shape=(1, 32))
    raw_sums = wp.tile_sum(raw_tile, axis=1)
    raw = raw_sums[row]
    first_part = wp.vec2i(0, 0)
    if i < points.shape[0]:
        first_part = partial_ray_count(tree, p, wp.vec3d(0.8170638283521119, 0.33141623424177, 0.4713172342319), ex, part, 32, 1)
    first_tile = wp.tile_reshape(wp.tile(first_part), shape=(1, 32))
    first_sums = wp.tile_sum(first_tile, axis=1)
    first = first_sums[row]
    second_part = wp.vec2i(0, 0)
    if i < points.shape[0] and first[1] != 0:
        second_part = partial_ray_count(tree, p, wp.vec3d(-0.213173497731, 0.93127172131, 0.44721714319), ex, part, 32, 1)
    second_tile = wp.tile_reshape(wp.tile(second_part), shape=(1, 32))
    second_sums = wp.tile_sum(second_tile, axis=1)
    second = second_sums[row]
    third_part = wp.vec2i(0, 0)
    if i < points.shape[0] and first[1] != 0 and second[1] != 0:
        third_part = partial_ray_count(tree, p, wp.vec3d(0.63719234173, -0.72819341297, 0.27191345121), ex, part, 32, 1)
    third_tile = wp.tile_reshape(wp.tile(third_part), shape=(1, 32))
    third_sums = wp.tile_sum(third_tile, axis=1)
    third = third_sums[row]
    side = first[0]
    if first[1] != 0:
        side = second[0]
        if second[1] != 0:
            side = third[0]
            if third[1] != 0:
                side = 128
    if part == 0 and i < points.shape[0]:
        out[i, 2] = side
        out[i, 0] = raw[0]
        out[i, 1] = int(raw[1] != 0)


@wp.kernel
def subtree_32_side(tree: SurfaceTree, points: wp.array[wp.vec3d], excluded: wp.array[int], out: wp.array2d[int]):
    block, lane = wp.tid()
    row = lane // 32
    part = lane % 32
    i = block * 1 + row
    p = wp.vec3d(0.0)
    ex = int(-1)
    if i < points.shape[0]:
        p = points[i]
        ex = excluded[i]
    first_part = wp.vec2i(0, 0)
    if i < points.shape[0]:
        first_part = partial_ray_count(tree, p, wp.vec3d(0.8170638283521119, 0.33141623424177, 0.4713172342319), ex, part, 32, 1)
    first_tile = wp.tile_reshape(wp.tile(first_part), shape=(1, 32))
    first_sums = wp.tile_sum(first_tile, axis=1)
    first = first_sums[row]
    second_part = wp.vec2i(0, 0)
    if i < points.shape[0] and first[1] != 0:
        second_part = partial_ray_count(tree, p, wp.vec3d(-0.213173497731, 0.93127172131, 0.44721714319), ex, part, 32, 1)
    second_tile = wp.tile_reshape(wp.tile(second_part), shape=(1, 32))
    second_sums = wp.tile_sum(second_tile, axis=1)
    second = second_sums[row]
    third_part = wp.vec2i(0, 0)
    if i < points.shape[0] and first[1] != 0 and second[1] != 0:
        third_part = partial_ray_count(tree, p, wp.vec3d(0.63719234173, -0.72819341297, 0.27191345121), ex, part, 32, 1)
    third_tile = wp.tile_reshape(wp.tile(third_part), shape=(1, 32))
    third_sums = wp.tile_sum(third_tile, axis=1)
    third = third_sums[row]
    side = first[0]
    if first[1] != 0:
        side = second[0]
        if second[1] != 0:
            side = third[0]
            if third[1] != 0:
                side = 128
    if part == 0 and i < points.shape[0]:
        out[i, 2] = side


