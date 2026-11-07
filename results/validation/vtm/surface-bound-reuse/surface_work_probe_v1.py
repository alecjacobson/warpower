"""Private query-work counters, not performance timing or production code."""
import warp as wp
from warpower._surface_bvh import SurfaceTree, Closest, box_distance, closest_triangle, ray_box, closest_point, material_side
wp.set_module_options({'enable_backward': False, 'fast_math': False})
Counts = wp.types.vector(length=9, dtype=wp.int32)

@wp.struct
class NearestWork:
    hit: Closest
    work: Counts

@wp.struct
class RayWork:
    result: wp.vec2i
    work: Counts

@wp.func(inline=True)
def counted_nearest(tree: SurfaceTree, p: wp.vec3d, limit: wp.float64):
    work = Counts()
    work[0] = 1
    work[7] = 1
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
        work[1] += 1
        work[8] += 1
        if box_distance(tree, node, p) >= result.distance2:
            work[2] += 1
            continue
        if node >= tree.leaf_base:
            work[4] += 1
            first = 4 * (node - tree.leaf_base)
            for j in range(first, wp.min(first + 4, tree.n)):
                work[5] += 1
                f = tree.ids[j]
                a = wp.vec3d(tree.points[tree.indices[3 * f]])
                b = wp.vec3d(tree.points[tree.indices[3 * f + 1]])
                c = wp.vec3d(tree.points[tree.indices[3 * f + 2]])
                q = closest_triangle(p, a, b, c)
                distance = wp.length_sq(p - q)
                if distance < result.distance2:
                    work[6] += 1
                    result.point = q
                    result.face = f
                    result.distance2 = distance
        else:
            work[3] += 1
            work[8] += 2
            left, right = int(tree.left[node]), int(tree.right[node])
            if box_distance(tree, left, p) < box_distance(tree, right, p):
                left, right = right, left
            stack[size] = left
            stack[size + 1] = right
            size += 2
            work[7] = wp.max(work[7], size)
    observed = NearestWork()
    observed.hit = result
    observed.work = work
    return observed


@wp.func(inline=True)
def counted_ray(tree: SurfaceTree, origin: wp.vec3d, direction: wp.vec3d, excluded_face: int = -1):
    # Return signed intersections and an ambiguity flag. Shared-edge hits are
    # retried with another direction instead of double-counting adjacent faces.
    work = Counts()
    work[0] = 1
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
    node = int(1)
    while node > 0:
        work[1] += 1
        work[8] += 1
        if not ray_box(tree, node, origin, inverse):
            work[2] += 1
            node = tree.escape[node]
            continue
        if node >= tree.leaf_base:
            work[4] += 1
            first = 4 * (node - tree.leaf_base)
            for j in range(first, wp.min(first + 4, tree.n)):
                f = tree.ids[j]
                if f == excluded_face:
                    continue
                work[5] += 1
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
                        work[7] += 1
                    work[6] += 1
                    crossings += wp.where(det > wp.float64(0.0), 1, -1)
            node = tree.escape[node]
        else:
            work[3] += 1
            node = tree.right[node]
    observed = RayWork()
    observed.result = wp.vec2i(crossings, ambiguous)
    observed.work = work
    return observed


@wp.func(inline=True)
def counted_side(tree: SurfaceTree, p: wp.vec3d):
    first = counted_ray(tree, p, wp.vec3d(0.8170638283521119, 0.33141623424177, 0.4713172342319))
    work = first.work
    result = first.result
    if result[1] != 0:
        second = counted_ray(tree, p, wp.vec3d(-0.213173497731, 0.93127172131, 0.44721714319))
        work += second.work
        result = second.result
    if result[1] != 0:
        third = counted_ray(tree, p, wp.vec3d(0.63719234173, -0.72819341297, 0.27191345121))
        work += third.work
        result = third.result
    output = RayWork()
    output.result = result
    output.work = work
    return output


@wp.func(inline=True)
def same(a: Closest, b: Closest):
    return int(a.face == b.face and a.distance2 == b.distance2
               and a.point[0] == b.point[0] and a.point[1] == b.point[1] and a.point[2] == b.point[2])


@wp.kernel
def observe(
    tree: SurfaceTree, centers: wp.array[wp.vec3d], radii: wp.array[wp.float64],
    count: wp.array[int], error: wp.array[int], tolerance: wp.float64,
    full_work: wp.array[Counts], retained_work: wp.array[Counts], ray_work: wp.array[Counts],
    status: wp.array2d[int],
):
    i = wp.tid()
    if error[0] != 0 or i >= count[0]:
        return
    p = centers[i]
    r = radii[i]
    full = counted_nearest(tree, p, wp.float64(1.0e150))
    reference = closest_point(tree, p, wp.float64(1.0e150))
    full_work[i] = full.work
    status[i, 0] = 1 - same(reference, full.hit)
    status[i, 1] = int(reference.face >= 0 and wp.isfinite(reference.distance2))
    rays = counted_side(tree, p)
    side = rays.result[0]
    if rays.result[1] != 0:
        side = 128
    ray_work[i] = rays.work
    status[i, 2] = int(side != material_side(tree, p))
    status[i, 3] = side
    nearest_work = Counts()
    if side != 1 or r >= wp.float64(1.0e140):
        limit = wp.float64(1.0e150)
        bounded = side == 0 and tolerance > wp.float64(0.0) and r < wp.float64(1.0e140)
        if bounded:
            limit = wp.min(limit, wp.max(wp.float64(1.0e-150), tolerance*r*(wp.float64(1.0)+wp.float64(1.0e-12))))
        status[i, 4] = 1 + int(bounded)
        limited = counted_nearest(tree, p, limit)
        expected = closest_point(tree, p, limit)
        nearest_work = limited.work
        status[i, 5] = 1 - same(expected, limited.hit)
        status[i, 6] = int(expected.face >= 0)
    retained_work[i] = nearest_work
