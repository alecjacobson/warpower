"""Private #197 fused retained queries, not a production implementation."""
import warp as wp
from warpower._surface_bvh import SurfaceTree, closest_point
from warpower._wide_center import exact_geometry, uncertain
from cooperative_rays_v2 import partial_ray_count
wp.set_module_options({'enable_backward': False, 'fast_math': False})

@wp.struct
class Geometry:
    center: wp.vec3d
    radius: wp.float64
    quality: wp.float64
    error: int

@wp.func(inline=True)
def retained_geometry(points: wp.array[wp.vec3], tets: wp.array2d[int], i: int):
    result = Geometry()
    ids = wp.vec4i(tets[i, 0], tets[i, 1], tets[i, 2], tets[i, 3])
    for k in range(4):
        if ids[k] < 0 or ids[k] >= points.shape[0]:
            result.error = 4
            return result
        p = points[ids[k]]
        if not wp.isfinite(p[0]) or not wp.isfinite(p[1]) or not wp.isfinite(p[2]):
            result.error = 4
            return result
    p0 = wp.vec3d(points[ids[0]])
    a = wp.vec3d(points[ids[1]]) - p0
    b = wp.vec3d(points[ids[2]]) - p0
    c = wp.vec3d(points[ids[3]]) - p0
    bc, ca, ab = wp.cross(b, c), wp.cross(c, a), wp.cross(a, b)
    determinant = wp.dot(a, bc)
    area2 = wp.length(bc) + wp.length(ca) + wp.length(ab) + wp.length(wp.cross(b - a, c - a))
    center, radius = wp.vec3d(), wp.float64(0.0)
    if uncertain(a, b, c, determinant):
        geometry = exact_geometry(points[ids[0]], points[ids[1]], points[ids[2]], points[ids[3]])
        if geometry.valid == 0:
            result.error = 8
            return result
        determinant = wp.float64(geometry.determinant)
        center = wp.vec3d(geometry.center)
        radius = wp.float64(geometry.radius)
        area2 = wp.float64(geometry.area2)
    else:
        if determinant <= wp.float64(0.0):
            result.error = 8
            return result
        offset = (wp.dot(a, a) * bc + wp.dot(b, b) * ca + wp.dot(c, c) * ab) / (
            wp.float64(2.0) * determinant
        )
        center, radius = p0 + offset, wp.length(offset)
    if not wp.isfinite(radius) or radius <= wp.float64(0.0):
        result.error = 8
        return result
    result.center = center
    result.radius = radius
    result.quality = wp.min(wp.float64(1.0), wp.float64(3.0) * determinant / (area2 * radius))
    return result

@wp.kernel
def retained_1(
    tree: SurfaceTree, points: wp.array[wp.vec3], tets: wp.array2d[int],
    count: wp.array[int], tolerance: wp.float64, error: wp.array[int],
    keep: wp.array[int], quality: wp.array[wp.float64],
):
    block, lane = wp.tid()
    row = lane // 1
    part = lane % 1
    i = block * 32 + row
    active = i < keep.shape[0] and i < count[0] and error[0] == 0
    if part == 0 and i < keep.shape[0]:
        keep[i] = 0
    geometry = Geometry()
    if active:
        geometry = retained_geometry(points, tets, i)
        if geometry.error != 0:
            wp.atomic_or(error, 0, geometry.error)
    center = geometry.center
    radius = geometry.radius
    first_part = wp.vec2i(0, 0)
    if active and geometry.error == 0:
        first_part = partial_ray_count(tree, center, wp.vec3d(0.8170638283521119, 0.33141623424177, 0.4713172342319), -1, part, 1, 0)
    first = first_part
    second_part = wp.vec2i(0, 0)
    if active and geometry.error == 0 and first[1] != 0:
        second_part = partial_ray_count(tree, center, wp.vec3d(-0.213173497731, 0.93127172131, 0.44721714319), -1, part, 1, 0)
    second = second_part
    third_part = wp.vec2i(0, 0)
    if active and geometry.error == 0 and first[1] != 0 and second[1] != 0:
        third_part = partial_ray_count(tree, center, wp.vec3d(0.63719234173, -0.72819341297, 0.27191345121), -1, part, 1, 0)
    third = third_part
    side = first[0]
    if first[1] != 0:
        side = second[0]
        if second[1] != 0:
            side = third[0]
            if third[1] != 0:
                side = 128
    if part != 0 or not active or geometry.error != 0:
        return
    retained = side == 1
    if not retained or radius >= wp.float64(1.0e140):
        limit = wp.float64(1.0e150)
        # Preserve the full classifier's finite search-limit failure behavior
        # for extreme circumradii as well as ambiguous ray results.
        bounded = side == 0 and tolerance > wp.float64(0.0) and radius < wp.float64(1.0e140)
        if bounded:
            # Expand the *search* cutoff, then apply the original strict d/R
            # predicate below. This avoids changing threshold ties through a
            # rounded product/square compared to the original division.
            limit = wp.min(
                limit,
                wp.max(
                    wp.float64(1.0e-150),
                    tolerance * radius * (wp.float64(1.0) + wp.float64(1.0e-12)),
                ),
            )
        hit = closest_point(tree, center, limit)
        if hit.face < 0:
            if not bounded or limit == wp.float64(1.0e150):
                wp.atomic_or(error, 0, 16)
                return
        else:
            if not wp.isfinite(hit.distance2):
                wp.atomic_or(error, 0, 16)
                return
            # Extraction treats an exact boundary center as inside, including
            # tolerance zero and ambiguous outgoing ray directions.
            if hit.distance2 == wp.float64(0.0):
                retained = True
            elif side != 0 and side != 1:
                wp.atomic_or(error, 0, 16)
                return
            else:
                retained = side == 1 or wp.sqrt(hit.distance2) / radius < tolerance
    keep[i] = int(retained)
    quality[i] = geometry.quality

@wp.kernel
def retained_2(
    tree: SurfaceTree, points: wp.array[wp.vec3], tets: wp.array2d[int],
    count: wp.array[int], tolerance: wp.float64, error: wp.array[int],
    keep: wp.array[int], quality: wp.array[wp.float64],
):
    block, lane = wp.tid()
    row = lane // 2
    part = lane % 2
    i = block * 16 + row
    active = i < keep.shape[0] and i < count[0] and error[0] == 0
    if part == 0 and i < keep.shape[0]:
        keep[i] = 0
    geometry = Geometry()
    if active:
        geometry = retained_geometry(points, tets, i)
        if geometry.error != 0:
            wp.atomic_or(error, 0, geometry.error)
    center = geometry.center
    radius = geometry.radius
    first_part = wp.vec2i(0, 0)
    if active and geometry.error == 0:
        first_part = partial_ray_count(tree, center, wp.vec3d(0.8170638283521119, 0.33141623424177, 0.4713172342319), -1, part, 2, 0)
    first_tile = wp.tile_reshape(wp.tile(first_part, preserve_type=True), shape=(16, 2))
    first_sums = wp.tile_sum(first_tile, axis=1)
    first = first_sums[row]
    second_part = wp.vec2i(0, 0)
    if active and geometry.error == 0 and first[1] != 0:
        second_part = partial_ray_count(tree, center, wp.vec3d(-0.213173497731, 0.93127172131, 0.44721714319), -1, part, 2, 0)
    second_tile = wp.tile_reshape(wp.tile(second_part, preserve_type=True), shape=(16, 2))
    second_sums = wp.tile_sum(second_tile, axis=1)
    second = second_sums[row]
    third_part = wp.vec2i(0, 0)
    if active and geometry.error == 0 and first[1] != 0 and second[1] != 0:
        third_part = partial_ray_count(tree, center, wp.vec3d(0.63719234173, -0.72819341297, 0.27191345121), -1, part, 2, 0)
    third_tile = wp.tile_reshape(wp.tile(third_part, preserve_type=True), shape=(16, 2))
    third_sums = wp.tile_sum(third_tile, axis=1)
    third = third_sums[row]
    side = first[0]
    if first[1] != 0:
        side = second[0]
        if second[1] != 0:
            side = third[0]
            if third[1] != 0:
                side = 128
    if part != 0 or not active or geometry.error != 0:
        return
    retained = side == 1
    if not retained or radius >= wp.float64(1.0e140):
        limit = wp.float64(1.0e150)
        # Preserve the full classifier's finite search-limit failure behavior
        # for extreme circumradii as well as ambiguous ray results.
        bounded = side == 0 and tolerance > wp.float64(0.0) and radius < wp.float64(1.0e140)
        if bounded:
            # Expand the *search* cutoff, then apply the original strict d/R
            # predicate below. This avoids changing threshold ties through a
            # rounded product/square compared to the original division.
            limit = wp.min(
                limit,
                wp.max(
                    wp.float64(1.0e-150),
                    tolerance * radius * (wp.float64(1.0) + wp.float64(1.0e-12)),
                ),
            )
        hit = closest_point(tree, center, limit)
        if hit.face < 0:
            if not bounded or limit == wp.float64(1.0e150):
                wp.atomic_or(error, 0, 16)
                return
        else:
            if not wp.isfinite(hit.distance2):
                wp.atomic_or(error, 0, 16)
                return
            # Extraction treats an exact boundary center as inside, including
            # tolerance zero and ambiguous outgoing ray directions.
            if hit.distance2 == wp.float64(0.0):
                retained = True
            elif side != 0 and side != 1:
                wp.atomic_or(error, 0, 16)
                return
            else:
                retained = side == 1 or wp.sqrt(hit.distance2) / radius < tolerance
    keep[i] = int(retained)
    quality[i] = geometry.quality

@wp.kernel
def retained_4(
    tree: SurfaceTree, points: wp.array[wp.vec3], tets: wp.array2d[int],
    count: wp.array[int], tolerance: wp.float64, error: wp.array[int],
    keep: wp.array[int], quality: wp.array[wp.float64],
):
    block, lane = wp.tid()
    row = lane // 4
    part = lane % 4
    i = block * 8 + row
    active = i < keep.shape[0] and i < count[0] and error[0] == 0
    if part == 0 and i < keep.shape[0]:
        keep[i] = 0
    geometry = Geometry()
    if active:
        geometry = retained_geometry(points, tets, i)
        if geometry.error != 0:
            wp.atomic_or(error, 0, geometry.error)
    center = geometry.center
    radius = geometry.radius
    first_part = wp.vec2i(0, 0)
    if active and geometry.error == 0:
        first_part = partial_ray_count(tree, center, wp.vec3d(0.8170638283521119, 0.33141623424177, 0.4713172342319), -1, part, 4, 0)
    first_tile = wp.tile_reshape(wp.tile(first_part, preserve_type=True), shape=(8, 4))
    first_sums = wp.tile_sum(first_tile, axis=1)
    first = first_sums[row]
    second_part = wp.vec2i(0, 0)
    if active and geometry.error == 0 and first[1] != 0:
        second_part = partial_ray_count(tree, center, wp.vec3d(-0.213173497731, 0.93127172131, 0.44721714319), -1, part, 4, 0)
    second_tile = wp.tile_reshape(wp.tile(second_part, preserve_type=True), shape=(8, 4))
    second_sums = wp.tile_sum(second_tile, axis=1)
    second = second_sums[row]
    third_part = wp.vec2i(0, 0)
    if active and geometry.error == 0 and first[1] != 0 and second[1] != 0:
        third_part = partial_ray_count(tree, center, wp.vec3d(0.63719234173, -0.72819341297, 0.27191345121), -1, part, 4, 0)
    third_tile = wp.tile_reshape(wp.tile(third_part, preserve_type=True), shape=(8, 4))
    third_sums = wp.tile_sum(third_tile, axis=1)
    third = third_sums[row]
    side = first[0]
    if first[1] != 0:
        side = second[0]
        if second[1] != 0:
            side = third[0]
            if third[1] != 0:
                side = 128
    if part != 0 or not active or geometry.error != 0:
        return
    retained = side == 1
    if not retained or radius >= wp.float64(1.0e140):
        limit = wp.float64(1.0e150)
        # Preserve the full classifier's finite search-limit failure behavior
        # for extreme circumradii as well as ambiguous ray results.
        bounded = side == 0 and tolerance > wp.float64(0.0) and radius < wp.float64(1.0e140)
        if bounded:
            # Expand the *search* cutoff, then apply the original strict d/R
            # predicate below. This avoids changing threshold ties through a
            # rounded product/square compared to the original division.
            limit = wp.min(
                limit,
                wp.max(
                    wp.float64(1.0e-150),
                    tolerance * radius * (wp.float64(1.0) + wp.float64(1.0e-12)),
                ),
            )
        hit = closest_point(tree, center, limit)
        if hit.face < 0:
            if not bounded or limit == wp.float64(1.0e150):
                wp.atomic_or(error, 0, 16)
                return
        else:
            if not wp.isfinite(hit.distance2):
                wp.atomic_or(error, 0, 16)
                return
            # Extraction treats an exact boundary center as inside, including
            # tolerance zero and ambiguous outgoing ray directions.
            if hit.distance2 == wp.float64(0.0):
                retained = True
            elif side != 0 and side != 1:
                wp.atomic_or(error, 0, 16)
                return
            else:
                retained = side == 1 or wp.sqrt(hit.distance2) / radius < tolerance
    keep[i] = int(retained)
    quality[i] = geometry.quality

