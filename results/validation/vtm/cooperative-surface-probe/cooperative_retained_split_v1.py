"""Private staged retained classifier. Extra scratch: 32 bytes/cell capacity."""
import warp as wp
from warpower._surface_bvh import SurfaceTree, closest_point
from cooperative_rays_v2 import partial_ray_count
from cooperative_retained_v1 import Geometry, retained_geometry
wp.set_module_options({'enable_backward': False, 'fast_math': False})

@wp.kernel
def prepare_retained(
    tree: SurfaceTree, points: wp.array[wp.vec3], tets: wp.array2d[int],
    count: wp.array[int], tolerance: wp.float64, error: wp.array[int],
    keep: wp.array[int], quality: wp.array[wp.float64],
    centers: wp.array[wp.vec3d], radii: wp.array[wp.float64],
):
    i = wp.tid()
    keep[i] = 0
    if i >= count[0] or error[0] != 0:
        return
    geometry = retained_geometry(points, tets, i)
    if geometry.error != 0:
        wp.atomic_or(error, 0, geometry.error)
        return
    centers[i] = geometry.center
    radii[i] = geometry.radius
    quality[i] = geometry.quality

@wp.kernel
def query_1(
    tree: SurfaceTree, points: wp.array[wp.vec3], tets: wp.array2d[int],
    count: wp.array[int], tolerance: wp.float64, error: wp.array[int],
    keep: wp.array[int], quality: wp.array[wp.float64],
    centers: wp.array[wp.vec3d], radii: wp.array[wp.float64],
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
        geometry.center = centers[i]
        geometry.radius = radii[i]
        geometry.quality = quality[i]
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
def query_2(
    tree: SurfaceTree, points: wp.array[wp.vec3], tets: wp.array2d[int],
    count: wp.array[int], tolerance: wp.float64, error: wp.array[int],
    keep: wp.array[int], quality: wp.array[wp.float64],
    centers: wp.array[wp.vec3d], radii: wp.array[wp.float64],
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
        geometry.center = centers[i]
        geometry.radius = radii[i]
        geometry.quality = quality[i]
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
def query_4(
    tree: SurfaceTree, points: wp.array[wp.vec3], tets: wp.array2d[int],
    count: wp.array[int], tolerance: wp.float64, error: wp.array[int],
    keep: wp.array[int], quality: wp.array[wp.float64],
    centers: wp.array[wp.vec3d], radii: wp.array[wp.float64],
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
        geometry.center = centers[i]
        geometry.radius = radii[i]
        geometry.quality = quality[i]
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

