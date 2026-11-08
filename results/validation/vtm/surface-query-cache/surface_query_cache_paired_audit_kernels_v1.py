import warp as wp
from warpower._extraction import SelectionCenter
from warpower._surface_bvh import Closest, SurfaceTree, closest_point, closest_triangle, material_side
from warpower._wide_center import exact_geometry, uncertain
from warpower._surface_query_cache import QueryState, lookup, same_center, cached_side, cached_nearest
wp.set_module_options({"enable_backward": False, "fast_math": False})

@wp.func(inline=True)
def audit_side(tree: SurfaceTree, center: wp.vec3d, previous: QueryState, table: wp.array[int], stats: wp.array[int]):
    j=lookup(center, previous, table)
    wp.atomic_add(stats,0,1)
    if j>=0 and previous.rays[j]!=0:wp.atomic_add(stats,1,1)
    a=cached_side(tree,center,previous,table)
    b=material_side(tree,center)
    if a!=b:wp.atomic_add(stats,2,1)
    return a

@wp.func(inline=True)
def audit_nearest(tree: SurfaceTree, center: wp.vec3d, limit: wp.float64, previous: QueryState, table: wp.array[int], stats: wp.array[int]):
    j=lookup(center,previous,table)
    wp.atomic_add(stats,3,1)
    if j>=0 and previous.nearest[j]!=0 and wp.cast(limit,wp.uint64)==wp.cast(previous.limits[j],wp.uint64):wp.atomic_add(stats,4,1)
    a=cached_nearest(tree,center,limit,previous,table)
    b=closest_point(tree,center,limit)
    if a.face!=b.face:wp.atomic_add(stats,5,1)
    if wp.cast(a.distance2,wp.uint64)!=wp.cast(b.distance2,wp.uint64):wp.atomic_add(stats,6,1)
    if not same_center(a.point,b.point):wp.atomic_add(stats,7,1)
    return a

@wp.kernel
def classify_cached(tree: SurfaceTree, points: wp.array[wp.vec3], tets: wp.array2d[int], count: wp.array[int], tolerance: wp.float64, error: wp.array[int], centers: wp.array[wp.vec3d], radii: wp.array[wp.float64], distances: wp.array[wp.float64], keep: wp.array[int], volumes: wp.array[wp.float64], quality: wp.array[wp.float64], probe: QueryState, previous: QueryState, table: wp.array[int], stats: wp.array[int]):
    i = wp.tid()
    keep[i] = 0
    probe.valid[i] = 0
    probe.rays[i] = 0
    probe.nearest[i] = 0
    if error[0] != 0 or i >= count[0]:
        return
    ids = wp.vec4i(tets[i, 0], tets[i, 1], tets[i, 2], tets[i, 3])
    for k in range(4):
        if ids[k] < 0 or ids[k] >= points.shape[0]:
            wp.atomic_or(error, 0, 4)
            return
        p = points[ids[k]]
        if not wp.isfinite(p[0]) or not wp.isfinite(p[1]) or (not wp.isfinite(p[2])):
            wp.atomic_or(error, 0, 4)
            return
    p0 = wp.vec3d(points[ids[0]])
    (a, b, c) = (wp.vec3d(points[ids[1]]) - p0, wp.vec3d(points[ids[2]]) - p0, wp.vec3d(points[ids[3]]) - p0)
    (bc, ca, ab) = (wp.cross(b, c), wp.cross(c, a), wp.cross(a, b))
    determinant = wp.dot(a, bc)
    area2 = wp.length(bc) + wp.length(ca) + wp.length(ab) + wp.length(wp.cross(b - a, c - a))
    (center, radius) = (wp.vec3d(), wp.float64(0.0))
    if uncertain(a, b, c, determinant):
        geometry = exact_geometry(points[ids[0]], points[ids[1]], points[ids[2]], points[ids[3]])
        if geometry.valid == 0:
            wp.atomic_or(error, 0, 8)
            return
        determinant = wp.float64(geometry.determinant)
        center = wp.vec3d(geometry.center)
        radius = wp.float64(geometry.radius)
        area2 = wp.float64(geometry.area2)
    else:
        if determinant <= wp.float64(0.0):
            wp.atomic_or(error, 0, 8)
            return
        offset = (wp.dot(a, a) * bc + wp.dot(b, b) * ca + wp.dot(c, c) * ab) / (wp.float64(2.0) * determinant)
        (center, radius) = (p0 + offset, wp.length(offset))
    if not wp.isfinite(radius) or radius <= wp.float64(0.0):
        wp.atomic_or(error, 0, 8)
        return
    probe.centers[i] = center
    probe.valid[i] = 1
    probe.nearest[i] = 1
    probe.limits[i] = wp.float64(1e+150)
    hit = audit_nearest(tree, center, wp.float64(1e+150), previous, table, stats)
    probe.distances[i] = hit.distance2
    probe.faces[i] = hit.face
    probe.hit_points[i] = hit.point
    if hit.face < 0 or not wp.isfinite(hit.distance2):
        wp.atomic_or(error, 0, 16)
        return
    side = int(1)
    if hit.distance2 > wp.float64(0.0):
        probe.rays[i] = 1
        side = audit_side(tree, center, previous, table, stats)
        probe.sides[i] = side
    if side != 0 and side != 1:
        wp.atomic_or(error, 0, 16)
        return
    distance = wp.sqrt(hit.distance2)
    centers[i] = center
    radii[i] = radius
    distances[i] = wp.where(side == 1, -distance, distance)
    keep[i] = int(side == 1 or distance / radius < tolerance)
    volumes[i] = determinant / wp.float64(6.0)
    quality[i] = wp.min(wp.float64(1.0), wp.float64(3.0) * determinant / (area2 * radius))

@wp.kernel
def retained_cached(tree: SurfaceTree, points: wp.array[wp.vec3], tets: wp.array2d[int], count: wp.array[int], tolerance: wp.float64, error: wp.array[int], keep: wp.array[int], quality: wp.array[wp.float64], probe: QueryState, previous: QueryState, table: wp.array[int], stats: wp.array[int]):
    i = wp.tid()
    keep[i] = 0
    probe.valid[i] = 0
    probe.rays[i] = 0
    probe.nearest[i] = 0
    if error[0] != 0 or i >= count[0]:
        return
    ids = wp.vec4i(tets[i, 0], tets[i, 1], tets[i, 2], tets[i, 3])
    for k in range(4):
        if ids[k] < 0 or ids[k] >= points.shape[0]:
            wp.atomic_or(error, 0, 4)
            return
        p = points[ids[k]]
        if not wp.isfinite(p[0]) or not wp.isfinite(p[1]) or (not wp.isfinite(p[2])):
            wp.atomic_or(error, 0, 4)
            return
    p0 = wp.vec3d(points[ids[0]])
    a = wp.vec3d(points[ids[1]]) - p0
    b = wp.vec3d(points[ids[2]]) - p0
    c = wp.vec3d(points[ids[3]]) - p0
    (bc, ca, ab) = (wp.cross(b, c), wp.cross(c, a), wp.cross(a, b))
    determinant = wp.dot(a, bc)
    area2 = wp.length(bc) + wp.length(ca) + wp.length(ab) + wp.length(wp.cross(b - a, c - a))
    (center, radius) = (wp.vec3d(), wp.float64(0.0))
    if uncertain(a, b, c, determinant):
        geometry = exact_geometry(points[ids[0]], points[ids[1]], points[ids[2]], points[ids[3]])
        if geometry.valid == 0:
            wp.atomic_or(error, 0, 8)
            return
        determinant = wp.float64(geometry.determinant)
        center = wp.vec3d(geometry.center)
        radius = wp.float64(geometry.radius)
        area2 = wp.float64(geometry.area2)
    else:
        if determinant <= wp.float64(0.0):
            wp.atomic_or(error, 0, 8)
            return
        offset = (wp.dot(a, a) * bc + wp.dot(b, b) * ca + wp.dot(c, c) * ab) / (wp.float64(2.0) * determinant)
        (center, radius) = (p0 + offset, wp.length(offset))
    if not wp.isfinite(radius) or radius <= wp.float64(0.0):
        wp.atomic_or(error, 0, 8)
        return
    probe.centers[i] = center
    probe.valid[i] = 1
    probe.rays[i] = 1
    side = audit_side(tree, center, previous, table, stats)
    probe.sides[i] = side
    retained = side == 1
    if not retained or radius >= wp.float64(1e+140):
        limit = wp.float64(1e+150)
        bounded = side == 0 and tolerance > wp.float64(0.0) and (radius < wp.float64(1e+140))
        if bounded:
            limit = wp.min(limit, wp.max(wp.float64(1e-150), tolerance * radius * (wp.float64(1.0) + wp.float64(1e-12))))
        probe.nearest[i] = 1
        probe.limits[i] = limit
        hit = audit_nearest(tree, center, limit, previous, table, stats)
        probe.distances[i] = hit.distance2
        probe.faces[i] = hit.face
        probe.hit_points[i] = hit.point
        if hit.face < 0:
            if not bounded or limit == wp.float64(1e+150):
                wp.atomic_or(error, 0, 16)
                return
        else:
            if not wp.isfinite(hit.distance2):
                wp.atomic_or(error, 0, 16)
                return
            if hit.distance2 == wp.float64(0.0):
                retained = True
            elif side != 0 and side != 1:
                wp.atomic_or(error, 0, 16)
                return
            else:
                retained = side == 1 or wp.sqrt(hit.distance2) / radius < tolerance
    keep[i] = int(retained)
    quality[i] = wp.min(wp.float64(1.0), wp.float64(3.0) * determinant / (area2 * radius))

@wp.func(inline=True)
def selection_center_cached(tree: SurfaceTree, center: wp.vec3d, radius: wp.float64, tolerance: wp.float64, probe: QueryState, slot: int, previous: QueryState, table: wp.array[int], stats: wp.array[int]):
    """Private retention and distance sign; magnitude is not a distance output.

    Radius/geometry are validated by the caller. Unproved or non-inside centers
    keep the complete query, including exact-zero and strict threshold handling.
    """
    result = SelectionCenter()
    proved = False
    if tree.n > 0 and wp.abs(center[0]) < wp.float64(1e+140) and (wp.abs(center[1]) < wp.float64(1e+140)) and (wp.abs(center[2]) < wp.float64(1e+140)):
        f = tree.ids[0]
        x = wp.vec3d(tree.points[tree.indices[3 * f]])
        y = wp.vec3d(tree.points[tree.indices[3 * f + 1]])
        z = wp.vec3d(tree.points[tree.indices[3 * f + 2]])
        q = closest_triangle(center, x, y, z)
        witness_distance = wp.length_sq(center - q)
        limit = wp.float64(1e+150)
        proved = wp.isfinite(witness_distance) and witness_distance < limit * limit
    side = int(-1)
    if proved:
        probe.rays[slot] = 1
        side = audit_side(tree, center, previous, table, stats)
        probe.sides[slot] = side
    if proved and side == 1:
        result.distance_sign = wp.float64(-1.0)
        result.retained = 1
    else:
        probe.nearest[slot] = 1
        probe.limits[slot] = wp.float64(1e+150)
        hit = audit_nearest(tree, center, wp.float64(1e+150), previous, table, stats)
        probe.distances[slot] = hit.distance2
        probe.faces[slot] = hit.face
        probe.hit_points[slot] = hit.point
        if hit.face < 0 or not wp.isfinite(hit.distance2):
            result.error = 16
            return result
        if hit.distance2 == wp.float64(0.0):
            side = 1
        elif not proved:
            probe.rays[slot] = 1
            side = audit_side(tree, center, previous, table, stats)
            probe.sides[slot] = side
        if side != 0 and side != 1:
            result.error = 16
            return result
        distance = wp.sqrt(hit.distance2)
        result.distance_sign = wp.where(side == 1, -distance, distance)
        result.retained = int(side == 1 or distance / radius < tolerance)
    return result

@wp.kernel
def selection_cached(tree: SurfaceTree, points: wp.array[wp.vec3], tets: wp.array2d[int], count: wp.array[int], tolerance: wp.float64, error: wp.array[int], centers: wp.array[wp.vec3d], radii: wp.array[wp.float64], distances: wp.array[wp.float64], keep: wp.array[int], volumes: wp.array[wp.float64], quality: wp.array[wp.float64], probe: QueryState, previous: QueryState, table: wp.array[int], stats: wp.array[int]):
    i = wp.tid()
    keep[i] = 0
    probe.valid[i] = 0
    probe.rays[i] = 0
    probe.nearest[i] = 0
    if error[0] != 0 or i >= count[0]:
        return
    ids = wp.vec4i(tets[i, 0], tets[i, 1], tets[i, 2], tets[i, 3])
    for k in range(4):
        if ids[k] < 0 or ids[k] >= points.shape[0]:
            wp.atomic_or(error, 0, 4)
            return
        p = points[ids[k]]
        if not wp.isfinite(p[0]) or not wp.isfinite(p[1]) or (not wp.isfinite(p[2])):
            wp.atomic_or(error, 0, 4)
            return
    p0 = wp.vec3d(points[ids[0]])
    (a, b, c) = (wp.vec3d(points[ids[1]]) - p0, wp.vec3d(points[ids[2]]) - p0, wp.vec3d(points[ids[3]]) - p0)
    (bc, ca, ab) = (wp.cross(b, c), wp.cross(c, a), wp.cross(a, b))
    determinant = wp.dot(a, bc)
    area2 = wp.length(bc) + wp.length(ca) + wp.length(ab) + wp.length(wp.cross(b - a, c - a))
    (center, radius) = (wp.vec3d(), wp.float64(0.0))
    if uncertain(a, b, c, determinant):
        geometry = exact_geometry(points[ids[0]], points[ids[1]], points[ids[2]], points[ids[3]])
        if geometry.valid == 0:
            wp.atomic_or(error, 0, 8)
            return
        determinant = wp.float64(geometry.determinant)
        center = wp.vec3d(geometry.center)
        radius = wp.float64(geometry.radius)
        area2 = wp.float64(geometry.area2)
    else:
        if determinant <= wp.float64(0.0):
            wp.atomic_or(error, 0, 8)
            return
        offset = (wp.dot(a, a) * bc + wp.dot(b, b) * ca + wp.dot(c, c) * ab) / (wp.float64(2.0) * determinant)
        (center, radius) = (p0 + offset, wp.length(offset))
    if not wp.isfinite(radius) or radius <= wp.float64(0.0):
        wp.atomic_or(error, 0, 8)
        return
    probe.centers[i] = center
    probe.valid[i] = 1
    status = selection_center_cached(tree, center, radius, tolerance, probe, i, previous, table, stats)
    if status.error != 0:
        wp.atomic_or(error, 0, status.error)
        return
    distances[i] = status.distance_sign
    keep[i] = status.retained
    centers[i] = center
    radii[i] = radius
    volumes[i] = determinant / wp.float64(6.0)
    quality[i] = wp.min(wp.float64(1.0), wp.float64(3.0) * determinant / (area2 * radius))
