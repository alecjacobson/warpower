import warp as wp
from warpower._surface_bvh import SurfaceTree, closest_point, closest_triangle, material_side
from warpower._extraction import SelectionCenter
from warpower._wide_center import exact_geometry, uncertain
wp.set_module_options({'enable_backward':False,'fast_math':False})

@wp.struct
class QueryState:
    centers:wp.array[wp.vec3d]
    valid:wp.array[int]
    rays:wp.array[int]
    sides:wp.array[int]
    nearest:wp.array[int]
    limits:wp.array[wp.float64]
    distances:wp.array[wp.float64]
    faces:wp.array[int]
    hit_points:wp.array[wp.vec3d]

@wp.kernel
def observe_full(
    tree: SurfaceTree,
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    count: wp.array[int],
    tolerance: wp.float64,
    error: wp.array[int],
    centers: wp.array[wp.vec3d],
    radii: wp.array[wp.float64],
    distances: wp.array[wp.float64],
    keep: wp.array[int],
    volumes: wp.array[wp.float64],
    quality: wp.array[wp.float64],
    probe: QueryState,
):
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
        if not wp.isfinite(p[0]) or not wp.isfinite(p[1]) or not wp.isfinite(p[2]):
            wp.atomic_or(error, 0, 4)
            return
    p0 = wp.vec3d(points[ids[0]])
    a, b, c = (
        wp.vec3d(points[ids[1]]) - p0,
        wp.vec3d(points[ids[2]]) - p0,
        wp.vec3d(points[ids[3]]) - p0,
    )
    bc, ca, ab = wp.cross(b, c), wp.cross(c, a), wp.cross(a, b)
    determinant = wp.dot(a, bc)
    area2 = wp.length(bc) + wp.length(ca) + wp.length(ab) + wp.length(wp.cross(b - a, c - a))
    center, radius = wp.vec3d(), wp.float64(0.0)
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
        offset = (wp.dot(a, a) * bc + wp.dot(b, b) * ca + wp.dot(c, c) * ab) / (
            wp.float64(2.0) * determinant
        )
        center, radius = p0 + offset, wp.length(offset)
    if not wp.isfinite(radius) or radius <= wp.float64(0.0):
        wp.atomic_or(error, 0, 8)
        return
    probe.centers[i] = center
    probe.valid[i] = 1
    probe.nearest[i] = 1
    probe.limits[i] = wp.float64(1.0e150)
    hit = closest_point(tree, center, wp.float64(1.0e150))
    probe.distances[i] = hit.distance2
    probe.faces[i] = hit.face
    probe.hit_points[i] = hit.point
    if hit.face < 0 or not wp.isfinite(hit.distance2):
        wp.atomic_or(error, 0, 16)
        return
    side = int(1)
    if hit.distance2 > wp.float64(0.0):
        probe.rays[i] = 1
        side = material_side(tree, center)
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
def observe_retained(
    tree: SurfaceTree,
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    count: wp.array[int],
    tolerance: wp.float64,
    error: wp.array[int],
    keep: wp.array[int],
    quality: wp.array[wp.float64],
    probe: QueryState,
):
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
        if not wp.isfinite(p[0]) or not wp.isfinite(p[1]) or not wp.isfinite(p[2]):
            wp.atomic_or(error, 0, 4)
            return
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
        offset = (wp.dot(a, a) * bc + wp.dot(b, b) * ca + wp.dot(c, c) * ab) / (
            wp.float64(2.0) * determinant
        )
        center, radius = p0 + offset, wp.length(offset)
    if not wp.isfinite(radius) or radius <= wp.float64(0.0):
        wp.atomic_or(error, 0, 8)
        return
    probe.centers[i] = center
    probe.valid[i] = 1
    probe.rays[i] = 1
    side = material_side(tree, center)
    probe.sides[i] = side
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
        probe.nearest[i] = 1
        probe.limits[i] = limit
        hit = closest_point(tree, center, limit)
        probe.distances[i] = hit.distance2
        probe.faces[i] = hit.face
        probe.hit_points[i] = hit.point
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
    quality[i] = wp.min(wp.float64(1.0), wp.float64(3.0) * determinant / (area2 * radius))

@wp.func(inline=True)
def observe_selection_center(
    tree: SurfaceTree, center: wp.vec3d, radius: wp.float64, tolerance: wp.float64, probe: QueryState, slot: int
):
    """Private retention and distance sign; magnitude is not a distance output.

    Radius/geometry are validated by the caller. Unproved or non-inside centers
    keep the complete query, including exact-zero and strict threshold handling.
    """
    result = SelectionCenter()
    proved = False
    if (
        tree.n > 0
        and wp.abs(center[0]) < wp.float64(1.0e140)
        and wp.abs(center[1]) < wp.float64(1.0e140)
        and wp.abs(center[2]) < wp.float64(1.0e140)
    ):
        f = tree.ids[0]
        x = wp.vec3d(tree.points[tree.indices[3 * f]])
        y = wp.vec3d(tree.points[tree.indices[3 * f + 1]])
        z = wp.vec3d(tree.points[tree.indices[3 * f + 2]])
        q = closest_triangle(center, x, y, z)
        witness_distance = wp.length_sq(center - q)
        limit = wp.float64(1.0e150)
        # Current finite float32 boxes have squared distances < 4e280
        # from a center within these bounds. Before a finite hit, the full
        # traversal cannot prune this witnessed leaf against its ~1e300
        # initial limit. A different earlier hit also proves query success.
        proved = wp.isfinite(witness_distance) and witness_distance < limit * limit
    side = int(-1)
    if proved:
        probe.rays[slot] = 1
        side = material_side(tree, center)
        probe.sides[slot] = side
    if proved and side == 1:
        # Private sign-only contract. Full distances remain unavailable here.
        result.distance_sign = wp.float64(-1.0)
        result.retained = 1
    else:
        probe.nearest[slot] = 1
        probe.limits[slot] = wp.float64(1.0e150)
        hit = closest_point(tree, center, wp.float64(1.0e150))
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
            side = material_side(tree, center)
            probe.sides[slot] = side
        if side != 0 and side != 1:
            result.error = 16
            return result
        distance = wp.sqrt(hit.distance2)
        result.distance_sign = wp.where(side == 1, -distance, distance)
        result.retained = int(side == 1 or distance / radius < tolerance)
    return result

@wp.kernel
def observe_selection(
    tree: SurfaceTree,
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    count: wp.array[int],
    tolerance: wp.float64,
    error: wp.array[int],
    centers: wp.array[wp.vec3d],
    radii: wp.array[wp.float64],
    distances: wp.array[wp.float64],
    keep: wp.array[int],
    volumes: wp.array[wp.float64],
    quality: wp.array[wp.float64],
    probe: QueryState,
):
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
        if not wp.isfinite(p[0]) or not wp.isfinite(p[1]) or not wp.isfinite(p[2]):
            wp.atomic_or(error, 0, 4)
            return
    # Keep physical geometry arithmetic identical to classify. This private
    # kernel changes only the surface-query decision, not public geometry.
    p0 = wp.vec3d(points[ids[0]])
    a, b, c = (
        wp.vec3d(points[ids[1]]) - p0,
        wp.vec3d(points[ids[2]]) - p0,
        wp.vec3d(points[ids[3]]) - p0,
    )
    bc, ca, ab = wp.cross(b, c), wp.cross(c, a), wp.cross(a, b)
    determinant = wp.dot(a, bc)
    area2 = wp.length(bc) + wp.length(ca) + wp.length(ab) + wp.length(wp.cross(b - a, c - a))
    center, radius = wp.vec3d(), wp.float64(0.0)
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
        offset = (wp.dot(a, a) * bc + wp.dot(b, b) * ca + wp.dot(c, c) * ab) / (
            wp.float64(2.0) * determinant
        )
        center, radius = p0 + offset, wp.length(offset)
    if not wp.isfinite(radius) or radius <= wp.float64(0.0):
        wp.atomic_or(error, 0, 8)
        return
    probe.centers[i] = center
    probe.valid[i] = 1
    status = observe_selection_center(tree, center, radius, tolerance, probe, i)
    if status.error != 0:
        wp.atomic_or(error, 0, status.error)
        return
    distances[i] = status.distance_sign
    keep[i] = status.retained
    centers[i] = center
    radii[i] = radius
    volumes[i] = determinant / wp.float64(6.0)
    quality[i] = wp.min(wp.float64(1.0), wp.float64(3.0) * determinant / (area2 * radius))
