import warp as wp
from warpower._surface_bvh import SurfaceTree, Closest, closest_point, closest_triangle, material_side
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

@wp.func
def hash_center(center:wp.vec3d,mask:int):
    h=wp.cast(center[0],wp.uint64)*wp.uint64(0x9E3779B97F4A7C15)
    h=h^(wp.cast(center[1],wp.uint64)*wp.uint64(0xBF58476D1CE4E5B9))
    h=h^(wp.cast(center[2],wp.uint64)*wp.uint64(0x94D049BB133111EB))
    h=(h^(h>>wp.uint64(30)))*wp.uint64(0xBF58476D1CE4E5B9)
    h=(h^(h>>wp.uint64(27)))*wp.uint64(0x94D049BB133111EB)
    h=h^(h>>wp.uint64(31))
    return int(h&wp.uint64(mask))

@wp.func
def same_center(a:wp.vec3d,b:wp.vec3d):
    return wp.cast(a[0],wp.uint64)==wp.cast(b[0],wp.uint64) and wp.cast(a[1],wp.uint64)==wp.cast(b[1],wp.uint64) and wp.cast(a[2],wp.uint64)==wp.cast(b[2],wp.uint64)

@wp.func
def lookup(center:wp.vec3d,previous:QueryState,table:wp.array[int]):
    slot=hash_center(center,table.shape[0]-1)
    for _ in range(128):
        j=table[slot]
        if j<0:return -1
        if same_center(center,previous.centers[j]):return j
        slot=(slot+1)&(table.shape[0]-1)
    return -1

@wp.func(inline=True)
def cached_side(tree:SurfaceTree,center:wp.vec3d,previous:QueryState,table:wp.array[int]):
    j=lookup(center,previous,table)
    if j>=0 and previous.rays[j]!=0:return previous.sides[j]
    return material_side(tree,center)

@wp.func(inline=True)
def cached_nearest(tree:SurfaceTree,center:wp.vec3d,limit:wp.float64,previous:QueryState,table:wp.array[int]):
    j=lookup(center,previous,table)
    if j>=0 and previous.nearest[j]!=0 and wp.cast(limit,wp.uint64)==wp.cast(previous.limits[j],wp.uint64):
        result=Closest()
        result.point=previous.hit_points[j]
        result.face=previous.faces[j]
        result.distance2=previous.distances[j]
        return result
    return closest_point(tree,center,limit)

@wp.kernel
def build_cache(previous:QueryState,table:wp.array[int],error:wp.array[int]):
    i=wp.tid()
    if error[0]!=0 or previous.valid[i]==0:return
    center=previous.centers[i];mask=table.shape[0]-1;slot=hash_center(center,mask)
    for _ in range(128):
        old=wp.atomic_cas(table,slot,-1,i)
        if old<0:return
        if same_center(center,previous.centers[old]):
            wp.atomic_min(table,slot,i)
            return
        slot=(slot+1)&mask
    # Uninserted keys remain misses, so bounded probing only reduces reuse.

@wp.kernel
def classify_cached(
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
    previous: QueryState,
    table: wp.array[int],
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
    hit = cached_nearest(tree, center, wp.float64(1.0e150), previous, table)
    probe.distances[i] = hit.distance2
    probe.faces[i] = hit.face
    probe.hit_points[i] = hit.point
    if hit.face < 0 or not wp.isfinite(hit.distance2):
        wp.atomic_or(error, 0, 16)
        return
    side = int(1)
    if hit.distance2 > wp.float64(0.0):
        probe.rays[i] = 1
        side = cached_side(tree, center, previous, table)
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
def retained_cached(
    tree: SurfaceTree,
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    count: wp.array[int],
    tolerance: wp.float64,
    error: wp.array[int],
    keep: wp.array[int],
    quality: wp.array[wp.float64],
    probe: QueryState,
    previous: QueryState,
    table: wp.array[int],
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
    side = cached_side(tree, center, previous, table)
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
        hit = cached_nearest(tree, center, limit, previous, table)
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
def selection_center_cached(
    tree: SurfaceTree, center: wp.vec3d, radius: wp.float64, tolerance: wp.float64, probe: QueryState, slot: int, previous: QueryState, table: wp.array[int]
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
        side = cached_side(tree, center, previous, table)
        probe.sides[slot] = side
    if proved and side == 1:
        # Private sign-only contract. Full distances remain unavailable here.
        result.distance_sign = wp.float64(-1.0)
        result.retained = 1
    else:
        probe.nearest[slot] = 1
        probe.limits[slot] = wp.float64(1.0e150)
        hit = cached_nearest(tree, center, wp.float64(1.0e150), previous, table)
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
            side = cached_side(tree, center, previous, table)
            probe.sides[slot] = side
        if side != 0 and side != 1:
            result.error = 16
            return result
        distance = wp.sqrt(hit.distance2)
        result.distance_sign = wp.where(side == 1, -distance, distance)
        result.retained = int(side == 1 or distance / radius < tolerance)
    return result

@wp.kernel
def selection_cached(
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
    previous: QueryState,
    table: wp.array[int],
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
    status = selection_center_cached(tree, center, radius, tolerance, probe, i, previous, table)
    if status.error != 0:
        wp.atomic_or(error, 0, status.error)
        return
    distances[i] = status.distance_sign
    keep[i] = status.retained
    centers[i] = center
    radii[i] = radius
    volumes[i] = determinant / wp.float64(6.0)
    quality[i] = wp.min(wp.float64(1.0), wp.float64(3.0) * determinant / (area2 * radius))
