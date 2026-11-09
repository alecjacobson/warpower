"""Optional native float32 surface queries for reasonably scaled, good meshes.

This backend uses Warp's triangle acceptance and oriented-ray sign semantics.
It does not verify successful queries through the reference surface traversal.
Physical tetrahedron geometry and accumulation outputs retain float64 here.
"""

import warp as wp

from ._wide_center import exact_geometry, uncertain

wp.set_module_options({"enable_backward": False, "fast_math": False})


@wp.struct
class CellGeometry:
    center: wp.vec3d
    radius: wp.float64
    volume: wp.float64
    quality: wp.float64
    error: int


@wp.func
def cell_geometry(points: wp.array[wp.vec3], tets: wp.array2d[int], i: int):
    result = CellGeometry()
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
    result.volume = determinant / wp.float64(6.0)
    result.quality = wp.min(wp.float64(1.0), wp.float64(3.0) * determinant / (area2 * radius))
    return result


@wp.struct
class SurfaceHit:
    point: wp.vec3
    distance: float
    inside: int
    face: int
    valid: int


@wp.func
def signed_hit(mesh: wp.uint64, point: wp.vec3, limit: float):
    hit = SurfaceHit()
    hit.face = -1
    if wp.isfinite(point[0]) and wp.isfinite(point[1]) and wp.isfinite(point[2]):
        # Normal accumulation can cancel at medial axes (e.g. a cavity center).
        # The native three-ray vote handles those cases without a reference query.
        query = wp.mesh_query_point(mesh, point, limit)
        if query.result:
            closest = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
            distance = wp.length(point - closest)
            if wp.isfinite(distance):
                hit.point = closest
                hit.distance = distance
                hit.inside = int(query.sign < 0.0 or distance == 0.0)
                hit.face = query.face
                hit.valid = 1
    return hit


@wp.kernel
def classify(
    mesh: wp.uint64,
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
):
    i = wp.tid()
    keep[i] = 0
    if error[0] != 0 or i >= count[0]:
        return
    geometry = cell_geometry(points, tets, i)
    if geometry.error != 0:
        wp.atomic_or(error, 0, geometry.error)
        return
    hit = signed_hit(mesh, wp.vec3(geometry.center), 1.0e30)
    if hit.valid == 0:
        wp.atomic_or(error, 0, 16)
        return
    distance = wp.float64(hit.distance)
    centers[i] = geometry.center
    radii[i] = geometry.radius
    distances[i] = wp.where(hit.inside != 0, -distance, distance)
    keep[i] = int(hit.inside != 0 or distance / geometry.radius < tolerance)
    volumes[i] = geometry.volume
    quality[i] = geometry.quality


@wp.kernel
def classify_retained(
    mesh: wp.uint64,
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    count: wp.array[int],
    tolerance: wp.float64,
    error: wp.array[int],
    keep: wp.array[int],
    quality: wp.array[wp.float64],
):
    i = wp.tid()
    keep[i] = 0
    if error[0] != 0 or i >= count[0]:
        return
    geometry = cell_geometry(points, tets, i)
    if geometry.error != 0:
        wp.atomic_or(error, 0, geometry.error)
        return
    hit = signed_hit(mesh, wp.vec3(geometry.center), 1.0e30)
    if hit.valid == 0:
        wp.atomic_or(error, 0, 16)
        return
    keep[i] = int(hit.inside != 0 or wp.float64(hit.distance) / geometry.radius < tolerance)
    quality[i] = geometry.quality


@wp.kernel
def surface_error(
    mesh: wp.uint64,
    points: wp.array[wp.vec3],
    faces: wp.array2d[int],
    count: wp.array[int],
    integral: wp.array[wp.float64],
    maximum: wp.array[wp.float64],
    error: wp.array[int],
):
    i, k = wp.tid() // 3, wp.tid() % 3
    if i >= count[0] or error[0] != 0:
        return
    a, b, c = (
        wp.vec3d(points[faces[i, 0]]),
        wp.vec3d(points[faces[i, 1]]),
        wp.vec3d(points[faces[i, 2]]),
    )
    weight = wp.length(wp.cross(b - a, c - a)) / wp.float64(6.0)
    dominant = a
    if k == 1:
        dominant = b
    elif k == 2:
        dominant = c
    sample = wp.vec3((a + b + c) / wp.float64(6.0) + dominant / wp.float64(2.0))
    hit = wp.mesh_query_point_no_sign(mesh, sample, 1.0e30)
    if not hit.result:
        wp.atomic_or(error, 0, 16)
    else:
        delta = sample - wp.mesh_eval_position(mesh, hit.face, hit.u, hit.v)
        distance2 = wp.float64(wp.length_sq(delta))
        if not wp.isfinite(distance2):
            wp.atomic_or(error, 0, 16)
        else:
            wp.atomic_add(integral, 0, weight * distance2)
            wp.atomic_max(maximum, 0, wp.sqrt(distance2))


@wp.kernel
def selection_surface_status(
    mesh: wp.uint64,
    points: wp.array[wp.vec3],
    faces: wp.array2d[int],
    count: wp.array[int],
    error: wp.array[int],
):
    i = wp.tid()
    if i >= count[0] or error[0] != 0:
        return
    a, b, c = (
        wp.vec3d(points[faces[i, 0]]),
        wp.vec3d(points[faces[i, 1]]),
        wp.vec3d(points[faces[i, 2]]),
    )
    for k in range(3):
        dominant = a
        if k == 1:
            dominant = b
        elif k == 2:
            dominant = c
        sample = wp.vec3((a + b + c) / wp.float64(6.0) + dominant / wp.float64(2.0))
        hit = wp.mesh_query_point_no_sign(mesh, sample, 1.0e30)
        if not hit.result:
            wp.atomic_or(error, 0, 16)


@wp.kernel
def domain_queries(
    mesh: wp.uint64,
    points: wp.array[wp.vec3],
    domain_error: wp.array[int],
    max_distance: float,
    closest: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    distance: wp.array[float],
    inside: wp.array[int],
    faces: wp.array[int],
    valid: wp.array[int],
    error: wp.array[int],
):
    i = wp.tid()
    closest[i] = wp.vec3(0.0)
    normal[i] = wp.vec3(0.0)
    distance[i] = max_distance
    inside[i] = 0
    faces[i] = -1
    valid[i] = 0
    if domain_error[0] != 0:
        wp.atomic_or(error, 0, domain_error[0])
        return
    p = points[i]
    if not wp.isfinite(p[0]) or not wp.isfinite(p[1]) or not wp.isfinite(p[2]):
        wp.atomic_or(error, 0, 64)
        return
    hit = signed_hit(mesh, p, max_distance)
    if hit.valid != 0:
        closest[i] = hit.point
        normal[i] = wp.mesh_eval_face_normal(mesh, hit.face)
        distance[i] = wp.where(hit.inside != 0, -hit.distance, hit.distance)
        inside[i] = hit.inside
        faces[i] = hit.face
        valid[i] = 1
