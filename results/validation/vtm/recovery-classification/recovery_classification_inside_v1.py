"""Private #194 inside-only witness probe, based on merged f338eb5.

Only private selective recovery uses this kernel. Signed distance is a sign
marker for shortcut interiors, explicitly unavailable as a physical distance.
Outside/ambiguous/unsupported centers use the complete original distance query.
No shortened outside search is used. No public workspace changes.
"""
import warp as wp
from warpower import _extraction
from warpower._surface_bvh import SurfaceTree, closest_point, closest_triangle, material_side
from warpower._wide_center import exact_geometry, uncertain
wp.set_module_options({'enable_backward': False, 'fast_math': False})

@wp.kernel
def classify_selection_inside(
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
):
    i = wp.tid()
    keep[i] = 0
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
    proved = False
    if tree.n > 0 and (wp.abs(center[0]) < wp.float64(1.0e140) and wp.abs(center[1]) < wp.float64(1.0e140) and wp.abs(center[2]) < wp.float64(1.0e140)):
        f = tree.ids[0]
        x = wp.vec3d(tree.points[tree.indices[3 * f]])
        y = wp.vec3d(tree.points[tree.indices[3 * f + 1]])
        z = wp.vec3d(tree.points[tree.indices[3 * f + 2]])
        q = closest_triangle(center, x, y, z)
        witness_distance = wp.length_sq(center - q)
        limit = wp.float64(1.0e150)
        proved = wp.isfinite(witness_distance) and witness_distance < limit * limit
    side = int(-1)
    if proved:
        side = material_side(tree, center)
    if proved and side == 1:
        # Private sign-only contract. Full distances remain unavailable here.
        distances[i] = wp.float64(-1.0)
        keep[i] = 1
    else:
        hit = closest_point(tree, center, wp.float64(1.0e150))
        if hit.face < 0 or not wp.isfinite(hit.distance2):
            wp.atomic_or(error, 0, 16)
            return
        if hit.distance2 == wp.float64(0.0):
            side = 1
        elif not proved:
            side = material_side(tree, center)
        if side != 0 and side != 1:
            wp.atomic_or(error, 0, 16)
            return
        distance = wp.sqrt(hit.distance2)
        distances[i] = wp.where(side == 1, -distance, distance)
        keep[i] = int(side == 1 or distance / radius < tolerance)
    centers[i] = center
    radii[i] = radius
    volumes[i] = determinant / wp.float64(6.0)
    quality[i] = wp.min(wp.float64(1.0), wp.float64(3.0) * determinant / (area2 * radius))


class PrivateInsideProbe(_extraction._SelectionExtractionWorkspace):
    def _launch(self, kernel, dim, args, *, block_dim=256):
        if kernel is _extraction.classify:
            kernel = classify_selection_inside
        return super()._launch(kernel, dim, args, block_dim=block_dim)


def install():
    wp.load_module(module=__name__, device='cuda:0')
    _extraction._SelectionExtractionWorkspace = PrivateInsideProbe
