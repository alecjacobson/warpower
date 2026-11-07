"""Device triangle contact with exact signs for unshared triangle pairs.

Calling kernels must disable fast math and floating-point contraction. The
filtered predicates enclose roundoff and use exact expansions for uncertainty.
"""

import warp as wp

from . import _filtered_predicates as fp
from ._surface_bvh import SurfaceTree

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.func
def projected(p: wp.vec3, axis: int):
    return wp.vec3(p[(axis + 1) % 3], p[(axis + 2) % 3], 0.0)


@wp.func
def orientation2(a: wp.vec3, b: wp.vec3, c: wp.vec3, axis: int):
    return fp.orient3d(
        projected(a, axis), projected(b, axis), projected(c, axis), wp.vec3(0.0, 0.0, 1.0)
    )


@wp.func
def projection_axis(a: wp.vec3, b: wp.vec3, c: wp.vec3):
    normal = wp.abs(wp.cross(wp.vec3d(b) - wp.vec3d(a), wp.vec3d(c) - wp.vec3d(a)))
    axis = int(0)
    if normal[1] > normal[axis]:
        axis = 1
    if normal[2] > normal[axis]:
        axis = 2
    # SurfaceDomain rejects locally collapsed triangles before this predicate.
    return axis


@wp.func
def between2(a: wp.vec3, b: wp.vec3, p: wp.vec3, axis: int):
    x, y, q = projected(a, axis), projected(b, axis), projected(p, axis)
    return (
        q[0] >= wp.min(x[0], y[0])
        and q[0] <= wp.max(x[0], y[0])
        and q[1] >= wp.min(x[1], y[1])
        and q[1] <= wp.max(x[1], y[1])
    )


@wp.func
def same_side(a: int, b: int, c: int):
    return (a >= 0 and b >= 0 and c >= 0) or (a <= 0 and b <= 0 and c <= 0)


@wp.func
def inside2(p: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3, axis: int):
    u, v, w = orientation2(a, b, p, axis), orientation2(b, c, p, axis), orientation2(c, a, p, axis)
    if u.valid == 0 or v.valid == 0 or w.valid == 0:
        return True  # Fail closed if an exact predicate cannot be evaluated.
    return same_side(u.sign, v.sign, w.sign)


@wp.func
def segments2(p: wp.vec3, q: wp.vec3, a: wp.vec3, b: wp.vec3, axis: int):
    u, v = orientation2(p, q, a, axis), orientation2(p, q, b, axis)
    x, y = orientation2(a, b, p, axis), orientation2(a, b, q, axis)
    if u.valid == 0 or v.valid == 0 or x.valid == 0 or y.valid == 0:
        return True
    if u.sign * v.sign < 0 and x.sign * y.sign < 0:
        return True
    return (
        (u.sign == 0 and between2(p, q, a, axis))
        or (v.sign == 0 and between2(p, q, b, axis))
        or (x.sign == 0 and between2(a, b, p, axis))
        or (y.sign == 0 and between2(a, b, q, axis))
    )


@wp.func
def coplanar_segment(p: wp.vec3, q: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3, axis: int):
    return (
        inside2(p, a, b, c, axis)
        or inside2(q, a, b, c, axis)
        or segments2(p, q, a, b, axis)
        or segments2(p, q, b, c, axis)
        or segments2(p, q, c, a, axis)
    )


@wp.func
def segment_triangle(p: wp.vec3, q: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3, sp: int, sq: int):
    if sp * sq > 0:
        return False
    if sp == 0 and sq == 0:
        return coplanar_segment(p, q, a, b, c, projection_axis(a, b, c))
    # Intersection with the plane is on the segment. Its barycentric signs
    # are the three orientations about the segment's supporting line.
    x, y, z = fp.orient3d(p, q, a, b), fp.orient3d(p, q, b, c), fp.orient3d(p, q, c, a)
    if x.valid == 0 or y.valid == 0 or z.valid == 0:
        return True
    return same_side(x.sign, y.sign, z.sign)


@wp.func
def triangles_intersect(a: wp.vec3, b: wp.vec3, c: wp.vec3, p: wp.vec3, q: wp.vec3, r: wp.vec3):
    sp, sq, sr = fp.orient3d(a, b, c, p), fp.orient3d(a, b, c, q), fp.orient3d(a, b, c, r)
    if sp.valid == 0 or sq.valid == 0 or sr.valid == 0:
        return True
    if (sp.sign > 0 and sq.sign > 0 and sr.sign > 0) or (
        sp.sign < 0 and sq.sign < 0 and sr.sign < 0
    ):
        return False
    sa, sb, sc = fp.orient3d(p, q, r, a), fp.orient3d(p, q, r, b), fp.orient3d(p, q, r, c)
    if sa.valid == 0 or sb.valid == 0 or sc.valid == 0:
        return True
    if (sa.sign > 0 and sb.sign > 0 and sc.sign > 0) or (
        sa.sign < 0 and sb.sign < 0 and sc.sign < 0
    ):
        return False
    # The two filled triangles intersect iff one of their boundary segments
    # meets the other triangle (also covers coplanar containment/contact).
    return (
        segment_triangle(p, q, a, b, c, sp.sign, sq.sign)
        or segment_triangle(q, r, a, b, c, sq.sign, sr.sign)
        or segment_triangle(r, p, a, b, c, sr.sign, sp.sign)
        or segment_triangle(a, b, p, q, r, sa.sign, sb.sign)
        or segment_triangle(b, c, p, q, r, sb.sign, sc.sign)
        or segment_triangle(c, a, p, q, r, sc.sign, sa.sign)
    )


@wp.func
def in_cone(direction: wp.vec3d, a: wp.vec3d, b: wp.vec3d):
    normal = wp.cross(a, b)
    return wp.dot(wp.cross(direction, b), normal) >= wp.float64(0.0) and wp.dot(
        wp.cross(a, direction), normal
    ) >= wp.float64(0.0)


@wp.func
def excess_contact(points: wp.array[wp.vec3], left: wp.vec3i, right: wp.vec3i):
    """Exclude legal indexed contact, including coplanar adjacent triangles."""
    shared = int(0)
    common = int(-1)
    unique_left = int(-1)
    unique_right = int(-1)
    for k in range(3):
        if left[k] == right[0] or left[k] == right[1] or left[k] == right[2]:
            shared += 1
            common = left[k]
        else:
            unique_left = left[k]
        if right[k] != left[0] and right[k] != left[1] and right[k] != left[2]:
            unique_right = right[k]
    if shared == 0:
        return triangles_intersect(
            points[left[0]],
            points[left[1]],
            points[left[2]],
            points[right[0]],
            points[right[1]],
            points[right[2]],
        )
    if shared == 3:
        return True
    origin = wp.vec3d(points[common])
    if shared == 2:
        edge = wp.vec3d(0.0)
        for k in range(3):
            if left[k] != common and left[k] != unique_left:
                edge = wp.vec3d(points[left[k]]) - origin
        a = wp.cross(edge, wp.vec3d(points[unique_left]) - origin)
        b = wp.cross(edge, wp.vec3d(points[unique_right]) - origin)
        parallel = wp.length_sq(wp.cross(a, b)) <= wp.float64(1.0e-24) * wp.length_sq(
            a
        ) * wp.length_sq(b)
        return parallel and wp.dot(a, b) > wp.float64(0.0)
    # Noncoplanar planes intersect along a line through the common vertex.
    # Extra contact exists iff either ray lies in both triangle tangent cones.
    a, b, c, d = wp.vec3d(0.0), wp.vec3d(0.0), wp.vec3d(0.0), wp.vec3d(0.0)
    for k in range(3):
        if left[k] == common:
            a = wp.vec3d(points[left[(k + 1) % 3]]) - origin
            b = wp.vec3d(points[left[(k + 2) % 3]]) - origin
        if right[k] == common:
            c = wp.vec3d(points[right[(k + 1) % 3]]) - origin
            d = wp.vec3d(points[right[(k + 2) % 3]]) - origin
    na, nb = wp.cross(a, b), wp.cross(c, d)
    direction = wp.cross(na, nb)
    parallel = wp.length_sq(direction) <= wp.float64(1.0e-24) * wp.length_sq(na) * wp.length_sq(nb)
    if parallel:
        return in_cone(a, c, d) or in_cone(b, c, d) or in_cone(c, a, b) or in_cone(d, a, b)
    return (in_cone(direction, a, b) and in_cone(direction, c, d)) or (
        in_cone(-direction, a, b) and in_cone(-direction, c, d)
    )


@wp.kernel
def check_intersections(tree: SurfaceTree, error: wp.array[int]):
    f = wp.tid()
    if error[0] != 0:
        return
    a, b, c = tree.indices[3 * f], tree.indices[3 * f + 1], tree.indices[3 * f + 2]
    p, q, r = tree.points[a], tree.points[b], tree.points[c]
    lo, hi = wp.min(p, wp.min(q, r)), wp.max(p, wp.max(q, r))
    node = int(1)
    while node > 0:
        lower, upper = tree.lower[node], tree.upper[node]
        if (
            lower[0] > hi[0]
            or lower[1] > hi[1]
            or lower[2] > hi[2]
            or upper[0] < lo[0]
            or upper[1] < lo[1]
            or upper[2] < lo[2]
        ):
            node = tree.escape[node]
            continue
        if node >= tree.leaf_base:
            first = 4 * (node - tree.leaf_base)
            for j in range(first, wp.min(first + 4, tree.n)):
                other = tree.ids[j]
                if other > f:
                    right = wp.vec3i(
                        tree.indices[3 * other],
                        tree.indices[3 * other + 1],
                        tree.indices[3 * other + 2],
                    )
                    x, y, z = tree.points[right[0]], tree.points[right[1]], tree.points[right[2]]
                    face_lower, face_upper = wp.min(x, wp.min(y, z)), wp.max(x, wp.max(y, z))
                    if (
                        face_lower[0] > hi[0]
                        or face_lower[1] > hi[1]
                        or face_lower[2] > hi[2]
                        or face_upper[0] < lo[0]
                        or face_upper[1] < lo[1]
                        or face_upper[2] < lo[2]
                    ):
                        continue
                    if excess_contact(tree.points, wp.vec3i(a, b, c), right):
                        wp.atomic_or(error, 0, 16)
            node = tree.escape[node]
        else:
            node = tree.right[node]
