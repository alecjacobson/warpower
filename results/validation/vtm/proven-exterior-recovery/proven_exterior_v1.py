"""Private exact proof of an entire tetrahedron's open interior being exterior."""
import warp as wp

from warpower import _integer as integer
from warpower import _predicates as predicates
from warpower._surface_bvh import SurfaceTree

wp.set_module_options({'enable_backward': False, 'fast_math': False, 'fuse_fp': False})


@wp.func
def add_rows(a: integer.Row, b: integer.Row):
    r = integer.Row()
    r.x = integer.plus(a.x, b.x)
    r.y = integer.plus(a.y, b.y)
    r.z = integer.plus(a.z, b.z)
    r.w = integer.empty()
    return r


@wp.func(inline=False)
def centroid_relative(q: wp.vec3, p0: wp.vec3, p1: wp.vec3, p2: wp.vec3, p3: wp.vec3, base: int):
    # Exact 4*q - sum(p_i), scaled by 2^-base. No rounded centroid exists.
    a = add_rows(integer.relative(q, p0, base), integer.relative(q, p1, base))
    b = add_rows(integer.relative(q, p2, base), integer.relative(q, p3, base))
    return add_rows(a, b)


@wp.func(inline=False)
def centroid_ray(tree: SurfaceTree, p0: wp.vec3, p1: wp.vec3, p2: wp.vec3, p3: wp.vec3, direction: wp.vec3):
    count = int(0)
    pb = wp.min(wp.min(integer.vector_base(p0), integer.vector_base(p1)),
                wp.min(integer.vector_base(p2), integer.vector_base(p3)))
    pb = wp.min(pb, integer.vector_base(direction))
    for f in range(tree.n):
        q0 = wp.vec3(tree.points[tree.indices[3*f]])
        q1 = wp.vec3(tree.points[tree.indices[3*f+1]])
        q2 = wp.vec3(tree.points[tree.indices[3*f+2]])
        base = wp.min(pb, wp.min(integer.vector_base(q0), wp.min(integer.vector_base(q1), integer.vector_base(q2))))
        a = centroid_relative(q0, p0, p1, p2, p3, base)
        b = centroid_relative(q1, p0, p1, p2, p3, base)
        c = centroid_relative(q2, p0, p1, p2, p3, base)
        d = integer.relative(direction, wp.vec3(), base)
        n = integer.det3(a, b, c)
        u = integer.det3(d, b, c)
        v = integer.det3(d, c, a)
        w = integer.det3(d, a, b)
        denominator = integer.plus(integer.plus(u, v), w)
        if n.valid == 0 or u.valid == 0 or v.valid == 0 or w.valid == 0 or denominator.valid == 0:
            return wp.vec2i(0, 2)
        if denominator.sign == 0:
            if n.sign == 0:
                return wp.vec2i(0, 1)
            continue
        if n.sign == 0 or n.sign != denominator.sign:
            continue
        if ((u.sign == 0 or u.sign == denominator.sign)
                and (v.sign == 0 or v.sign == denominator.sign)
                and (w.sign == 0 or w.sign == denominator.sign)):
            if u.sign == 0 or v.sign == 0 or w.sign == 0:
                return wp.vec2i(0, 1)
            count += denominator.sign
    return wp.vec2i(count, 0)


@wp.func(inline=False)
def exterior(tree: SurfaceTree, points: wp.array[wp.vec3], ids: wp.vec4i):
    p0 = wp.vec3(points[ids[0]])
    p1 = wp.vec3(points[ids[1]])
    p2 = wp.vec3(points[ids[2]])
    p3 = wp.vec3(points[ids[3]])
    orientation = predicates.orient3d(p0, p1, p2, p3)
    if orientation.valid == 0 or orientation.sign <= 0:
        return 0
    lo = wp.min(wp.min(p0, p1), wp.min(p2, p3))
    hi = wp.max(wp.max(p0, p1), wp.max(p2, p3))
    for f in range(tree.n):
        a = wp.vec3(tree.points[tree.indices[3*f]])
        b = wp.vec3(tree.points[tree.indices[3*f+1]])
        c = wp.vec3(tree.points[tree.indices[3*f+2]])
        qlo, qhi = wp.min(a, wp.min(b, c)), wp.max(a, wp.max(b, c))
        if qhi[0] < lo[0] or qhi[1] < lo[1] or qhi[2] < lo[2] or qlo[0] > hi[0] or qlo[1] > hi[1] or qlo[2] > hi[2]:
            continue
        positive = bool(False)
        negative = bool(False)
        for k in range(4):
            side = predicates.orient3d(a, b, c, wp.vec3(points[ids[k]]))
            if side.valid == 0:
                return 0
            positive = positive or side.sign > 0
            negative = negative or side.sign < 0
        if positive and negative:
            return 0
    # The positively oriented tet has nonzero volume. Separation from every
    # overlapping triangle plane makes its exact centroid a strict witness.
    r = centroid_ray(tree, p0, p1, p2, p3, wp.vec3(11.0, 23.0, 37.0))
    if r[1] == 1:
        r = centroid_ray(tree, p0, p1, p2, p3, wp.vec3(-17.0, 29.0, 43.0))
    if r[1] == 1:
        r = centroid_ray(tree, p0, p1, p2, p3, wp.vec3(31.0, -47.0, 19.0))
    return int(r[1] == 0 and r[0] == 0)


@wp.kernel
def prove(tree: SurfaceTree, points: wp.array[wp.vec3], tets: wp.array2d[int], result: wp.array[int]):
    i = wp.tid()
    result[i] = exterior(tree, points, wp.vec4i(tets[i, 0], tets[i, 1], tets[i, 2], tets[i, 3]))
