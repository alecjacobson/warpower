"""Conservative whole-tetrahedron exterior proof on original binary32 geometry.

A plane-separation certificate and exact rational-centroid winding exclude the
open tetrahedron interior. Uncertain filtered signs use dyadic integers; an
unresolved ray or failed certificate is inconclusive, never an exterior claim.
Calling kernels must disable fast math and floating-point contraction.
"""

import warp as wp

from . import _filtered_predicates as fp
from . import _integer as integer
from ._surface_bvh import SurfaceTree

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


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


@wp.struct
class BallRow:
    x: fp.Ball
    y: fp.Ball
    z: fp.Ball


@wp.func
def centroid_component(q: float, p0: float, p1: float, p2: float, p3: float):
    return fp.plus(
        fp.plus(fp.difference(q, p0), fp.difference(q, p1)),
        fp.plus(fp.difference(q, p2), fp.difference(q, p3)),
    )


@wp.func
def centroid_row(q: wp.vec3, p0: wp.vec3, p1: wp.vec3, p2: wp.vec3, p3: wp.vec3):
    r = BallRow()
    r.x = centroid_component(q[0], p0[0], p1[0], p2[0], p3[0])
    r.y = centroid_component(q[1], p0[1], p1[1], p2[1], p3[1])
    r.z = centroid_component(q[2], p0[2], p1[2], p2[2], p3[2])
    return r


@wp.func
def direction_row(direction: wp.vec3):
    r = BallRow()
    r.x = fp.scalar(wp.float64(direction[0]))
    r.y = fp.scalar(wp.float64(direction[1]))
    r.z = fp.scalar(wp.float64(direction[2]))
    return r


@wp.func
def determinant(a: BallRow, b: BallRow, c: BallRow):
    return fp.det3(a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z)


@wp.func
def sign(value: fp.Ball):
    return int(value.value > wp.float64(0.0)) - int(value.value < wp.float64(0.0))


@wp.func
def event(n: int, u: int, v: int, w: int, denominator: int):
    if denominator == 0:
        if n == 0:
            return wp.vec2i(0, 1)
        return wp.vec2i(0, 0)
    if n == 0 or n != denominator:
        return wp.vec2i(0, 0)
    if (
        (u == 0 or u == denominator)
        and (v == 0 or v == denominator)
        and (w == 0 or w == denominator)
    ):
        if u == 0 or v == 0 or w == 0:
            return wp.vec2i(0, 1)
        return wp.vec2i(denominator, 0)
    return wp.vec2i(0, 0)


@wp.func(inline=False)
def exact_triangle(
    p0: wp.vec3,
    p1: wp.vec3,
    p2: wp.vec3,
    p3: wp.vec3,
    q0: wp.vec3,
    q1: wp.vec3,
    q2: wp.vec3,
    direction: wp.vec3,
):
    pb = wp.min(
        wp.min(integer.vector_base(p0), integer.vector_base(p1)),
        wp.min(integer.vector_base(p2), integer.vector_base(p3)),
    )
    pb = wp.min(pb, integer.vector_base(direction))
    base = wp.min(
        pb,
        wp.min(integer.vector_base(q0), wp.min(integer.vector_base(q1), integer.vector_base(q2))),
    )
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
    return event(n.sign, u.sign, v.sign, w.sign, denominator.sign)


@wp.func(inline=False)
def triangle(
    p0: wp.vec3,
    p1: wp.vec3,
    p2: wp.vec3,
    p3: wp.vec3,
    q0: wp.vec3,
    q1: wp.vec3,
    q2: wp.vec3,
    direction: wp.vec3,
):
    a = centroid_row(q0, p0, p1, p2, p3)
    b = centroid_row(q1, p0, p1, p2, p3)
    c = centroid_row(q2, p0, p1, p2, p3)
    d = direction_row(direction)
    n = determinant(a, b, c)
    u = determinant(d, b, c)
    v = determinant(d, c, a)
    w = determinant(d, a, b)
    denominator = fp.plus(fp.plus(u, v), w)
    if (
        fp.certain(n)
        and fp.certain(u)
        and fp.certain(v)
        and fp.certain(w)
        and fp.certain(denominator)
    ):
        return event(sign(n), sign(u), sign(v), sign(w), sign(denominator))
    return exact_triangle(p0, p1, p2, p3, q0, q1, q2, direction)


@wp.func(inline=False)
def centroid_ray(
    tree: SurfaceTree, p0: wp.vec3, p1: wp.vec3, p2: wp.vec3, p3: wp.vec3, direction: wp.vec3
):
    count = int(0)
    for f in range(tree.n):
        q0 = wp.vec3(tree.points[tree.indices[3 * f]])
        q1 = wp.vec3(tree.points[tree.indices[3 * f + 1]])
        q2 = wp.vec3(tree.points[tree.indices[3 * f + 2]])
        result = triangle(p0, p1, p2, p3, q0, q1, q2, direction)
        if result[1] != 0:
            return result
        count += result[0]
    return wp.vec2i(count, 0)


@wp.func(inline=False)
def exterior(tree: SurfaceTree, points: wp.array[wp.vec3], ids: wp.vec4i):
    p0 = wp.vec3(points[ids[0]])
    p1 = wp.vec3(points[ids[1]])
    p2 = wp.vec3(points[ids[2]])
    p3 = wp.vec3(points[ids[3]])
    orientation = fp.orient3d(p0, p1, p2, p3)
    if orientation.valid == 0 or orientation.sign <= 0:
        return 0
    lo = wp.min(wp.min(p0, p1), wp.min(p2, p3))
    hi = wp.max(wp.max(p0, p1), wp.max(p2, p3))
    for f in range(tree.n):
        a = wp.vec3(tree.points[tree.indices[3 * f]])
        b = wp.vec3(tree.points[tree.indices[3 * f + 1]])
        c = wp.vec3(tree.points[tree.indices[3 * f + 2]])
        qlo, qhi = wp.min(a, wp.min(b, c)), wp.max(a, wp.max(b, c))
        if (
            qhi[0] < lo[0]
            or qhi[1] < lo[1]
            or qhi[2] < lo[2]
            or qlo[0] > hi[0]
            or qlo[1] > hi[1]
            or qlo[2] > hi[2]
        ):
            continue
        positive = bool(False)
        negative = bool(False)
        for k in range(4):
            side = fp.orient3d(a, b, c, wp.vec3(points[ids[k]]))
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
