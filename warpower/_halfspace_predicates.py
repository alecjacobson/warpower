"""Original-float32 site/box halfspace signs, including symbolic lifting ties."""

import warp as wp

from . import _filtered_predicates as fp
from . import _integer as integer
from ._delaunay_certificate import direction

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})
Ids5 = wp.types.vector(5, int)


@wp.struct
class Plane:
    x: fp.Ball
    y: fp.Ball
    z: fp.Ball
    w: fp.Ball


@wp.func
def weight(index: int, mode: int, selected: int):
    value = float(0.0)
    if mode == 1:
        value = direction(index)
    elif mode == 2 and index == selected:
        value = 1.0
    return value


@wp.func
def row(
    points: wp.array[wp.vec3],
    bounds: wp.array[wp.vec3],
    owner: int,
    index: int,
    mode: int,
    selected: int,
):
    result = Plane()
    result.x = fp.scalar(wp.float64(0.0))
    result.y = fp.scalar(wp.float64(0.0))
    result.z = fp.scalar(wp.float64(0.0))
    result.w = fp.scalar(wp.float64(0.0))
    if index >= 0:
        p, q = points[owner], points[index]
        factor = fp.scalar(wp.float64(-2.0))
        result.x = fp.times(factor, fp.difference(q[0], p[0]))
        result.y = fp.times(factor, fp.difference(q[1], p[1]))
        result.z = fp.times(factor, fp.difference(q[2], p[2]))
        lq = fp.lift(
            fp.scalar(wp.float64(q[0])),
            fp.scalar(wp.float64(q[1])),
            fp.scalar(wp.float64(q[2])),
            weight(index, mode, selected),
            0.0,
        )
        lp = fp.lift(
            fp.scalar(wp.float64(p[0])),
            fp.scalar(wp.float64(p[1])),
            fp.scalar(wp.float64(p[2])),
            weight(owner, mode, selected),
            0.0,
        )
        result.w = fp.minus(lq, lp)
    elif index == -7:
        result.w = fp.scalar(wp.float64(1.0))
    else:
        plane = -index - 1
        axis, side = plane // 2, plane % 2
        sign = wp.float64(1 - 2 * side)
        if axis == 0:
            result.x = fp.scalar(sign)
        elif axis == 1:
            result.y = fp.scalar(sign)
        else:
            result.z = fp.scalar(sign)
        result.w = fp.scalar(-sign * wp.float64(bounds[side][axis]))
    return result


@wp.func
def normal_det(a: Plane, b: Plane, c: Plane):
    return fp.det3(a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z)


@wp.func
def filtered_det(a: Plane, b: Plane, c: Plane, d: Plane):
    return fp.minus(
        fp.plus(fp.times(b.w, normal_det(a, c, d)), fp.times(d.w, normal_det(a, b, c))),
        fp.plus(fp.times(a.w, normal_det(b, c, d)), fp.times(c.w, normal_det(a, b, d))),
    )


@wp.func
def halfspace_row(
    points: wp.array[wp.vec3],
    bounds: wp.array[wp.vec3],
    owner: int,
    index: int,
    mode: int,
    selected: int,
    base: int,
):
    r = integer.Row()
    r.x = integer.empty()
    r.y = integer.empty()
    r.z = integer.empty()
    r.w = integer.empty()
    if index >= 0:
        p, q = points[owner], points[index]
        r = integer.relative(p, q, base)
        r.x = integer.plus(r.x, r.x)
        r.y = integer.plus(r.y, r.y)
        r.z = integer.plus(r.z, r.z)
        rp = integer.relative(p, wp.vec3(0.0), base)
        rq = integer.relative(q, wp.vec3(0.0), base)
        r.w = integer.plus(
            integer.minus(integer.norm(rq), integer.norm(rp)),
            integer.minus(
                integer.from_float(weight(owner, mode, selected), 2 * base),
                integer.from_float(weight(index, mode, selected), 2 * base),
            ),
        )
    elif index == -7:
        r.w = integer.from_float(1.0, 2 * base)
    else:
        plane = -index - 1
        axis, side = plane // 2, plane % 2
        s = 1.0 - 2.0 * float(side)
        if axis == 0:
            r.x = integer.from_float(s, base)
        elif axis == 1:
            r.y = integer.from_float(s, base)
        else:
            r.z = integer.from_float(s, base)
        r.w = integer.from_float(-s * bounds[side][axis], 2 * base)
    return r


@wp.func(inline=False)
def exact_det(
    points: wp.array[wp.vec3],
    bounds: wp.array[wp.vec3],
    owner: int,
    ids: wp.vec4i,
    mode: int,
    selected: int,
):
    base = wp.min(
        integer.vector_base(points[owner]), integer.base_for(weight(owner, mode, selected), 2)
    )
    for k in range(4):
        index = ids[k]
        if index >= 0:
            base = wp.min(
                base,
                wp.min(
                    integer.vector_base(points[index]),
                    integer.base_for(weight(index, mode, selected), 2),
                ),
            )
        elif index != -7:
            plane = -index - 1
            base = wp.min(base, integer.base_for(bounds[plane % 2][plane // 2], 2))
    value = integer.det4(
        halfspace_row(points, bounds, owner, ids[0], mode, selected, base),
        halfspace_row(points, bounds, owner, ids[1], mode, selected, base),
        halfspace_row(points, bounds, owner, ids[2], mode, selected, base),
        halfspace_row(points, bounds, owner, ids[3], mode, selected, base),
    )
    return integer.predicate(value, 5 * base)


@wp.func
def determinant(
    points: wp.array[wp.vec3],
    bounds: wp.array[wp.vec3],
    owner: int,
    ids: wp.vec4i,
    mode: int,
    selected: int,
):
    a = row(points, bounds, owner, ids[0], mode, selected)
    b = row(points, bounds, owner, ids[1], mode, selected)
    c = row(points, bounds, owner, ids[2], mode, selected)
    d = row(points, bounds, owner, ids[3], mode, selected)
    value = filtered_det(a, b, c, d)
    if fp.certain(value):
        return fp.predicate(value)
    return exact_det(points, bounds, owner, ids, mode, selected)


@wp.func(inline=False)
def classify(points: wp.array[wp.vec3], bounds: wp.array[wp.vec3], owner: int, ids: wp.vec4i):
    normal = determinant(points, bounds, owner, wp.vec4i(ids[0], ids[1], ids[2], -7), 0, -1)
    result = normal
    if normal.valid == 0 or normal.sign == 0:
        result.valid = 0
        return result
    side = determinant(points, bounds, owner, ids, 0, -1)
    if side.valid != 0 and side.sign == 0:
        side = determinant(points, bounds, owner, ids, 1, -1)
    if side.valid != 0 and side.sign == 0:
        last = int(-1)
        rows = Ids5(ids[0], ids[1], ids[2], ids[3], owner)
        for _ in range(5):
            best = int(2147483647)
            for k in range(5):
                if rows[k] > last and rows[k] < best:
                    best = rows[k]
            if best == 2147483647:
                break
            side = determinant(points, bounds, owner, ids, 2, best)
            if side.valid == 0 or side.sign != 0:
                break
            last = best
    result.valid = int(side.valid != 0 and side.sign != 0)
    result.sign = normal.sign * side.sign
    result.estimate = wp.float64(normal.sign) * side.estimate
    return result
