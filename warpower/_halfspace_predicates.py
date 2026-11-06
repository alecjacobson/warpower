"""Original-float32 site/box halfspace signs, including symbolic lifting ties."""

import warp as wp

from . import _filtered_predicates as fp
from . import _predicates as exact
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


@wp.func(inline=False)
def exact_det(
    points: wp.array[wp.vec3],
    bounds: wp.array[wp.vec3],
    owner: int,
    ids: wp.vec4i,
    mode: int,
    selected: int,
):
    # Each site row is (-2*x, |x|^2-weight, 1). Box rows have last
    # coordinate zero. Subtracting the owner row gives the original halfspace
    # equations; expansion along the final column yields their 4x4 determinant.
    rows = Ids5(ids[0], ids[1], ids[2], ids[3], owner)
    positions = exact.Positions5()
    constant = exact.Weights5()
    sites = Ids5(0)
    total = exact.empty_expansion()
    for r in range(5):
        index = rows[r]
        if index >= 0:
            p = points[index]
            sites[r] = 1
            for k in range(3):
                positions[r, k] = wp.float64(-2.0) * wp.float64(p[k])
                if not wp.isfinite(positions[r, k]):
                    total.valid = 0
            constant[r] = -wp.float64(weight(index, mode, selected))
        elif index == -7:
            constant[r] = wp.float64(1.0)
        else:
            plane = -index - 1
            axis, side = plane // 2, plane % 2
            sign = wp.float64(1 - 2 * side)
            positions[r, axis] = sign
            constant[r] = -sign * wp.float64(bounds[side][axis])
            if not wp.isfinite(constant[r]):
                total.valid = 0
    for k in range(120):
        code = exact.POWER_TERMS[k]
        a, b, c, d = code & 7, (code >> 3) & 7, (code >> 6) & 7, (code >> 9) & 7
        last = 10 - a - b - c - d
        if (
            sites[last] == 0
            or positions[a, 0] == wp.float64(0.0)
            or positions[b, 1] == wp.float64(0.0)
            or positions[c, 2] == wp.float64(0.0)
        ):
            continue
        term = exact.grow(exact.empty_expansion(), constant[d])
        if sites[d] != 0:
            for j in range(3):
                # Power-of-two scaling and each original float32 square are
                # exact in float64, even though their sum need not be.
                value = positions[d, j] * wp.float64(-0.5)
                term = exact.grow(term, value * value)
        term = exact.scale(term, positions[a, 0])
        term = exact.scale(term, positions[b, 1])
        term = exact.scale(term, positions[c, 2])
        total = exact.add(total, term, wp.float64(1 - 2 * ((code >> 12) & 1)))
    result = exact.negative_determinant(total)
    result.sign = -result.sign
    result.estimate = -result.estimate
    return result


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
