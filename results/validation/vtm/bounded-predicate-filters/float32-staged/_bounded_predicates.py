"""Bounded-range binary32 strict filters for private VTM consumers.

Formula grouping follows Shewchuk's public-domain May 18, 1996 predicates.c,
https://www.cs.cmu.edu/~quake/robust.html, via Warpower's existing first stages.
Only finite original float32 coordinates with magnitude <=16 enter the new
arithmetic. Generic rounding-factor and absolute-underflow bounds are checked
by benchmarks/bounded_predicate_proof.py. Every uncertain/unsupported result retains
the old float64/Ball64/integer/symbolic path. No tolerance defines a tie.
"""

import warp as wp

from . import _analytic_predicates as old
from . import _filtered_predicates as fp

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})
ORIENT_BOUND = wp.constant(wp.float32(2.0**-19))
SPHERE_BOUND = wp.constant(wp.float32(2.0**-17))
ABS_FLOOR = wp.constant(wp.float32(2.0**-64))


@wp.func(inline=True)
def bounded(p: wp.vec3):
    return (
        wp.isfinite(p[0])
        and wp.isfinite(p[1])
        and wp.isfinite(p[2])
        and wp.abs(p[0]) <= 16.0
        and wp.abs(p[1]) <= 16.0
        and wp.abs(p[2]) <= 16.0
    )


@wp.func(inline=True)
def unavailable():
    result = fp.Ball()
    result.value = wp.float64(0.0)
    result.error = wp.float64(wp.inf)
    return result


@wp.func
def orientation_filter(a: wp.vec3, b: wp.vec3, c: wp.vec3, d: wp.vec3):
    if not (bounded(a) and bounded(b) and bounded(c) and bounded(d)):
        return unavailable()
    (ax, ay, az) = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    (bx, by, bz) = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    (cx, cy, cz) = (d[0] - a[0], d[1] - a[1], d[2] - a[2])
    (bxcy, cxby) = (bx * cy, cx * by)
    (cxay, axcy) = (cx * ay, ax * cy)
    (axby, bxay) = (ax * by, bx * ay)
    det = az * (bxcy - cxby) + bz * (cxay - axcy) + cz * (axby - bxay)
    permanent = (
        (wp.abs(bxcy) + wp.abs(cxby)) * wp.abs(az)
        + (wp.abs(cxay) + wp.abs(axcy)) * wp.abs(bz)
        + (wp.abs(axby) + wp.abs(bxay)) * wp.abs(cz)
    )
    value = fp.Ball()
    value.value = wp.float64(det)
    value.error = wp.float64(ORIENT_BOUND * permanent + ABS_FLOOR)
    return value


@wp.func
def sphere_filter(a: wp.vec3, b: wp.vec3, c: wp.vec3, d: wp.vec3, q: wp.vec3):
    if not (bounded(a) and bounded(b) and bounded(c) and bounded(d) and bounded(q)):
        return unavailable()
    (ax, ay, az) = (a[0] - q[0], a[1] - q[1], a[2] - q[2])
    (bx, by, bz) = (b[0] - q[0], b[1] - q[1], b[2] - q[2])
    (cx, cy, cz) = (c[0] - q[0], c[1] - q[1], c[2] - q[2])
    (dx, dy, dz) = (d[0] - q[0], d[1] - q[1], d[2] - q[2])
    (axby, bxay) = (ax * by, bx * ay)
    (bxcy, cxby) = (bx * cy, cx * by)
    (cxdy, dxcy) = (cx * dy, dx * cy)
    (dxay, axdy) = (dx * ay, ax * dy)
    (axcy, cxay) = (ax * cy, cx * ay)
    (bxdy, dxby) = (bx * dy, dx * by)
    (ab, bc, cd, da, ac, bd) = (
        axby - bxay,
        bxcy - cxby,
        cxdy - dxcy,
        dxay - axdy,
        axcy - cxay,
        bxdy - dxby,
    )
    abc = az * bc - bz * ac + cz * ab
    bcd = bz * cd - cz * bd + dz * bc
    cda = cz * da + dz * ac + az * cd
    dab = dz * ab + az * bd + bz * da
    al = ax * ax + ay * ay + az * az
    bl = bx * bx + by * by + bz * bz
    cl = cx * cx + cy * cy + cz * cz
    dl = dx * dx + dy * dy + dz * dz
    det = -(dl * abc - cl * dab + (bl * cda - al * bcd))
    pa = (
        (wp.abs(cxdy) + wp.abs(dxcy)) * wp.abs(bz)
        + (wp.abs(dxby) + wp.abs(bxdy)) * wp.abs(cz)
        + (wp.abs(bxcy) + wp.abs(cxby)) * wp.abs(dz)
    ) * al
    pb = (
        (wp.abs(dxay) + wp.abs(axdy)) * wp.abs(cz)
        + (wp.abs(axcy) + wp.abs(cxay)) * wp.abs(dz)
        + (wp.abs(cxdy) + wp.abs(dxcy)) * wp.abs(az)
    ) * bl
    pc = (
        (wp.abs(axby) + wp.abs(bxay)) * wp.abs(dz)
        + (wp.abs(bxdy) + wp.abs(dxby)) * wp.abs(az)
        + (wp.abs(dxay) + wp.abs(axdy)) * wp.abs(bz)
    ) * cl
    pd = (
        (wp.abs(bxcy) + wp.abs(cxby)) * wp.abs(az)
        + (wp.abs(cxay) + wp.abs(axcy)) * wp.abs(bz)
        + (wp.abs(axby) + wp.abs(bxay)) * wp.abs(cz)
    ) * dl
    value = fp.Ball()
    value.value = wp.float64(det)
    value.error = wp.float64(SPHERE_BOUND * (pa + pb + pc + pd) + ABS_FLOOR)
    return value


@wp.func(inline=True)
def orient3d(a: wp.vec3, b: wp.vec3, c: wp.vec3, d: wp.vec3):
    value = orientation_filter(a, b, c, d)
    if old.strict(value):
        return fp.predicate(value)
    return old.orient3d(a, b, c, d)


@wp.func(inline=True)
def symbolic_sphere(points: wp.array[wp.vec3], ids: wp.vec4i, query: int):
    value = sphere_filter(
        points[ids[0]], points[ids[1]], points[ids[2]], points[ids[3]], points[query]
    )
    if old.strict(value):
        return fp.predicate(value)
    return old.symbolic_sphere(points, ids, query)
