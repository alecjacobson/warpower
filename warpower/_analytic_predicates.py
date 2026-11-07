"""Analytic first-stage filters for sign-only VTM facet searches.

Adapted from Jonathan Richard Shewchuk's public-domain predicates.c (May 1996),
orient3d/insphere first stages. https://www.cs.cmu.edu/~quake/robust.html
Original finite float32 coordinates are promoted BEFORE any arithmetic.
Calling kernels require fast_math=False and fuse_fp=False. Grouping and strict
nonzero acceptance are essential. Uncertain signs use the unchanged filters and
exact fallback. Consumers must use signs, not assume the old estimate bits.
"""

import warp as wp

from . import _filtered_predicates as fp
from ._delaunay_certificate import symbolic_sphere as legacy_symbolic_sphere

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})
# Exact powers of two, greater than the published first-stage coefficients
# (7+56u)u and (16+224u)u, even with conservative u=2^-52 (actual u=2^-53).
ORIENT_BOUND = wp.constant(wp.float64(2.0**-49))
SPHERE_BOUND = wp.constant(wp.float64(2.0**-47))


@wp.func
def orientation_filter(a: wp.vec3, b: wp.vec3, c: wp.vec3, d: wp.vec3):
    # Shewchuk's pa,pb,pc,pd = b,c,d,a gives Warpower's positive convention.
    ax, ay, az = (
        wp.float64(b[0]) - wp.float64(a[0]),
        wp.float64(b[1]) - wp.float64(a[1]),
        wp.float64(b[2]) - wp.float64(a[2]),
    )
    bx, by, bz = (
        wp.float64(c[0]) - wp.float64(a[0]),
        wp.float64(c[1]) - wp.float64(a[1]),
        wp.float64(c[2]) - wp.float64(a[2]),
    )
    cx, cy, cz = (
        wp.float64(d[0]) - wp.float64(a[0]),
        wp.float64(d[1]) - wp.float64(a[1]),
        wp.float64(d[2]) - wp.float64(a[2]),
    )
    bxcy, cxby = bx * cy, cx * by
    cxay, axcy = cx * ay, ax * cy
    axby, bxay = ax * by, bx * ay
    det = (az * (bxcy - cxby) + bz * (cxay - axcy)) + cz * (axby - bxay)
    permanent = (
        (wp.abs(bxcy) + wp.abs(cxby)) * wp.abs(az) + (wp.abs(cxay) + wp.abs(axcy)) * wp.abs(bz)
    ) + (wp.abs(axby) + wp.abs(bxay)) * wp.abs(cz)
    value = fp.Ball()
    value.value = det
    value.error = ORIENT_BOUND * permanent
    return value


@wp.func
def sphere_filter(a: wp.vec3, b: wp.vec3, c: wp.vec3, d: wp.vec3, q: wp.vec3):
    ax, ay, az = (
        wp.float64(a[0]) - wp.float64(q[0]),
        wp.float64(a[1]) - wp.float64(q[1]),
        wp.float64(a[2]) - wp.float64(q[2]),
    )
    bx, by, bz = (
        wp.float64(b[0]) - wp.float64(q[0]),
        wp.float64(b[1]) - wp.float64(q[1]),
        wp.float64(b[2]) - wp.float64(q[2]),
    )
    cx, cy, cz = (
        wp.float64(c[0]) - wp.float64(q[0]),
        wp.float64(c[1]) - wp.float64(q[1]),
        wp.float64(c[2]) - wp.float64(q[2]),
    )
    dx, dy, dz = (
        wp.float64(d[0]) - wp.float64(q[0]),
        wp.float64(d[1]) - wp.float64(q[1]),
        wp.float64(d[2]) - wp.float64(q[2]),
    )
    axby, bxay = ax * by, bx * ay
    bxcy, cxby = bx * cy, cx * by
    cxdy, dxcy = cx * dy, dx * cy
    dxay, axdy = dx * ay, ax * dy
    axcy, cxay = ax * cy, cx * ay
    bxdy, dxby = bx * dy, dx * by
    ab, bc, cd, da, ac, bd = (
        axby - bxay,
        bxcy - cxby,
        cxdy - dxcy,
        dxay - axdy,
        axcy - cxay,
        bxdy - dxby,
    )
    abc = (az * bc - bz * ac) + cz * ab
    bcd = (bz * cd - cz * bd) + dz * bc
    cda = (cz * da + dz * ac) + az * cd
    dab = (dz * ab + az * bd) + bz * da
    al = (ax * ax + ay * ay) + az * az
    bl = (bx * bx + by * by) + bz * bz
    cl = (cx * cx + cy * cy) + cz * cz
    dl = (dx * dx + dy * dy) + dz * dz
    # Warpower's sphere sign is the negative of Shewchuk's convention.
    det = -((dl * abc - cl * dab) + (bl * cda - al * bcd))
    pa = (
        ((wp.abs(cxdy) + wp.abs(dxcy)) * wp.abs(bz) + (wp.abs(dxby) + wp.abs(bxdy)) * wp.abs(cz))
        + (wp.abs(bxcy) + wp.abs(cxby)) * wp.abs(dz)
    ) * al
    pb = (
        ((wp.abs(dxay) + wp.abs(axdy)) * wp.abs(cz) + (wp.abs(axcy) + wp.abs(cxay)) * wp.abs(dz))
        + (wp.abs(cxdy) + wp.abs(dxcy)) * wp.abs(az)
    ) * bl
    pc = (
        ((wp.abs(axby) + wp.abs(bxay)) * wp.abs(dz) + (wp.abs(bxdy) + wp.abs(dxby)) * wp.abs(az))
        + (wp.abs(dxay) + wp.abs(axdy)) * wp.abs(bz)
    ) * cl
    pd = (
        ((wp.abs(bxcy) + wp.abs(cxby)) * wp.abs(az) + (wp.abs(cxay) + wp.abs(axcy)) * wp.abs(bz))
        + (wp.abs(axby) + wp.abs(bxay)) * wp.abs(cz)
    ) * dl
    value = fp.Ball()
    value.value = det
    value.error = SPHERE_BOUND * (((pa + pb) + pc) + pd)
    return value


@wp.func
def strict(value: fp.Ball):
    return (
        wp.isfinite(value.value) and wp.isfinite(value.error) and wp.abs(value.value) > value.error
    )


@wp.func(inline=True)
def orient3d(a: wp.vec3, b: wp.vec3, c: wp.vec3, d: wp.vec3):
    value = orientation_filter(a, b, c, d)
    if strict(value):
        return fp.predicate(value)
    return fp.orient3d(a, b, c, d)


@wp.func(inline=True)
def symbolic_sphere(points: wp.array[wp.vec3], ids: wp.vec4i, query: int):
    value = sphere_filter(
        points[ids[0]], points[ids[1]], points[ids[2]], points[ids[3]], points[query]
    )
    if strict(value):
        return fp.predicate(value)
    return legacy_symbolic_sphere(points, ids, query)
