"""Test-only explicit flush variant of the bounded binary32 arithmetic."""

import warp as wp

from warpower import _filtered_predicates as fp
from warpower._bounded_predicates import ABS_FLOOR, ORIENT_BOUND, SPHERE_BOUND, bounded, unavailable

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})
MIN_NORMAL = wp.constant(wp.float32(2.0**-126))


@wp.func(inline=True)
def flush(value: float):
    if wp.abs(value) < MIN_NORMAL:
        return 0.0
    return value


@wp.func(inline=True)
def flushed_add(a: float, b: float):
    return flush(flush(a) + flush(b))


@wp.func(inline=True)
def flushed_sub(a: float, b: float):
    return flush(flush(a) - flush(b))


@wp.func(inline=True)
def flushed_mul(a: float, b: float):
    return flush(flush(a) * flush(b))


@wp.func(inline=True)
def flushed_neg(a: float):
    return flush(-flush(a))


@wp.func(inline=True)
def flushed_abs(a: float):
    return flush(wp.abs(flush(a)))


@wp.func
def orientation_filter(a: wp.vec3, b: wp.vec3, c: wp.vec3, d: wp.vec3):
    if not (bounded(a) and bounded(b) and bounded(c) and bounded(d)):
        return unavailable()
    (ax, ay, az) = (flushed_sub(b[0], a[0]), flushed_sub(b[1], a[1]), flushed_sub(b[2], a[2]))
    (bx, by, bz) = (flushed_sub(c[0], a[0]), flushed_sub(c[1], a[1]), flushed_sub(c[2], a[2]))
    (cx, cy, cz) = (flushed_sub(d[0], a[0]), flushed_sub(d[1], a[1]), flushed_sub(d[2], a[2]))
    (bxcy, cxby) = (flushed_mul(bx, cy), flushed_mul(cx, by))
    (cxay, axcy) = (flushed_mul(cx, ay), flushed_mul(ax, cy))
    (axby, bxay) = (flushed_mul(ax, by), flushed_mul(bx, ay))
    det = flushed_add(
        flushed_add(
            flushed_mul(az, flushed_sub(bxcy, cxby)), flushed_mul(bz, flushed_sub(cxay, axcy))
        ),
        flushed_mul(cz, flushed_sub(axby, bxay)),
    )
    permanent = flushed_add(
        flushed_add(
            flushed_mul(flushed_add(flushed_abs(bxcy), flushed_abs(cxby)), flushed_abs(az)),
            flushed_mul(flushed_add(flushed_abs(cxay), flushed_abs(axcy)), flushed_abs(bz)),
        ),
        flushed_mul(flushed_add(flushed_abs(axby), flushed_abs(bxay)), flushed_abs(cz)),
    )
    value = fp.Ball()
    value.value = wp.float64(det)
    value.error = wp.float64(flushed_add(flushed_mul(ORIENT_BOUND, permanent), ABS_FLOOR))
    return value


@wp.func
def sphere_filter(a: wp.vec3, b: wp.vec3, c: wp.vec3, d: wp.vec3, q: wp.vec3):
    if not (bounded(a) and bounded(b) and bounded(c) and bounded(d) and bounded(q)):
        return unavailable()
    (ax, ay, az) = (flushed_sub(a[0], q[0]), flushed_sub(a[1], q[1]), flushed_sub(a[2], q[2]))
    (bx, by, bz) = (flushed_sub(b[0], q[0]), flushed_sub(b[1], q[1]), flushed_sub(b[2], q[2]))
    (cx, cy, cz) = (flushed_sub(c[0], q[0]), flushed_sub(c[1], q[1]), flushed_sub(c[2], q[2]))
    (dx, dy, dz) = (flushed_sub(d[0], q[0]), flushed_sub(d[1], q[1]), flushed_sub(d[2], q[2]))
    (axby, bxay) = (flushed_mul(ax, by), flushed_mul(bx, ay))
    (bxcy, cxby) = (flushed_mul(bx, cy), flushed_mul(cx, by))
    (cxdy, dxcy) = (flushed_mul(cx, dy), flushed_mul(dx, cy))
    (dxay, axdy) = (flushed_mul(dx, ay), flushed_mul(ax, dy))
    (axcy, cxay) = (flushed_mul(ax, cy), flushed_mul(cx, ay))
    (bxdy, dxby) = (flushed_mul(bx, dy), flushed_mul(dx, by))
    (ab, bc, cd, da, ac, bd) = (
        flushed_sub(axby, bxay),
        flushed_sub(bxcy, cxby),
        flushed_sub(cxdy, dxcy),
        flushed_sub(dxay, axdy),
        flushed_sub(axcy, cxay),
        flushed_sub(bxdy, dxby),
    )
    abc = flushed_add(flushed_sub(flushed_mul(az, bc), flushed_mul(bz, ac)), flushed_mul(cz, ab))
    bcd = flushed_add(flushed_sub(flushed_mul(bz, cd), flushed_mul(cz, bd)), flushed_mul(dz, bc))
    cda = flushed_add(flushed_add(flushed_mul(cz, da), flushed_mul(dz, ac)), flushed_mul(az, cd))
    dab = flushed_add(flushed_add(flushed_mul(dz, ab), flushed_mul(az, bd)), flushed_mul(bz, da))
    al = flushed_add(flushed_add(flushed_mul(ax, ax), flushed_mul(ay, ay)), flushed_mul(az, az))
    bl = flushed_add(flushed_add(flushed_mul(bx, bx), flushed_mul(by, by)), flushed_mul(bz, bz))
    cl = flushed_add(flushed_add(flushed_mul(cx, cx), flushed_mul(cy, cy)), flushed_mul(cz, cz))
    dl = flushed_add(flushed_add(flushed_mul(dx, dx), flushed_mul(dy, dy)), flushed_mul(dz, dz))
    det = flushed_neg(
        flushed_add(
            flushed_sub(flushed_mul(dl, abc), flushed_mul(cl, dab)),
            flushed_sub(flushed_mul(bl, cda), flushed_mul(al, bcd)),
        )
    )
    pa = flushed_mul(
        flushed_add(
            flushed_add(
                flushed_mul(flushed_add(flushed_abs(cxdy), flushed_abs(dxcy)), flushed_abs(bz)),
                flushed_mul(flushed_add(flushed_abs(dxby), flushed_abs(bxdy)), flushed_abs(cz)),
            ),
            flushed_mul(flushed_add(flushed_abs(bxcy), flushed_abs(cxby)), flushed_abs(dz)),
        ),
        al,
    )
    pb = flushed_mul(
        flushed_add(
            flushed_add(
                flushed_mul(flushed_add(flushed_abs(dxay), flushed_abs(axdy)), flushed_abs(cz)),
                flushed_mul(flushed_add(flushed_abs(axcy), flushed_abs(cxay)), flushed_abs(dz)),
            ),
            flushed_mul(flushed_add(flushed_abs(cxdy), flushed_abs(dxcy)), flushed_abs(az)),
        ),
        bl,
    )
    pc = flushed_mul(
        flushed_add(
            flushed_add(
                flushed_mul(flushed_add(flushed_abs(axby), flushed_abs(bxay)), flushed_abs(dz)),
                flushed_mul(flushed_add(flushed_abs(bxdy), flushed_abs(dxby)), flushed_abs(az)),
            ),
            flushed_mul(flushed_add(flushed_abs(dxay), flushed_abs(axdy)), flushed_abs(bz)),
        ),
        cl,
    )
    pd = flushed_mul(
        flushed_add(
            flushed_add(
                flushed_mul(flushed_add(flushed_abs(bxcy), flushed_abs(cxby)), flushed_abs(az)),
                flushed_mul(flushed_add(flushed_abs(cxay), flushed_abs(axcy)), flushed_abs(bz)),
            ),
            flushed_mul(flushed_add(flushed_abs(axby), flushed_abs(bxay)), flushed_abs(cz)),
        ),
        dl,
    )
    value = fp.Ball()
    value.value = wp.float64(det)
    value.error = wp.float64(
        flushed_add(
            flushed_mul(SPHERE_BOUND, flushed_add(flushed_add(flushed_add(pa, pb), pc), pd)),
            ABS_FLOOR,
        )
    )
    return value
