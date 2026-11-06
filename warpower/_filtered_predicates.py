"""Roundoff-bounded float64 filters with exact expansion fallbacks.

Calling kernels must set fast_math=False and fuse_fp=False. Bounds assume finite
float32 input coordinates/weights, so all nonzero degree-five products and their
roundoff bounds are normal float64 numbers. No tolerance is treated as a tie.
"""

import warp as wp

from . import _predicates as exact

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})
EPS = wp.constant(wp.float64(2.0**-52))
ADD_PAD = wp.constant(wp.float64(1.0 + 4.0 * 2.0**-52))
MUL_PAD = wp.constant(wp.float64(1.0 + 8.0 * 2.0**-52))


@wp.struct
class Ball:
    value: wp.float64
    error: wp.float64


@wp.func
def scalar(value: wp.float64):
    result = Ball()
    result.value = value
    result.error = wp.float64(0.0)
    return result


@wp.func
def plus(a: Ball, b: Ball):
    result = Ball()
    result.value = a.value + b.value
    result.error = (a.error + b.error + EPS * wp.abs(result.value)) * ADD_PAD
    return result


@wp.func
def minus(a: Ball, b: Ball):
    result = Ball()
    result.value = a.value - b.value
    result.error = (a.error + b.error + EPS * wp.abs(result.value)) * ADD_PAD
    return result


@wp.func
def times(a: Ball, b: Ball):
    result = Ball()
    result.value = a.value * b.value
    result.error = (
        wp.abs(a.value) * b.error
        + wp.abs(b.value) * a.error
        + a.error * b.error
        + EPS * wp.abs(result.value)
    ) * MUL_PAD
    return result


@wp.func
def difference(a: float, b: float):
    return minus(scalar(wp.float64(a)), scalar(wp.float64(b)))


@wp.func
def cross2(ax: Ball, ay: Ball, bx: Ball, by: Ball):
    return minus(times(ax, by), times(bx, ay))


@wp.func
def det3(ax: Ball, ay: Ball, az: Ball, bx: Ball, by: Ball, bz: Ball, cx: Ball, cy: Ball, cz: Ball):
    return plus(
        minus(times(az, cross2(bx, by, cx, cy)), times(bz, cross2(ax, ay, cx, cy))),
        times(cz, cross2(ax, ay, bx, by)),
    )


@wp.func
def orientation_ball(a: wp.vec3, b: wp.vec3, c: wp.vec3, d: wp.vec3):
    return det3(
        difference(b[0], a[0]),
        difference(b[1], a[1]),
        difference(b[2], a[2]),
        difference(c[0], a[0]),
        difference(c[1], a[1]),
        difference(c[2], a[2]),
        difference(d[0], a[0]),
        difference(d[1], a[1]),
        difference(d[2], a[2]),
    )


@wp.func
def lift(x: Ball, y: Ball, z: Ball, weight: float, query_weight: float):
    return plus(plus(times(x, x), times(y, y)), plus(times(z, z), difference(query_weight, weight)))


@wp.func
def sphere_ball(
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    d: wp.vec3,
    q: wp.vec3,
    wa: float,
    wb: float,
    wc: float,
    wd: float,
    wq: float,
):
    ax, ay, az = difference(a[0], q[0]), difference(a[1], q[1]), difference(a[2], q[2])
    bx, by, bz = difference(b[0], q[0]), difference(b[1], q[1]), difference(b[2], q[2])
    cx, cy, cz = difference(c[0], q[0]), difference(c[1], q[1]), difference(c[2], q[2])
    dx, dy, dz = difference(d[0], q[0]), difference(d[1], q[1]), difference(d[2], q[2])
    al, bl = lift(ax, ay, az, wa, wq), lift(bx, by, bz, wb, wq)
    cl, dl = lift(cx, cy, cz, wc, wq), lift(dx, dy, dz, wd, wq)
    abc = det3(ax, ay, az, bx, by, bz, cx, cy, cz)
    bcd = det3(bx, by, bz, cx, cy, cz, dx, dy, dz)
    cda = det3(cx, cy, cz, dx, dy, dz, ax, ay, az)
    dab = det3(dx, dy, dz, ax, ay, az, bx, by, bz)
    return plus(minus(times(al, bcd), times(bl, cda)), minus(times(cl, dab), times(dl, abc)))


@wp.func
def certain(value: Ball):
    return (
        wp.isfinite(value.value)
        and wp.isfinite(value.error)
        and (wp.abs(value.value) > value.error or value.error == wp.float64(0.0))
    )


@wp.func
def predicate(value: Ball):
    result = exact.Predicate()
    result.valid = 1
    result.estimate = value.value
    result.sign = int(value.value > wp.float64(0.0)) - int(value.value < wp.float64(0.0))
    return result


@wp.func
def integer_difference(a: float, b: float):
    """Whether the original difference is an exact integer in [-512, 512]."""
    x, y = wp.float64(a), wp.float64(b)
    value = x - y
    if not wp.isfinite(value) or wp.abs(value) > wp.float64(512.0):
        return False
    # Error-free TwoDiff. A rounded integer alone is insufficient: a tiny
    # operand may have disappeared during subtraction from a larger integer.
    b_virtual = x - value
    a_virtual = value + b_virtual
    residual = (x - a_virtual) + (b_virtual - y)
    return residual == wp.float64(0.0) and value == wp.float64(int(value))


@wp.func
def integer_vector(a: wp.vec3, b: wp.vec3):
    return (
        integer_difference(a[0], b[0])
        and integer_difference(a[1], b[1])
        and integer_difference(a[2], b[2])
    )


@wp.func
def integer_weight(value: float):
    if not wp.isfinite(value) or wp.abs(value) > 262144.0:
        return False
    return wp.float64(value) == wp.float64(int(value))


@wp.func
def integer_sphere(
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    d: wp.vec3,
    q: wp.vec3,
    wa: float,
    wb: float,
    wc: float,
    wd: float,
    wq: float,
):
    # With M=512 and W=M^2, all partial integer determinant sums/products are
    # bounded by 24*M^3*(3*M^2+2W) = 120*512^5 < 2^52. Thus sphere_ball.value
    # evaluates exactly in binary64, even if its conservative ball spans zero.
    return (
        integer_vector(a, q)
        and integer_vector(b, q)
        and integer_vector(c, q)
        and integer_vector(d, q)
        and integer_weight(wa)
        and integer_weight(wb)
        and integer_weight(wc)
        and integer_weight(wd)
        and integer_weight(wq)
    )


@wp.func(inline=True)
def orient3d(a: wp.vec3, b: wp.vec3, c: wp.vec3, d: wp.vec3):
    result = orientation_ball(a, b, c, d)
    if certain(result):
        return predicate(result)
    # Every partial triple product/sum is at most 6*512^3 < 2^30.
    if integer_vector(b, a) and integer_vector(c, a) and integer_vector(d, a):
        return predicate(result)
    return exact.orient3d(a, b, c, d)


@wp.func(inline=True)
def power_sphere(
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    d: wp.vec3,
    q: wp.vec3,
    wa: float,
    wb: float,
    wc: float,
    wd: float,
    wq: float,
):
    result = sphere_ball(a, b, c, d, q, wa, wb, wc, wd, wq)
    if certain(result):
        return predicate(result)
    if integer_sphere(a, b, c, d, q, wa, wb, wc, wd, wq):
        return predicate(result)
    return exact.power_sphere(a, b, c, d, q, wa, wb, wc, wd, wq)


@wp.func(inline=True)
def insphere(a: wp.vec3, b: wp.vec3, c: wp.vec3, d: wp.vec3, q: wp.vec3):
    return power_sphere(a, b, c, d, q, 0.0, 0.0, 0.0, 0.0, 0.0)
