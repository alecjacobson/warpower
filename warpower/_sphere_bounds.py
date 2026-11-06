"""Conservative float32 boxes of original-coordinate circumspheres.

Calling kernels must disable fast_math and fused multiply-add, as for the
underlying float64 Ball arithmetic. Failure to construct finite bounds means
that callers must retain their original exact-safe traversal.
"""

import warp as wp

from . import _filtered_predicates as fp

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})
MIN64 = wp.constant(wp.float64(2.0**-1022))
MIN32 = wp.constant(wp.float64(2.0**-126))
EPS32 = wp.constant(wp.float64(2.0**-23))


@wp.struct
class Box:
    lower: wp.vec3
    upper: wp.vec3
    valid: int


@wp.func
def quotient(a: fp.Ball, b: fp.Ball):
    result = fp.Ball()
    result.value = wp.float64(0.0)
    result.error = wp.float64(wp.inf)
    if (
        not wp.isfinite(a.value)
        or not wp.isfinite(a.error)
        or not wp.isfinite(b.value)
        or not wp.isfinite(b.error)
        or b.value <= b.error
    ):
        return result
    q = a.value / b.value
    # EPS is twice unit roundoff. For rounded q=a.value/b.value,
    # |a.value-b.value*q| <= EPS*b.value*|q|. Input interval uncertainty
    # adds a.error+b.error*|q|. Divide by a downward-padded positive lower
    # bound on the denominator, then pad the positive arithmetic and division.
    denominator = (b.value - b.error) * (wp.float64(1.0) - wp.float64(4.0) * fp.EPS)
    if denominator <= wp.float64(0.0) or not wp.isfinite(q):
        return result
    numerator = (a.error + wp.abs(q) * b.error + (fp.EPS * b.value) * wp.abs(q)) * (
        wp.float64(1.0) + wp.float64(16.0) * fp.EPS
    )
    error = (numerator / denominator) * (wp.float64(1.0) + wp.float64(16.0) * fp.EPS)
    error += wp.float64(16.0) * MIN64
    if wp.isfinite(error):
        result.value = q
        result.error = error
    return result


@wp.func
def bounds(origin: wp.vec3, nx: fp.Ball, ny: fp.Ball, nz: fp.Ball, determinant: fp.Ball):
    result = Box()
    result.lower = wp.vec3(0.0)
    result.upper = wp.vec3(0.0)
    result.valid = 0
    half = fp.scalar(wp.float64(0.5))
    x = fp.times(quotient(nx, determinant), half)
    y = fp.times(quotient(ny, determinant), half)
    z = fp.times(quotient(nz, determinant), half)
    if not wp.isfinite(x.error) or not wp.isfinite(y.error) or not wp.isfinite(z.error):
        return result
    radius2 = fp.plus(fp.plus(fp.times(x, x), fp.times(y, y)), fp.times(z, z))
    upper = (radius2.value + radius2.error) * (wp.float64(1.0) + wp.float64(4.0) * fp.EPS)
    upper += wp.float64(16.0) * MIN64
    if not wp.isfinite(upper) or upper <= wp.float64(0.0):
        return result
    radius = wp.sqrt(upper) * (wp.float64(1.0) + wp.float64(4.0) * fp.EPS)
    radius += wp.float64(16.0) * MIN64
    values, errors = wp.vec3d(x.value, y.value, z.value), wp.vec3d(x.error, y.error, z.error)
    for axis in range(3):
        center = fp.Ball()
        center.value = values[axis]
        center.error = errors[axis]
        center = fp.plus(fp.scalar(wp.float64(origin[axis])), center)
        margin = wp.float64(16.0) * fp.EPS * (wp.abs(center.value) + center.error + radius)
        margin += wp.float64(16.0) * MIN64
        lo = center.value - center.error - radius - margin
        hi = center.value + center.error + radius + margin
        # Pad before the float32 conversion. The absolute normal allowance
        # also encloses conversion/flush behavior throughout the subnormal range.
        lo = lo - wp.float64(4.0) * EPS32 * wp.abs(lo) - wp.float64(4.0) * MIN32
        hi = hi + wp.float64(4.0) * EPS32 * wp.abs(hi) + wp.float64(4.0) * MIN32
        lower, upper32 = float(lo), float(hi)
        if not wp.isfinite(lower) or not wp.isfinite(upper32) or lower > upper32:
            return result
        result.lower[axis] = lower
        result.upper[axis] = upper32
    result.valid = 1
    return result


@wp.func(inline=True)
def outside(box: Box, lower: wp.vec3, upper: wp.vec3):
    return box.valid != 0 and (
        upper[0] < box.lower[0]
        or lower[0] > box.upper[0]
        or upper[1] < box.lower[1]
        or lower[1] > box.upper[1]
        or upper[2] < box.lower[2]
        or lower[2] > box.upper[2]
    )
