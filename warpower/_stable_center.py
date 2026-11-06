"""Filtered, correctly rounded float32 circumcenters of original float32 sites.

Only generated centers are rounded. Original sites are never changed. The fast
calculation is isolated from the rounding filter to preserve its FMA policy.
Uncertain rounding intervals use bounded integer numerator/denominator tests.
"""

import warp as wp

from . import _integer as integer

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})
ERROR_SCALE = wp.constant(wp.float64(64.0 * 2.0**-53))
MAX_FLOAT = wp.constant(float(3.4028234663852886e38))


@wp.struct
class Center:
    point: wp.vec3
    wide: wp.vec3d
    radius: wp.float64
    valid: int
    exact: int


@wp.struct
class Rounded:
    value: float
    valid: int


@wp.func
def steps(value: float):
    """Adjacent positive-magnitude float32 steps, including zero/subnormals."""
    up = integer.power2(-149)
    down = up
    if value != 0.0:
        data = integer.info(wp.abs(value))
        bits = data.mantissa
        exponent = data.exponent
        while bits > wp.uint32(1):
            bits >>= wp.uint32(1)
            exponent += 1
        up = integer.power2(wp.max(exponent - 23, -149))
        down = up
        if data.mantissa == wp.uint32(1) and exponent > -126:
            down /= wp.float64(2.0)
    return wp.vec2d(down, up)


@wp.func(inline=False)
def shift_left(a: integer.Big, shift: int):
    r = integer.empty()
    r.valid = a.valid
    if shift < 0:
        r.valid = 0
    if r.valid == 0 or a.sign == 0:
        return r
    words, bits = shift // 32, shift % 32
    r.sign = a.sign
    for i in range(a.count):
        at = i + words
        if at >= 48:
            r.valid = 0
            return r
        value = wp.uint64(a.words[i]) << wp.uint64(bits)
        r.words[at] |= wp.uint32(value & wp.uint64(0xFFFFFFFF))
        high = wp.uint32(value >> wp.uint64(32))
        if high != wp.uint32(0):
            if at + 1 >= 48:
                r.valid = 0
                return r
            r.words[at + 1] |= high
    r.count = wp.min(48, a.count + words + int(bits != 0))
    return integer.trim(r)


@wp.func(inline=False)
def round_ratio(numerator: integer.Big, denominator: integer.Big, base: int):
    """Round N/D * 2**base, D>0, using exact midpoint comparisons."""
    result = Rounded()
    if numerator.valid == 0 or denominator.valid == 0 or denominator.sign != 1:
        return result
    if numerator.sign == 0:
        result.valid = 1
        return result
    n = numerator
    n.sign = 1
    # Physical estimates avoid overflowing/underflowing either integer scale.
    estimate = (
        integer.predicate(n, 4 * base).estimate / integer.predicate(denominator, 3 * base).estimate
    )
    if not wp.isfinite(estimate) or estimate < wp.float64(0.0):
        return result
    candidate = wp.min(float(estimate), MAX_FLOAT)
    for attempt in range(2):
        spacing = steps(candidate)
        exponent = wp.min(base, integer.base_for(candidate, 1))
        exponent = wp.min(exponent, integer.base_for(float(spacing[0]), 1))
        exponent = wp.min(exponent, integer.base_for(float(spacing[1]), 1))
        left = shift_left(n, base - exponent + 1)
        twice_c = shift_left(integer.from_float(candidate, exponent), 1)
        upper = integer.times(
            denominator,
            integer.plus(twice_c, integer.from_float(float(spacing[1]), exponent)),
        )
        lower = integer.times(
            denominator,
            integer.minus(twice_c, integer.from_float(float(spacing[0]), exponent)),
        )
        if left.valid == 0 or upper.valid == 0 or lower.valid == 0:
            return result
        even = (int(wp.float64(candidate) / spacing[1]) & 1) == 0
        high = integer.compare(left, upper)
        low = int(1)
        if lower.sign >= 0:
            low = integer.compare(left, lower)
        if (low > 0 or (low == 0 and even)) and (high < 0 or (high == 0 and even)):
            result.value = float(numerator.sign) * candidate
            if candidate == 0.0:
                result.value = 0.0
            result.valid = 1
            return result
        if high > 0 or (high == 0 and not even):
            candidate = float(wp.float64(candidate) + spacing[1])
        else:
            candidate = float(wp.float64(candidate) - spacing[0])
        if not wp.isfinite(candidate) or candidate < 0.0:
            return result
    return result


@wp.func
def coordinate(row: integer.Row, axis: int):
    value = row.x
    if axis == 1:
        value = row.y
    elif axis == 2:
        value = row.z
    return value


@wp.func(inline=False)
def cofactor(b: integer.Row, c: integer.Row, axis: int):
    j, k = (axis + 1) % 3, (axis + 2) % 3
    return integer.minus(
        integer.times(coordinate(b, j), coordinate(c, k)),
        integer.times(coordinate(b, k), coordinate(c, j)),
    )


@wp.func(inline=False)
def exact_center(p0: wp.vec3, p1: wp.vec3, p2: wp.vec3, p3: wp.vec3):
    result = Center()
    result.exact = 1
    base = wp.min(
        wp.min(integer.vector_base(p0), integer.vector_base(p1)),
        wp.min(integer.vector_base(p2), integer.vector_base(p3)),
    )
    a, b, c = (
        integer.relative(p1, p0, base),
        integer.relative(p2, p0, base),
        integer.relative(p3, p0, base),
    )
    denominator = shift_left(integer.det3(a, b, c), 1)
    if denominator.valid == 0 or denominator.sign == 0:
        return result
    norms = integer.Row()
    norms.x = integer.norm(a)
    norms.y = integer.norm(b)
    norms.z = integer.norm(c)
    sign = denominator.sign
    positive_d = denominator
    positive_d.sign = 1
    d_value = integer.predicate(positive_d, 3 * base).estimate
    offset = wp.vec3d()
    for axis in range(3):
        relative_n = integer.plus(
            integer.plus(
                integer.times(norms.x, cofactor(b, c, axis)),
                integer.times(norms.y, cofactor(c, a, axis)),
            ),
            integer.times(norms.z, cofactor(a, b, axis)),
        )
        n = integer.plus(integer.times(denominator, integer.from_float(p0[axis], base)), relative_n)
        n.sign *= sign
        rounded = round_ratio(n, positive_d, base)
        if rounded.valid == 0:
            return result
        result.point[axis] = rounded.value
        result.wide[axis] = integer.predicate(n, 4 * base).estimate / d_value
        relative_n.sign *= sign
        offset[axis] = integer.predicate(relative_n, 4 * base).estimate / d_value
    # Evaluate the local offset directly: subtracting approximate global centers
    # loses radius accuracy for small cells far from the coordinate origin.
    result.radius = wp.length(offset)
    result.valid = int(wp.isfinite(result.radius) and result.radius > wp.float64(0.0))
    return result


@wp.func(inline=False)
def approximate_center(p0: wp.vec3, p1: wp.vec3, p2: wp.vec3, p3: wp.vec3):
    result = Center()
    a, b, c = wp.vec3d(p1) - wp.vec3d(p0), wp.vec3d(p2) - wp.vec3d(p0), wp.vec3d(p3) - wp.vec3d(p0)
    det = wp.dot(a, wp.cross(b, c))
    if det == wp.float64(0.0) or not wp.isfinite(det):
        return result
    offset = (
        wp.dot(a, a) * wp.cross(b, c)
        + wp.dot(b, b) * wp.cross(c, a)
        + wp.dot(c, c) * wp.cross(a, b)
    ) / (wp.float64(2.0) * det)
    result.wide = wp.vec3d(p0) + offset
    result.point = wp.vec3(result.wide)
    result.radius = wp.length(offset)
    result.valid = int(
        wp.isfinite(result.point[0])
        and wp.isfinite(result.point[1])
        and wp.isfinite(result.point[2])
        and wp.isfinite(result.radius)
        and result.radius > wp.float64(0.0)
    )
    return result


@wp.func
def cross_permanent(b: wp.vec3d, c: wp.vec3d):
    return wp.vec3d(
        wp.abs(b[1] * c[2]) + wp.abs(b[2] * c[1]),
        wp.abs(b[2] * c[0]) + wp.abs(b[0] * c[2]),
        wp.abs(b[0] * c[1]) + wp.abs(b[1] * c[0]),
    )


@wp.func(inline=False)
def rounding_proved(p0: wp.vec3, p1: wp.vec3, p2: wp.vec3, p3: wp.vec3, candidate: wp.vec3):
    a, b, c = wp.vec3d(p1) - wp.vec3d(p0), wp.vec3d(p2) - wp.vec3d(p0), wp.vec3d(p3) - wp.vec3d(p0)
    bc, ca, ab = wp.cross(b, c), wp.cross(c, a), wp.cross(a, b)
    bcp, cap, abp = cross_permanent(b, c), cross_permanent(c, a), cross_permanent(a, b)
    determinant = wp.dot(a, bc)
    dp = wp.abs(a[0]) * bcp[0] + wp.abs(a[1]) * bcp[1] + wp.abs(a[2]) * bcp[2]
    d = wp.float64(2.0) * determinant
    de = wp.float64(2.0) * ERROR_SCALE * dp
    if not wp.isfinite(d) or wp.abs(d) <= de:
        return False
    sign = wp.where(d > wp.float64(0.0), wp.float64(1.0), wp.float64(-1.0))
    aa, bb, cc = wp.dot(a, a), wp.dot(b, b), wp.dot(c, c)
    for axis in range(3):
        n = (wp.vec3d(p0)[axis] * d + aa * bc[axis] + bb * ca[axis] + cc * ab[axis]) * sign
        np = (
            wp.float64(2.0) * wp.abs(wp.vec3d(p0)[axis]) * dp
            + aa * bcp[axis]
            + bb * cap[axis]
            + cc * abp[axis]
        )
        spacing = steps(candidate[axis])
        lower = wp.float64(candidate[axis]) - spacing[0] / wp.float64(2.0)
        upper = wp.float64(candidate[axis]) + spacing[1] / wp.float64(2.0)
        if candidate[axis] < 0.0:
            lower = wp.float64(candidate[axis]) - spacing[1] / wp.float64(2.0)
            upper = wp.float64(candidate[axis]) + spacing[0] / wp.float64(2.0)
        for side in range(2):
            middle = wp.where(side == 0, lower, upper)
            product = wp.abs(d) * middle
            difference = n - product
            error = ERROR_SCALE * (np + wp.abs(n) + wp.abs(product)) + wp.abs(middle) * de
            if not wp.isfinite(error) or not wp.isfinite(difference):
                return False
            if side == 0 and difference <= error:
                return False
            if side == 1 and difference >= -error:
                return False
    return True


@wp.func(inline=False)
def circumcenter(p0: wp.vec3, p1: wp.vec3, p2: wp.vec3, p3: wp.vec3):
    result = approximate_center(p0, p1, p2, p3)
    if result.valid == 1:
        if rounding_proved(p0, p1, p2, p3, result.point):
            return result
    return exact_center(p0, p1, p2, p3)
