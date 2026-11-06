"""Bounded exact dyadic arithmetic for finite float32 geometry.

48 little-endian uint32 limbs hold the conservative <1400-bit determinants used
here, with explicit overflow failure. All arithmetic runs as pure Warp code.
See docs/integer-predicates.md for the range argument and validation scope.
"""

import warp as wp

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})
Words = wp.types.vector(48, wp.uint32)
Powers = wp.constant(wp.types.vector(10, wp.float64)(*[2.0 ** (1 << i) for i in range(10)]))


@wp.struct
class Predicate:
    sign: int
    estimate: wp.float64
    valid: int


@wp.struct
class Info:
    mantissa: wp.uint32
    exponent: int
    sign: int
    valid: int


@wp.func
def info(x: float):
    r = Info()
    r.valid = 1
    if not wp.isfinite(x):
        r.valid = 0
        return r
    if x == 0.0:
        return r
    r.sign = wp.where(x > 0.0, 1, -1)
    powers = Powers
    value = wp.abs(wp.float64(x))
    exponent = int(0)
    if value >= wp.float64(1.0):
        for j in range(7, -1, -1):
            if value >= powers[j]:
                value /= powers[j]
                exponent += 1 << j
    else:
        for j in range(7, -1, -1):
            if value < wp.float64(1.0) / powers[j]:
                value *= powers[j]
                exponent -= 1 << j
        if value < wp.float64(1.0):
            value *= wp.float64(2.0)
            exponent -= 1
    r.mantissa = wp.uint32(value * wp.float64(8388608.0))
    r.exponent = exponent - 23
    while (r.mantissa & wp.uint32(1)) == wp.uint32(0):
        r.mantissa >>= wp.uint32(1)
        r.exponent += 1
    return r


@wp.func
def base_for(x: float, degree: int):
    r = info(x)
    exponent = int(0)
    if r.sign != 0:
        exponent = r.exponent
        if degree == 2:
            # Explicit floor division, including negative odd exponents.
            exponent = wp.where(exponent < 0, -((-exponent + 1) // 2), exponent // 2)
    return wp.min(0, exponent)


@wp.struct
class Big:
    words: Words
    count: int
    sign: int
    valid: int


@wp.func
def empty():
    r = Big()
    r.valid = 1
    return r


@wp.func
def trim(a: Big):
    r = a
    while r.count > 0:
        if r.words[r.count - 1] != wp.uint32(0):
            break
        r.count -= 1
    if r.count == 0:
        r.sign = 0
    return r


@wp.func
def from_float(x: float, base: int):
    a = info(x)
    r = empty()
    r.valid = a.valid
    if a.sign == 0 or a.valid == 0:
        return r
    shift = a.exponent - base
    if shift < 0:
        r.valid = 0
        return r
    index = shift // 32
    bits = shift % 32
    if index >= 47:
        r.valid = 0
        return r
    value = wp.uint64(a.mantissa) << wp.uint64(bits)
    r.words[index] = wp.uint32(value & wp.uint64(0xFFFFFFFF))
    r.words[index + 1] = wp.uint32(value >> wp.uint64(32))
    r.count = index + 1 + int(r.words[index + 1] != wp.uint32(0))
    r.sign = a.sign
    return r


@wp.func
def compare(a: Big, b: Big):
    if a.count != b.count:
        return wp.where(a.count > b.count, 1, -1)
    for i in range(a.count - 1, -1, -1):
        if a.words[i] != b.words[i]:
            return wp.where(a.words[i] > b.words[i], 1, -1)
    return 0


@wp.func
def neg(a: Big):
    r = a
    r.sign = -r.sign
    return r


@wp.func(inline=False)
def plus(a: Big, b: Big):
    r = empty()
    r.valid = a.valid & b.valid
    if r.valid == 0:
        return r
    if a.sign == 0:
        return b
    if b.sign == 0:
        return a
    if a.sign == b.sign:
        r.sign = a.sign
        r.count = wp.max(a.count, b.count)
        carry = wp.uint64(0)
        for i in range(r.count):
            value = wp.uint64(a.words[i]) + wp.uint64(b.words[i]) + carry
            r.words[i] = wp.uint32(value & wp.uint64(0xFFFFFFFF))
            carry = value >> wp.uint64(32)
        if carry != wp.uint64(0):
            if r.count >= 48:
                r.valid = 0
                return r
            r.words[r.count] = wp.uint32(carry)
            r.count += 1
        return r
    larger = compare(a, b)
    if larger == 0:
        return r
    left, right = a, b
    if larger < 0:
        left, right = b, a
    r.sign = left.sign
    r.count = left.count
    borrow = wp.uint64(0)
    for i in range(r.count):
        x = wp.uint64(left.words[i])
        y = wp.uint64(right.words[i]) + borrow
        r.words[i] = wp.uint32((x - y) & wp.uint64(0xFFFFFFFF))
        borrow = wp.uint64(x < y)
    return trim(r)


@wp.func
def minus(a: Big, b: Big):
    return plus(a, neg(b))


@wp.func(inline=False)
def times(a: Big, b: Big):
    r = empty()
    r.valid = a.valid & b.valid
    if a.sign == 0 or b.sign == 0 or r.valid == 0:
        return r
    r.sign = a.sign * b.sign
    r.count = wp.min(48, a.count + b.count)
    for i in range(a.count):
        carry = wp.uint64(0)
        for j in range(b.count):
            k = i + j
            if k >= 48:
                r.valid = 0
                return r
            value = wp.uint64(a.words[i]) * wp.uint64(b.words[j]) + wp.uint64(r.words[k]) + carry
            r.words[k] = wp.uint32(value & wp.uint64(0xFFFFFFFF))
            carry = value >> wp.uint64(32)
        if carry != wp.uint64(0):
            if i + b.count >= 48:
                r.valid = 0
                return r
            r.words[i + b.count] = wp.uint32(carry)
    return trim(r)


@wp.struct
class Row:
    x: Big
    y: Big
    z: Big
    w: Big


@wp.func
def relative(a: wp.vec3, b: wp.vec3, base: int):
    r = Row()
    r.x = minus(from_float(a[0], base), from_float(b[0], base))
    r.y = minus(from_float(a[1], base), from_float(b[1], base))
    r.z = minus(from_float(a[2], base), from_float(b[2], base))
    r.w = empty()
    return r


@wp.func
def norm(r: Row):
    return plus(plus(times(r.x, r.x), times(r.y, r.y)), times(r.z, r.z))


@wp.func(inline=False)
def det3(a: Row, b: Row, c: Row):
    x = times(a.x, minus(times(b.y, c.z), times(b.z, c.y)))
    y = times(a.y, minus(times(b.z, c.x), times(b.x, c.z)))
    z = times(a.z, minus(times(b.x, c.y), times(b.y, c.x)))
    return plus(plus(x, y), z)


@wp.func(inline=False)
def det4(a: Row, b: Row, c: Row, d: Row):
    pos = plus(times(b.w, det3(a, c, d)), times(d.w, det3(a, b, c)))
    negated = plus(times(a.w, det3(b, c, d)), times(c.w, det3(a, b, d)))
    return minus(pos, negated)


@wp.func
def power2(exponent: int):
    value = wp.float64(1.0)
    powers = Powers
    magnitude = wp.abs(exponent)
    for i in range(10):
        if (magnitude & (1 << i)) != 0:
            if exponent < 0:
                value /= powers[i]
            else:
                value *= powers[i]
    return value


@wp.func
def predicate(a: Big, exponent: int):
    r = Predicate()
    r.valid = a.valid
    r.sign = a.sign
    if a.valid == 0:
        r.sign = 0
    if r.sign != 0:
        low = wp.max(0, a.count - 3)
        value = wp.float64(0.0)
        for i in range(a.count - 1, low - 1, -1):
            value = value * wp.float64(4294967296.0) + wp.float64(a.words[i])
        r.estimate = wp.float64(a.sign) * value * power2(32 * low + exponent)
    return r


@wp.func
def vector_base(p: wp.vec3):
    return wp.min(base_for(p[0], 1), wp.min(base_for(p[1], 1), base_for(p[2], 1)))


@wp.func
def lifted(a: wp.vec3, q: wp.vec3, wa: float, wq: float, base: int):
    r = relative(a, q, base)
    r.w = plus(minus(norm(r), from_float(wa, 2 * base)), from_float(wq, 2 * base))
    return r
