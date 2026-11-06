"""Exact determinant signs for finite float32 sites and additive power weights.

Optional recovery foundation; existing geometry paths do not import this module.
Homogeneous coordinates avoid rounding a translated difference before expansion
arithmetic. Products have degree at most five, within float64 exponent range for
float32 inputs. Capacity/finiteness failures are explicit, never an exact zero.
Calling kernels must use fast_math=False and fuse_fp=False as this module does.
"""

import itertools

import warp as wp

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})

LIMBS = wp.constant(64)
Coefficients = wp.types.vector(64, wp.float64)
Positions4 = wp.types.matrix((4, 3), wp.float64)
Positions5 = wp.types.matrix((5, 3), wp.float64)
Weights5 = wp.types.vector(5, wp.float64)


def _permutations(n, bits):
    encoded = []
    for p in itertools.permutations(range(n)):
        odd = sum(p[i] > p[j] for i in range(n) for j in range(i + 1, n)) % 2
        # The last determinant column is constant one; its row index is unused.
        encoded.append(sum(p[i] << (bits * i) for i in range(n - 1)) | (odd << (bits * (n - 1))))
    return wp.constant(wp.types.vector(len(encoded), wp.int32)(encoded))


ORIENTATION_TERMS = _permutations(4, 2)
POWER_TERMS = _permutations(5, 3)


@wp.struct
class Expansion:
    values: Coefficients
    count: int
    valid: int


@wp.struct
class Predicate:
    sign: int
    estimate: wp.float64
    valid: int


@wp.func(inline=True)
def empty_expansion():
    result = Expansion()
    result.count = 0
    result.valid = 1
    return result


@wp.func(inline=True)
def compress(source: Expansion):
    """Two-pass expansion compression, retaining the exact sum."""
    result = empty_expansion()
    result.valid = source.valid
    if source.count == 0:
        return result
    work = source
    q = source.values[source.count - 1]
    bottom = source.count - 1
    i = source.count - 2
    while i >= 0:
        a = source.values[i]
        total = q + a
        error = a - (total - q)
        if error != wp.float64(0.0):
            work.values[bottom] = total
            bottom -= 1
            q = error
        else:
            q = total
        i -= 1
    for j in range(bottom + 1, source.count):
        a = work.values[j]
        total = a + q
        error = q - (total - a)
        if error != wp.float64(0.0):
            result.values[result.count] = error
            result.count += 1
        q = total
    if q != wp.float64(0.0):
        result.values[result.count] = q
        result.count += 1
    return result


@wp.func(inline=True)
def grow(source: Expansion, value: wp.float64):
    """Error-free TwoSum, accumulating nonzero low components in order."""
    result = empty_expansion()
    result.valid = source.valid
    q = value
    for i in range(source.count):
        a = source.values[i]
        total = q + a
        b_virtual = total - q
        a_virtual = total - b_virtual
        error = (q - a_virtual) + (a - b_virtual)
        if error != wp.float64(0.0):
            if result.count == LIMBS:
                result.valid = 0
                return result
            result.values[result.count] = error
            result.count += 1
        q = total
    if not wp.isfinite(q):
        result.valid = 0
    if q != wp.float64(0.0):
        if result.count == LIMBS:
            result.valid = 0
            return result
        result.values[result.count] = q
        result.count += 1
    if result.count >= 32:
        return compress(result)
    return result


@wp.func(inline=True)
def scale(source: Expansion, value: wp.float64):
    result = empty_expansion()
    splitter = wp.float64(134217729.0)
    split_b = splitter * value
    b_hi = split_b - (split_b - value)
    b_lo = value - b_hi
    for i in range(source.count):
        a = source.values[i]
        split_a = splitter * a
        a_hi = split_a - (split_a - a)
        a_lo = a - a_hi
        product = a * value
        error1 = product - a_hi * b_hi
        error2 = error1 - a_lo * b_hi
        error3 = error2 - a_hi * b_lo
        error = a_lo * b_lo - error3
        result = grow(result, error)
        result = grow(result, product)
    result.valid = result.valid & source.valid
    return result


@wp.func(inline=True)
def add(left: Expansion, right: Expansion, sign: wp.float64):
    result = left
    for i in range(right.count):
        result = grow(result, sign * right.values[i])
    result.valid = result.valid & right.valid
    return result


@wp.func(inline=True)
def negative_determinant(value: Expansion):
    result = Predicate()
    result.valid = value.valid
    result.sign = 0
    result.estimate = wp.float64(0.0)
    if value.valid != 0 and value.count > 0:
        result.estimate = -value.values[value.count - 1]
        result.sign = wp.where(result.estimate > wp.float64(0.0), 1, -1)
    return result


@wp.func(inline=True)
def orient3d(a: wp.vec3, b: wp.vec3, c: wp.vec3, d: wp.vec3):
    """Sign of det(b-a,c-a,d-a); zero means exactly coplanar float32 sites."""
    p = Positions4(
        wp.float64(a[0]),
        wp.float64(a[1]),
        wp.float64(a[2]),
        wp.float64(b[0]),
        wp.float64(b[1]),
        wp.float64(b[2]),
        wp.float64(c[0]),
        wp.float64(c[1]),
        wp.float64(c[2]),
        wp.float64(d[0]),
        wp.float64(d[1]),
        wp.float64(d[2]),
    )
    total = empty_expansion()
    for i in range(4):
        for j in range(3):
            if not wp.isfinite(p[i, j]):
                total.valid = 0
                return negative_determinant(total)
    for i in range(24):
        code = ORIENTATION_TERMS[i]
        term = grow(empty_expansion(), p[code & 3, 0])
        term = scale(term, p[(code >> 2) & 3, 1])
        term = scale(term, p[(code >> 4) & 3, 2])
        sign = wp.float64(1 - 2 * ((code >> 6) & 1))
        total = add(total, term, sign)
    return negative_determinant(total)


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
    """Positive means q lies inside a positively oriented tet's power sphere.

    Swapping two defining vertices reverses the sign. A zero is an exact power
    tie, not a tolerance band. The lifted coordinate is x*x+y*y+z*z-weight.
    """
    p = Positions5(
        wp.float64(a[0]),
        wp.float64(a[1]),
        wp.float64(a[2]),
        wp.float64(b[0]),
        wp.float64(b[1]),
        wp.float64(b[2]),
        wp.float64(c[0]),
        wp.float64(c[1]),
        wp.float64(c[2]),
        wp.float64(d[0]),
        wp.float64(d[1]),
        wp.float64(d[2]),
        wp.float64(q[0]),
        wp.float64(q[1]),
        wp.float64(q[2]),
    )
    weights = Weights5(
        wp.float64(wa), wp.float64(wb), wp.float64(wc), wp.float64(wd), wp.float64(wq)
    )
    total = empty_expansion()
    for i in range(5):
        if not wp.isfinite(weights[i]):
            total.valid = 0
            return negative_determinant(total)
        for j in range(3):
            if not wp.isfinite(p[i, j]):
                total.valid = 0
                return negative_determinant(total)
    for i in range(120):
        code = POWER_TERMS[i]
        lift_row = (code >> 9) & 7
        term = grow(empty_expansion(), -weights[lift_row])
        for j in range(3):
            # A product of two finite float32 values is exact in float64.
            term = grow(term, p[lift_row, j] * p[lift_row, j])
        term = scale(term, p[code & 7, 0])
        term = scale(term, p[(code >> 3) & 7, 1])
        term = scale(term, p[(code >> 6) & 7, 2])
        sign = wp.float64(1 - 2 * ((code >> 12) & 1))
        total = add(total, term, sign)
    return negative_determinant(total)


@wp.func(inline=True)
def insphere(a: wp.vec3, b: wp.vec3, c: wp.vec3, d: wp.vec3, q: wp.vec3):
    return power_sphere(a, b, c, d, q, 0.0, 0.0, 0.0, 0.0, 0.0)
