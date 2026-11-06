"""Physical determinants with exact fallback for uncertain float32 tetrahedra.

The fast expression is supplied by the caller, preserving its normal arithmetic
and FMA policy. The permanent bounds six triple products. A margin of 64 binary64
unit roundoffs covers coordinate differences, products, and sums, including a
conservatively rounded permanent. Finite float32 inputs keep these degree-three
terms and the bound normal and finite in binary64. FMA only removes roundings.

Uncertain cells use integer arithmetic, not the interval predicates (whose
callers require fuse_fp=False). The physical determinant estimate is normal and
finite for nonzero float32 input determinants; its sign is exact.
"""

import warp as wp

from . import _predicates as exact

wp.set_module_options({"enable_backward": False, "fast_math": False})
ERROR_SCALE = wp.constant(wp.float64(32.0 * 2.0**-52))


@wp.func(inline=False)
def determinant(
    p0: wp.vec3,
    p1: wp.vec3,
    p2: wp.vec3,
    p3: wp.vec3,
    a: wp.vec3d,
    b: wp.vec3d,
    c: wp.vec3d,
    value: wp.float64,
):
    permanent = (
        wp.abs(a[0]) * (wp.abs(b[1] * c[2]) + wp.abs(b[2] * c[1]))
        + wp.abs(a[1]) * (wp.abs(b[2] * c[0]) + wp.abs(b[0] * c[2]))
        + wp.abs(a[2]) * (wp.abs(b[0] * c[1]) + wp.abs(b[1] * c[0]))
    )
    bound = ERROR_SCALE * permanent
    if wp.isfinite(value) and wp.isfinite(bound) and wp.abs(value) > bound:
        return value
    result = exact.orient3d(p0, p1, p2, p3)
    if result.valid == 0:
        return wp.float64(wp.nan)
    return result.estimate
