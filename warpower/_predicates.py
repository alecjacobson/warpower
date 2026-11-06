"""Exact signs and float64 estimates for finite float32 geometric predicates.

The existing interval filters call these bounded integer fallbacks only when
uncertain. Original-coordinate differences and products are formed exactly;
nonfinite inputs and capacity failures remain explicit.
"""

import warp as wp

from ._integer import Predicate as Predicate
from ._integer import base_for, det3, det4, lifted, neg, predicate, relative, vector_base

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.func(inline=False)
def orient3d(a: wp.vec3, b: wp.vec3, c: wp.vec3, d: wp.vec3):
    base = wp.min(wp.min(vector_base(a), vector_base(b)), wp.min(vector_base(c), vector_base(d)))
    return predicate(
        det3(relative(b, a, base), relative(c, a, base), relative(d, a, base)), 3 * base
    )


@wp.func(inline=False)
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
    base = wp.min(wp.min(vector_base(a), vector_base(b)), wp.min(vector_base(c), vector_base(d)))
    base = wp.min(base, vector_base(q))
    base = wp.min(
        base,
        wp.min(wp.min(base_for(wa, 2), base_for(wb, 2)), wp.min(base_for(wc, 2), base_for(wd, 2))),
    )
    base = wp.min(base, base_for(wq, 2))
    value = det4(
        lifted(a, q, wa, wq, base),
        lifted(b, q, wb, wq, base),
        lifted(c, q, wc, wq, base),
        lifted(d, q, wd, wq, base),
    )
    return predicate(neg(value), 5 * base)


@wp.func(inline=True)
def insphere(a: wp.vec3, b: wp.vec3, c: wp.vec3, d: wp.vec3, q: wp.vec3):
    return power_sphere(a, b, c, d, q, 0.0, 0.0, 0.0, 0.0, 0.0)
