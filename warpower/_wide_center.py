"""Wide circumcenter fallback for uncertain physical extraction geometry.

Unlike generated insertion sites, extraction centers need not fit float32.
Integer numerators and denominators are estimated in their physical float64
scale before division. This is a stable metric estimate, not an exact rational
surface query or a correctly rounded float64 division guarantee.
"""

import warp as wp

from . import _integer as integer
from ._stable_center import cofactor, shift_left
from ._stable_volume import ERROR_SCALE

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.struct
class Geometry:
    determinant: wp.float64
    center: wp.vec3d
    offset: wp.vec3d
    radius: wp.float64
    area2: wp.float64
    valid: int


@wp.func
def uncertain(a: wp.vec3d, b: wp.vec3d, c: wp.vec3d, value: wp.float64):
    permanent = (
        wp.abs(a[0]) * (wp.abs(b[1] * c[2]) + wp.abs(b[2] * c[1]))
        + wp.abs(a[1]) * (wp.abs(b[2] * c[0]) + wp.abs(b[0] * c[2]))
        + wp.abs(a[2]) * (wp.abs(b[0] * c[1]) + wp.abs(b[1] * c[0]))
    )
    bound = ERROR_SCALE * permanent
    return not wp.isfinite(value) or not wp.isfinite(bound) or wp.abs(value) <= bound


@wp.func(inline=False)
def exact_geometry(p0: wp.vec3, p1: wp.vec3, p2: wp.vec3, p3: wp.vec3):
    result = Geometry()
    base = wp.min(
        wp.min(integer.vector_base(p0), integer.vector_base(p1)),
        wp.min(integer.vector_base(p2), integer.vector_base(p3)),
    )
    a, b, c = (
        integer.relative(p1, p0, base),
        integer.relative(p2, p0, base),
        integer.relative(p3, p0, base),
    )
    det = integer.det3(a, b, c)
    physical = integer.predicate(det, 3 * base)
    if physical.valid == 0 or physical.sign != 1:
        return result
    result.determinant = physical.estimate
    denominator = shift_left(det, 1)
    d = integer.predicate(denominator, 3 * base)
    if d.valid == 0:
        return result
    norms = integer.Row()
    norms.x = integer.norm(a)
    norms.y = integer.norm(b)
    norms.z = integer.norm(c)
    offset = wp.vec3d()
    normal0, normal1, normal2, normal3 = wp.vec3d(), wp.vec3d(), wp.vec3d(), wp.vec3d()
    for axis in range(3):
        bc, ca, ab = cofactor(b, c, axis), cofactor(c, a, axis), cofactor(a, b, axis)
        bc_value = integer.predicate(bc, 2 * base)
        ca_value = integer.predicate(ca, 2 * base)
        ab_value = integer.predicate(ab, 2 * base)
        other_value = integer.predicate(integer.plus(integer.plus(bc, ca), ab), 2 * base)
        if (
            bc_value.valid == 0
            or ca_value.valid == 0
            or ab_value.valid == 0
            or other_value.valid == 0
        ):
            return result
        normal0[axis] = bc_value.estimate
        normal1[axis] = ca_value.estimate
        normal2[axis] = ab_value.estimate
        normal3[axis] = other_value.estimate
        relative_n = integer.plus(
            integer.plus(
                integer.times(norms.x, bc),
                integer.times(norms.y, ca),
            ),
            integer.times(norms.z, ab),
        )
        n = integer.plus(integer.times(denominator, integer.from_float(p0[axis], base)), relative_n)
        global_value = integer.predicate(n, 4 * base)
        local_value = integer.predicate(relative_n, 4 * base)
        if global_value.valid == 0 or local_value.valid == 0:
            return result
        result.center[axis] = global_value.estimate / d.estimate
        offset[axis] = local_value.estimate / d.estimate
    # Subtracting a rounded global center loses radius accuracy far from zero.
    result.offset = offset
    result.radius = wp.length(offset)
    result.area2 = wp.length(normal0) + wp.length(normal1) + wp.length(normal2) + wp.length(normal3)
    result.valid = int(
        wp.isfinite(result.center[0])
        and wp.isfinite(result.center[1])
        and wp.isfinite(result.center[2])
        and wp.isfinite(result.radius)
        and result.radius > wp.float64(0.0)
        and wp.isfinite(result.area2)
        and result.area2 > wp.float64(0.0)
    )
    return result
