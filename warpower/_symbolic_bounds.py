"""Strict geometric rejection for the unweighted limit of a symbolic cell.

Normals are unperturbed; symbolic weights change only halfspace constants.
A strict separating gap at every limiting vertex persists for sufficiently
small positive infinitesimals. Uncertain bounds retain the initial box.
"""

import warp as wp

from . import _filtered_predicates as fp
from . import _halfspace_predicates as hp
from ._cell import Scratch64, plane_at
from ._delaunay_certificate import cofactor
from ._sphere_bounds import MIN64, quotient
from ._symbolic_cell import SymbolicCell

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.struct
class Box:
    lower: wp.vec3d
    upper: wp.vec3d


@wp.func
def lower(value: fp.Ball):
    pad = wp.float64(16.0) * fp.EPS * (wp.abs(value.value) + value.error)
    pad += wp.float64(16.0) * MIN64
    return value.value - value.error - pad


@wp.func
def upper(value: fp.Ball):
    pad = wp.float64(16.0) * fp.EPS * (wp.abs(value.value) + value.error)
    pad += wp.float64(16.0) * MIN64
    return value.value + value.error + pad


@wp.func
def vertex_box(points: wp.array[wp.vec3], bounds: wp.array[wp.vec3], owner: int, ids: wp.vec3i):
    result = Box()
    result.lower = wp.vec3d(bounds[0])
    result.upper = wp.vec3d(bounds[1])
    a = hp.row(points, bounds, owner, ids[0], 0, -1)
    b = hp.row(points, bounds, owner, ids[1], 0, -1)
    c = hp.row(points, bounds, owner, ids[2], 0, -1)
    determinant = hp.normal_det(a, b, c)
    nx = cofactor(a.w, b.w, c.w, a.y, a.z, b.y, b.z, c.y, c.z)
    ny = cofactor(a.w, b.w, c.w, a.z, a.x, b.z, b.x, c.z, c.x)
    nz = cofactor(a.w, b.w, c.w, a.x, a.y, b.x, b.y, c.x, c.y)
    # Cramer's RHS is -w. Negation itself is exact. Normalize determinant
    # orientation for the positive-denominator quotient enclosure.
    sign = wp.where(determinant.value < wp.float64(0.0), wp.float64(1.0), wp.float64(-1.0))
    determinant.value = wp.abs(determinant.value)
    nx.value *= sign
    ny.value *= sign
    nz.value *= sign
    x, y, z = quotient(nx, determinant), quotient(ny, determinant), quotient(nz, determinant)
    values = wp.vec3d(x.value, y.value, z.value)
    errors = wp.vec3d(x.error, y.error, z.error)
    for axis in range(3):
        value = fp.Ball()
        value.value = values[axis]
        value.error = errors[axis]
        lo, hi = lower(value), upper(value)
        if wp.isfinite(lo) and wp.isfinite(hi):
            lo = wp.max(lo, result.lower[axis])
            hi = wp.min(hi, result.upper[axis])
            if lo <= hi:
                result.lower[axis] = lo
                result.upper[axis] = hi
    return result


@wp.func(inline=False)
def update(
    scratch: Scratch64,
    cell: SymbolicCell,
    points: wp.array[wp.vec3],
    bounds: wp.array[wp.vec3],
    owner: int,
):
    lo, hi = wp.vec3d(bounds[1]), wp.vec3d(bounds[0])
    for k in range(cell.nv):
        v = scratch.vertices[cell.base + 32 * k]
        ids = wp.vec3i(0)
        for j in range(3):
            ids[j] = scratch.neighbors[cell.base + 32 * plane_at(v, j)]
        box = vertex_box(points, bounds, owner, ids)
        lo = wp.min(lo, box.lower)
        hi = wp.max(hi, box.upper)
    cell.lo = lo
    cell.hi = hi
    return cell


@wp.func
def quadratic(x: wp.float64, delta: wp.float64):
    a, b = fp.scalar(x), fp.scalar(delta)
    return fp.minus(fp.times(fp.scalar(wp.float64(2.0)), fp.times(a, b)), fp.times(b, b))


@wp.func(inline=False)
def outside(lo: wp.vec3d, hi: wp.vec3d, point: wp.vec3, node_lo: wp.vec3, node_hi: wp.vec3):
    total = fp.scalar(wp.float64(0.0))
    for axis in range(3):
        p = fp.scalar(wp.float64(point[axis]))
        xl = lower(fp.minus(fp.scalar(lo[axis]), p))
        xh = upper(fp.minus(fp.scalar(hi[axis]), p))
        dl = lower(fp.difference(node_lo[axis], point[axis]))
        dh = upper(fp.difference(node_hi[axis], point[axis]))
        if (
            not wp.isfinite(xl)
            or not wp.isfinite(xh)
            or not wp.isfinite(dl)
            or not wp.isfinite(dh)
            or xl > xh
            or dl > dh
        ):
            return False
        # The maximum over x is at one of its interval endpoints. For fixed
        # x, the concave quadratic in delta peaks at clamp(x, dl, dh).
        left = upper(quadratic(xl, wp.clamp(xl, dl, dh)))
        right = upper(quadratic(xh, wp.clamp(xh, dl, dh)))
        total = fp.plus(total, fp.scalar(wp.max(left, right)))
    bound = upper(total)
    return wp.isfinite(bound) and bound < wp.float64(0.0)
