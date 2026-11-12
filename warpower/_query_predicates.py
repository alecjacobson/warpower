"""Check deferred clipping decisions against original FP32 site coordinates.

Interior predicates use filtered FP64 arithmetic and exact integer fallback.
Box-plane predicates use interval filters; uncertainty schedules recovery.
The hot clipping screen and subsequent FP64 rebuilding do not constitute a
global exactness certificate for every output edge.
"""

import warp as wp

from . import _filtered_predicates as fp
from . import _retry
from ._bvh import Tree

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.struct
class Plane:
    x: fp.Ball
    y: fp.Ball
    z: fp.Ball
    w: fp.Ball


@wp.func
def normal_det(a: Plane, b: Plane, c: Plane):
    return fp.det3(a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z)


@wp.func
def filtered_det(a: Plane, b: Plane, c: Plane, d: Plane):
    return fp.minus(
        fp.plus(fp.times(b.w, normal_det(a, c, d)), fp.times(d.w, normal_det(a, b, c))),
        fp.plus(fp.times(a.w, normal_det(b, c, d)), fp.times(c.w, normal_det(a, b, d))),
    )


@wp.func
def original_row(
    points: wp.array[wp.vec3],
    weights: wp.array[float],
    lower: wp.vec3,
    upper: wp.vec3,
    i: int,
    j: int,
):
    p = points[i]
    r = Plane()
    r.x = fp.scalar(wp.float64(0.0))
    r.y = fp.scalar(wp.float64(0.0))
    r.z = fp.scalar(wp.float64(0.0))
    r.w = fp.scalar(wp.float64(0.0))
    if j >= 0:
        q = points[j]
        dx, dy, dz = fp.difference(q[0], p[0]), fp.difference(q[1], p[1]), fp.difference(q[2], p[2])
        minus = fp.scalar(wp.float64(-1.0))
        r.x = fp.times(minus, dx)
        r.y = fp.times(minus, dy)
        r.z = fp.times(minus, dz)
        r.w = fp.times(fp.scalar(wp.float64(0.5)), fp.lift(dx, dy, dz, weights[j], weights[i]))
    else:
        k = -j - 1
        axis, side = k // 2, k % 2
        sign = wp.float64(1 - 2 * side)
        if axis == 0:
            r.x = fp.scalar(sign)
        elif axis == 1:
            r.y = fp.scalar(sign)
        else:
            r.z = fp.scalar(sign)
        bound = wp.where(side == 0, lower[axis], upper[axis])
        r.w = fp.times(fp.scalar(-sign), fp.difference(bound, p[axis]))
    return r


@wp.func
def relation(
    points: wp.array[wp.vec3],
    weights: wp.array[float],
    lower: wp.vec3,
    upper: wp.vec3,
    i: int,
    ids: wp.vec4i,
):
    if ids[0] >= 0 and ids[1] >= 0 and ids[2] >= 0:
        pa, pb, pc, pd, pq = (
            points[i],
            points[ids[0]],
            points[ids[1]],
            points[ids[2]],
            points[ids[3]],
        )
        orient = fp.orient3d(pa, pb, pc, pd)
        sphere_result = fp.power_sphere(
            pa,
            pb,
            pc,
            pd,
            pq,
            weights[i],
            weights[ids[0]],
            weights[ids[1]],
            weights[ids[2]],
            weights[ids[3]],
        )
        if orient.valid == 0 or orient.sign == 0 or sphere_result.valid == 0:
            return -1
        return int(orient.sign * sphere_result.sign > 0)
    a = original_row(points, weights, lower, upper, i, ids[0])
    b = original_row(points, weights, lower, upper, i, ids[1])
    c = original_row(points, weights, lower, upper, i, ids[2])
    d = original_row(points, weights, lower, upper, i, ids[3])
    normal = normal_det(a, b, c)
    side = filtered_det(a, b, c, d)
    if not fp.certain(normal) or normal.value == wp.float64(0.0) or not fp.certain(side):
        return -1
    return int(
        (side.value > wp.float64(0.0) and normal.value < wp.float64(0.0))
        or (side.value < wp.float64(0.0) and normal.value > wp.float64(0.0))
    )


@wp.kernel
def check_queries(
    points: wp.array[wp.vec3],
    weights: wp.array[float],
    tree: Tree,
    queries: wp.array[wp.vec4i],
    owners: wp.array[int],
    count: wp.array[int],
    status: wp.array[int],
    neighbors: wp.array2d[int],
    counts: wp.array[int],
    source: wp.array[int],
    by_job: bool,
):
    k = wp.tid()
    if k >= count[0]:
        return
    encoded = owners[k]
    i, expected = encoded >> 1, encoded & 1
    row = i
    if by_job:
        row = source[i]
    previous = status[row]
    if previous != 0 and previous != 3:
        return
    observed = relation(
        points,
        weights,
        tree.lower[1] - wp.vec3(1.00001),
        tree.upper[1] + wp.vec3(1.00001),
        i,
        queries[k],
    )
    if observed != expected:
        if wp.atomic_cas(status, row, previous, 6) == previous:
            for j in range(counts[row], neighbors.shape[1]):
                neighbors[row, j] = -1
            counts[row] = 0


@wp.kernel(enable_backward=False)
def collect_precision(
    status: wp.array[int],
    jobs: wp.array[int],
    count: wp.array[int],
    states: wp.array[wp.vec2i],
    initial_capacity: int,
    initialize: bool,
):
    i = wp.tid()
    state = wp.vec2i(0, initial_capacity)
    if initialize:
        state = _retry.after_attempt(state, status[i], initial_capacity, False, False, True)
    else:
        state = _retry.promote(states[i], status[i])
    states[i] = state
    if _retry.cooperative_eligible(state, status[i]):
        k = wp.atomic_add(count, 0, 1)
        jobs[k] = i
        # A second counter records bounded cooperative attempts for eager diagnostics.
        if count.shape[0] > 1 and k < 512:
            wp.atomic_add(count, 1, 1)


@wp.kernel(enable_backward=False)
def collect_precision_retry(
    status: wp.array[int],
    owners: wp.array[int],
    jobs: wp.array[int],
    count: wp.array[int],
    states: wp.array[wp.vec2i],
):
    row = wp.tid()
    owner = owners[row]
    state = _retry.promote(states[owner], status[row])
    states[owner] = state
    if _retry.cooperative_eligible(state, status[row]):
        k = wp.atomic_add(count, 0, 1)
        if k < jobs.shape[0]:
            jobs[k] = owners[row]
            if count.shape[0] > 1:
                wp.atomic_add(count, 1, 1)
