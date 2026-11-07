"""Current-input Morton groups for private VTM seed certification.

Every invocation refreshes exact float32 point-group boxes. A strict sphere-box
separation can skip a complete group. Surviving points use a bounded binary32
first stage before the original strict Ball64 test and complete symbolic
fallback. No temporal proof is stored.
"""

import warp as wp

from . import _bounded_predicates as low
from . import _sphere_bounds as sb
from ._analytic_predicates import strict
from ._bvh import Tree
from ._delaunay_certificate import Sphere, symbolic_sphere
from ._prepared_seeds import PreparedSeedQueries, point_value, prepare

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.kernel
def group_bounds(
    points: wp.array[wp.vec3], tree: Tree, lower: wp.array[wp.vec3], upper: wp.array[wp.vec3]
):
    # The current hierarchy supplies a permutation of every input slot. Including
    # noncanonical duplicates in a group only makes its box more conservative.
    group = wp.tid()
    first = group * 32
    lo = wp.vec3(points[tree.ids[first]])
    hi = wp.vec3(lo)
    for j in range(first + 1, wp.min(first + 32, points.shape[0])):
        p = points[tree.ids[j]]
        lo = wp.min(lo, p)
        hi = wp.max(hi, p)
    lower[group] = lo
    upper[group] = hi


@wp.kernel
def check_groups(
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    count: wp.array[int],
    tree: Tree,
    live: wp.array[int],
    error: wp.array[int],
    shapes: wp.array[Sphere],
    boxes: wp.array[sb.Box],
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
):
    tid = wp.tid()
    i, lane = tid // 32, tid % 32
    if error[0] != 0 or i >= count[0]:
        return
    ids = wp.vec4i(tets[i, 0], tets[i, 1], tets[i, 2], tets[i, 3])
    a, b, c, d = points[ids[0]], points[ids[1]], points[ids[2]], points[ids[3]]
    shape = shapes[i]
    box = boxes[i]
    for group in range(lower.shape[0]):
        # All 32 lanes see the same group box. An invalid sphere enclosure
        # cannot skip any group; all surviving queries retain the exact policy.
        if sb.outside(box, lower[group], upper[group]):
            continue
        rank = 32 * group + lane
        if rank >= points.shape[0]:
            continue
        q = tree.ids[rank]
        if tree.owners[q] != q or q == ids[0] or q == ids[1] or q == ids[2] or q == ids[3]:
            continue
        p = points[q]
        if sb.outside(box, p, p):
            continue
        fast = low.sphere_filter(a, b, c, d, p)
        if strict(fast):
            if fast.value > wp.float64(0.0):
                wp.atomic_and(live, i, 0)
            continue
        value = point_value(shape, p)
        if (
            wp.isfinite(value.value)
            and wp.isfinite(value.error)
            and wp.abs(value.value) > value.error
        ):
            if value.value > wp.float64(0.0):
                wp.atomic_and(live, i, 0)
            continue
        result = symbolic_sphere(points, ids, q)
        if result.valid == 0:
            wp.atomic_or(error, 0, 2)
            wp.atomic_and(live, i, 0)
        elif result.sign >= 0:
            wp.atomic_and(live, i, 0)


class GroupedSeedQueries(PreparedSeedQueries):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        groups = (len(self._check_inputs[0]) + 31) // 32
        self.lower = wp.empty(groups, dtype=wp.vec3, device=self.device)
        self.upper = wp.empty_like(self.lower)
        self._group_inputs = [self._check_inputs[0], self._check_inputs[3], self.lower, self.upper]
        self._query_inputs = [*self._check_inputs, self.lower, self.upper]
        wp.load_module(module=__name__, device=self.device, block_dim=self.block_dim)

    def run(self):
        wp.launch(
            group_bounds,
            len(self.lower),
            inputs=self._group_inputs,
            block_dim=self.block_dim,
            device=self.device,
        )
        wp.launch(
            prepare,
            self.capacity,
            inputs=self._prepare_inputs,
            block_dim=self.block_dim,
            device=self.device,
        )
        wp.launch(
            check_groups,
            32 * self.capacity,
            inputs=self._query_inputs,
            block_dim=self.block_dim,
            device=self.device,
        )
