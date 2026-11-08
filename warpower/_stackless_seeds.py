"""Traverse existing current site bounds in 32-site groups, without query stacks."""

import warp as wp

from warpower import _bounded_predicates as low
from warpower import _sphere_bounds as sb
from warpower._analytic_predicates import strict
from warpower._bvh import Tree
from warpower._delaunay_certificate import Sphere, symbolic_sphere
from warpower._prepared_seeds import PreparedSeedQueries, point_value, prepare

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.func(inline=True)
def successor(node: int):
    current = node
    while current > 1 and (current & 1) != 0:
        current //= 2
    if current == 1:
        return 0
    return current + 1


@wp.kernel
def strict_check(
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    count: wp.array[int],
    tree: Tree,
    live: wp.array[int],
    error: wp.array[int],
    shapes: wp.array[Sphere],
    boxes: wp.array[sb.Box],
    segment: Tree,
    pending: wp.array[wp.uint32],
    unresolved: wp.array2d[int],
):
    tid = wp.tid()
    i, lane = tid // 32, tid % 32
    if error[0] != 0 or i >= count[0]:
        return
    ids = wp.vec4i(tets[i, 0], tets[i, 1], tets[i, 2], tets[i, 3])
    a, b, c, d = points[ids[0]], points[ids[1]], points[ids[2]], points[ids[3]]
    shape = shapes[i]
    box = boxes[i]
    group_base = wp.max(1, segment.leaf_base // 32)
    first = int(-1)
    node = int(1)
    visits = int(0)
    # Every lane follows the same topology. The full binary tree has at most
    # 2*group_base-1 visited nodes, including groups; no per-lane stack exists.
    while node > 0 and visits < 2 * group_base - 1:
        current = node
        visits += 1
        if sb.outside(box, segment.lower[current], segment.upper[current]):
            node = successor(current)
            continue
        if current < group_base:
            node = 2 * current
            continue
        node = successor(current)
        rank = 32 * (current - group_base) + lane
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
        if first == -1:
            first = q
        else:
            first = -2
    if node != 0:
        wp.atomic_or(error, 0, 16)
        wp.atomic_and(live, i, 0)
    if first != -1:
        unresolved[i, lane] = first
        wp.atomic_or(pending, i, wp.uint32(1) << wp.uint32(lane))


@wp.kernel
def exact_check(
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    count: wp.array[int],
    tree: Tree,
    live: wp.array[int],
    error: wp.array[int],
    shapes: wp.array[Sphere],
    boxes: wp.array[sb.Box],
    segment: Tree,
    pending: wp.array[wp.uint32],
    unresolved: wp.array2d[int],
):
    tid = wp.tid()
    i, lane = tid // 32, tid % 32
    if error[0] != 0 or i >= count[0]:
        return
    if (pending[i] & (wp.uint32(1) << wp.uint32(lane))) == wp.uint32(0):
        return
    ids = wp.vec4i(tets[i, 0], tets[i, 1], tets[i, 2], tets[i, 3])
    saved = unresolved[i, lane]
    if saved >= 0:
        if saved >= points.shape[0]:
            wp.atomic_or(error, 0, 16)
            wp.atomic_and(live, i, 0)
            return
        # This lane already checked the current owner/incidence/box and both
        # strict filters. No input changes between these ordered passes.
        result = symbolic_sphere(points, ids, saved)
        if result.valid == 0:
            wp.atomic_or(error, 0, 2)
            wp.atomic_and(live, i, 0)
        elif result.sign >= 0:
            wp.atomic_and(live, i, 0)
        return
    if saved != -2:
        wp.atomic_or(error, 0, 16)
        wp.atomic_and(live, i, 0)
        return
    a, b, c, d = points[ids[0]], points[ids[1]], points[ids[2]], points[ids[3]]
    shape = shapes[i]
    box = boxes[i]
    group_base = wp.max(1, segment.leaf_base // 32)
    node = int(1)
    visits = int(0)
    # Every lane follows the same topology. The full binary tree has at most
    # 2*group_base-1 visited nodes, including groups; no per-lane stack exists.
    while node > 0 and visits < 2 * group_base - 1:
        current = node
        visits += 1
        if sb.outside(box, segment.lower[current], segment.upper[current]):
            node = successor(current)
            continue
        if current < group_base:
            node = 2 * current
            continue
        node = successor(current)
        rank = 32 * (current - group_base) + lane
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
    if node != 0:
        wp.atomic_or(error, 0, 16)
        wp.atomic_and(live, i, 0)


class StacklessSeedQueries(PreparedSeedQueries):
    """Requires the caller to rebuild the shared segment from current points."""

    def __init__(self, *args, segment, **kwargs):
        super().__init__(*args, **kwargs)
        points, _, _, tree = self._check_inputs[:4]
        if (
            segment.n != len(points)
            or tree.n != len(points)
            or segment.ids.ptr != tree.ids.ptr
            or segment.leaf_size != 1
            or segment.leaf_base != 1 << (len(points) - 1).bit_length()
        ):
            raise ValueError("Query segment must share the current point permutation")
        self.segment = segment
        # Only flagged lanes read these current-invocation indices. The first
        # unresolved point is cached; -2 means the lane needs a complete rescan.
        self.pending = wp.zeros(self.capacity, dtype=wp.uint32, device=self.device)
        self.unresolved = wp.empty((self.capacity, 32), dtype=int, device=self.device)
        self._query_inputs = [*self._check_inputs, segment, self.pending, self.unresolved]
        wp.load_module(module=__name__, device=self.device, block_dim=self.block_dim)

    def run(self):
        self.pending.zero_()
        wp.launch(
            prepare,
            self.capacity,
            inputs=self._prepare_inputs,
            block_dim=self.block_dim,
            device=self.device,
        )
        wp.launch(
            strict_check,
            32 * self.capacity,
            inputs=self._query_inputs,
            block_dim=self.block_dim,
            device=self.device,
        )
        wp.launch(
            exact_check,
            32 * self.capacity,
            inputs=self._query_inputs,
            block_dim=self.block_dim,
            device=self.device,
        )
