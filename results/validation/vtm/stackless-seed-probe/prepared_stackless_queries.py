"""Private #177 probe: cooperative stackless traversal of current segment bounds."""

import weakref

import warp as wp

from warpower import _sphere_bounds as sb
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
def check_stackless(
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    count: wp.array[int],
    tree: Tree,
    live: wp.array[int],
    error: wp.array[int],
    shapes: wp.array[Sphere],
    boxes: wp.array[sb.Box],
    segment: Tree,
):
    tid = wp.tid()
    i, lane = tid // 32, tid % 32
    if error[0] != 0 or i >= count[0]:
        return
    ids = wp.vec4i(tets[i, 0], tets[i, 1], tets[i, 2], tets[i, 3])
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
    def __init__(self, *args, segment, **kwargs):
        super().__init__(*args, **kwargs)
        points, _, _, tree = self._check_inputs[:4]
        assert segment.n == len(points) == tree.n
        assert segment.ids.ptr == tree.ids.ptr and segment.leaf_size == 1
        assert segment.leaf_base == 1 << (len(points) - 1).bit_length()
        self.segment = segment
        self._query_inputs = [*self._check_inputs, segment]
        wp.load_module(module=__name__, device=self.device, block_dim=self.block_dim)

    def run(self):
        wp.launch(
            prepare, self.capacity, inputs=self._prepare_inputs,
            block_dim=self.block_dim, device=self.device,
        )
        wp.launch(
            check_stackless, 32 * self.capacity, inputs=self._query_inputs,
            block_dim=self.block_dim, device=self.device,
        )


def install():
    """Private setup routing only; production adoption would pass segment explicitly."""
    from warpower import _grouped_seeds, _site_bvh

    original = _site_bvh.SiteHierarchy.__init__
    registered = weakref.WeakValueDictionary()

    def remember(self, points):
        original(self, points)
        registered[(points.ptr, self.tree.ids.ptr, len(points))] = self

    class RegisteredStacklessQueries(StacklessSeedQueries):
        def __init__(self, *args, **kwargs):
            points, _, _, tree = args[:4]
            hierarchy = registered[(points.ptr, tree.ids.ptr, len(points))]
            assert hierarchy.points.ptr == points.ptr
            super().__init__(*args, segment=hierarchy._segment, **kwargs)

    _site_bvh.SiteHierarchy.__init__ = remember
    _grouped_seeds.GroupedSeedQueries = RegisteredStacklessQueries

