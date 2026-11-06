"""Preallocated site hierarchy for evolving nearest-site assignments."""

import warp as wp
from warp.utils import radix_sort_pairs

from . import _bvh
from ._graph import reset_bounds

wp.set_module_options({"enable_backward": False})


@wp.struct
class SiteHit:
    index: int
    distance2: wp.float64


@wp.func(inline=True)
def nearest_site(
    tree: _bvh.Tree,
    points: wp.array[wp.vec3],
    p: wp.vec3d,
    active: wp.array[int],
    occupied: wp.array[int],
    only_free: bool,
):
    result = SiteHit()
    result.index = -1
    result.distance2 = wp.float64(1.0e300)
    stack = wp.zeros(shape=128, dtype=int)
    stack[0] = 1
    size = int(1)
    while size > 0:
        size -= 1
        node = stack[size]
        delta = wp.max(
            wp.max(wp.vec3d(tree.lower[node]) - p, p - wp.vec3d(tree.upper[node])), wp.vec3d(0.0)
        )
        if wp.dot(delta, delta) > result.distance2:
            continue
        if tree.count[node] > 0:
            for k in range(tree.start[node], tree.start[node] + tree.count[node]):
                i = tree.ids[k]
                if i >= active[0]:
                    continue
                if only_free and occupied[i] >= 0:
                    continue
                distance = wp.length_sq(wp.vec3d(points[i]) - p)
                if distance < result.distance2 or (
                    distance == result.distance2 and i < result.index
                ):
                    result.index = i
                    result.distance2 = distance
        else:
            left, right = int(tree.left[node]), int(tree.right[node])
            dl = wp.max(
                wp.max(wp.vec3d(tree.lower[left]) - p, p - wp.vec3d(tree.upper[left])),
                wp.vec3d(0.0),
            )
            dr = wp.max(
                wp.max(wp.vec3d(tree.lower[right]) - p, p - wp.vec3d(tree.upper[right])),
                wp.vec3d(0.0),
            )
            if wp.dot(dl, dl) < wp.dot(dr, dr):
                left, right = right, left
            stack[size] = left
            stack[size + 1] = right
            size += 2
    return result


class SiteHierarchy:
    """Fixed capacity, rebuilt Morton/radix topology; all storage survives replay."""

    def __init__(self, points):
        self.points, self.device, self.n = points, points.device, len(points)
        if self.n < 1:
            raise ValueError("Site hierarchy needs at least one site")
        with wp.ScopedDevice(self.device):
            self._groups = (self.n + 255) // 256
            self._lo = wp.empty(self._groups, dtype=wp.vec3)
            self._hi = wp.empty_like(self._lo)
            self._bounds = wp.empty(2, dtype=wp.vec3)
            self._keys = wp.empty(2 * self.n, dtype=wp.uint64)
            self._ids = wp.empty(2 * self.n, dtype=int)
            self._weights = wp.zeros(self.n, dtype=float)
            self._segment = self._tree(1 << (self.n - 1).bit_length(), 1)
            self.tree = self._tree(self.n, 4)
            # Initialize outside capture. During capture callers rebuild in a
            # validated child graph so sort scratch is owned by graph setup.
            self.rebuild()
            wp.synchronize_device(self.device)

    def _tree(self, base, leaf_size):
        tree = _bvh.Tree()
        tree.n, tree.leaf_base, tree.leaf_size = self.n, base, leaf_size
        tree.ids = self._ids
        tree.lower = wp.empty(2 * base, dtype=wp.vec3)
        tree.upper = wp.empty_like(tree.lower)
        tree.weight = wp.empty(2 * base, dtype=float)
        tree.left = wp.empty(2 * base, dtype=int)
        tree.right = wp.empty(2 * base, dtype=int)
        tree.start = wp.empty(2 * base, dtype=int)
        tree.count = wp.empty(2 * base, dtype=int)
        tree.owners = wp.empty(self.n, dtype=int)
        return tree

    def rebuild(self):
        wp.launch(reset_bounds, 1, inputs=[self._bounds], device=self.device)
        wp.launch_tiled(
            _bvh.block_bounds,
            dim=self._groups,
            inputs=[self.points, self._lo, self._hi],
            block_dim=256,
            device=self.device,
        )
        wp.launch(
            _bvh.merge_bounds,
            self._groups,
            inputs=[self._lo, self._hi, self._bounds],
            device=self.device,
        )
        wp.launch(
            _bvh.morton_keys,
            self.n,
            inputs=[self.points, self._bounds, self._keys, self._ids],
            device=self.device,
        )
        radix_sort_pairs(self._keys, self._ids, self.n, end_bit=63)
        wp.launch(
            _bvh.make_leaves,
            self._segment.leaf_base,
            inputs=[self.points, self._weights, self._segment],
            device=self.device,
        )
        level = self._segment.leaf_base // 2
        while level:
            wp.launch(_bvh.merge_level, level, inputs=[self._segment, level], device=self.device)
            level //= 2
        wp.launch(
            _bvh.radix_nodes,
            self.n,
            inputs=[self._keys, self._segment, self.tree],
            device=self.device,
        )
        wp.launch(
            _bvh.radix_leaves,
            self.n,
            inputs=[self.points, self._weights, self.tree],
            device=self.device,
        )
