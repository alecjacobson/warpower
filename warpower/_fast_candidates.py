"""Untrusted float32 interiors with float64 cell repair for certified meshing.

Only DelaunayWorkspace uses this private subclass. Acceptance still requires
its exact original-coordinate certificate; the original builder remains the
next bounded fallback. Standalone diagram/regular-mesh kernels are unchanged.
"""

from functools import lru_cache
from types import SimpleNamespace

import warp as wp

from . import _bvh
from ._cell import Cell, Scratch, allocate_scratch, clip, init_cell, plane_at
from ._construct import node_may_clip
from ._graph import DiagramWorkspace
from ._tetrahedra import sorted4

wp.set_module_options({"enable_backward": False, "fast_math": False})


@wp.func
def box_distance2(tree: _bvh.Tree, node: int, p: wp.vec3):
    delta = wp.max(wp.max(tree.lower[node] - p, p - tree.upper[node]), wp.vec3(0.0))
    return wp.dot(delta, delta)


@lru_cache(maxsize=None)
def make_cells(queue_capacity=128):
    queue_size = wp.constant(queue_capacity)

    @wp.kernel(enable_backward=False, module="unique", module_options={"fast_math": False})
    def build_cells(
        points: wp.array[wp.vec3],
        weights: wp.array[float],
        tree: _bvh.Tree,
        jobs: wp.array[int],
        first: wp.array[int],
        job_count: wp.array[int],
        bounds: wp.array[wp.vec3],
        scratch: Scratch,
        status: wp.array[int],
        candidates: wp.array2d[int],
        candidate_count: wp.array[int],
    ):
        slot = wp.tid()
        job = first[0] + slot
        if job >= job_count[0]:
            return
        i = jobs[job]
        if status[i] == 0 or status[i] == 3:
            return
        if tree.owners[i] != i:
            status[i] = 3
            return
        p = wp.vec3(points[i])
        wi = float(weights[i])
        lo, hi = tree.lower[1], tree.upper[1]
        span = wp.max(hi[0] - lo[0], wp.max(hi[1] - lo[1], hi[2] - lo[2]))
        cell = init_cell(scratch, slot, points[i], lo - wp.vec3(span), hi + wp.vec3(span), Cell())
        nodes = wp.zeros(shape=queue_size, dtype=int)
        distances = wp.zeros(shape=queue_size, dtype=float)
        nodes[0] = 1
        distances[0] = float(0.0)
        size = int(1)
        while size > 0 and cell.status == 0:
            nearest = int(0)
            for j in range(1, size):
                if distances[j] < distances[nearest]:
                    nearest = j
            node = nodes[nearest]
            size -= 1
            nodes[nearest] = nodes[size]
            distances[nearest] = distances[size]
            if not node_may_clip(tree, node, p, wi, scratch, cell, False):
                continue
            if tree.count[node] > 0:
                for k in range(tree.start[node], tree.start[node] + tree.count[node]):
                    j = tree.ids[k]
                    if tree.owners[j] == j:
                        cell = clip(
                            scratch,
                            cell,
                            i,
                            j,
                            wp.vec3(points[j]) - p,
                            weights[i],
                            weights[j],
                            False,
                        )
                        if cell.status != 0:
                            break
            else:
                for side in range(2):
                    child = tree.left[node]
                    if side == 1:
                        child = tree.right[node]
                    if size == queue_size:
                        cell.status = 4
                        break
                    nodes[size] = child
                    distances[size] = box_distance2(tree, child, p)
                    size += 1
        status[i] = cell.status
        if cell.status != 0:
            return
        # Interior cells only: all clipped/unbounded cells retry with float64.
        for k in range(cell.nv):
            v = scratch.vertices[cell.base + 32 * k]
            for axis in range(3):
                if scratch.neighbors[cell.base + 32 * plane_at(v, axis)] < 0:
                    status[i] = 5
                    return
        # A true finite dual vertex is incident to three site bisectors. Box
        # planes have neighbor -1 and must never become tetrahedron vertices.
        for k in range(cell.nv):
            v = scratch.vertices[cell.base + 32 * k]
            a = scratch.neighbors[cell.base + 32 * plane_at(v, 0)]
            b = scratch.neighbors[cell.base + 32 * plane_at(v, 1)]
            c = scratch.neighbors[cell.base + 32 * plane_at(v, 2)]
            if i < a and i < b and i < c:
                tet = sorted4(wp.vec4i(i, a, b, c))
                at = wp.atomic_add(candidate_count, 0, 1)
                if at < candidates.shape[0]:
                    for j in range(4):
                        candidates[at, j] = tet[j]

    return build_cells


@wp.kernel
def count_distinct(owners: wp.array[int], count: wp.array[int]):
    i = wp.tid()
    if owners[i] == i:
        wp.atomic_add(count, 0, 1)


@wp.kernel
def choose(
    enabled: wp.array[int], count: wp.array[int], fast: wp.array[int], original: wp.array[int]
):
    fast[0] = int(enabled[0] != 0 and count[0] >= 8192)
    original[0] = 1 - fast[0]


class CandidateWorkspace(DiagramWorkspace):
    """Private unweighted workspace; reference swaps happen only during capture."""

    def _init_tetrahedra(self, capacity):
        super()._init_tetrahedra(capacity)
        if self.weighted or self._adjacency:
            raise ValueError("Fast candidates require an unweighted tetrahedra-only workspace")
        self.enabled = wp.ones(1, dtype=int)
        self.attempted = wp.zeros(1, dtype=int)
        self.repaired_cells = wp.zeros(1, dtype=int)
        self._distinct = wp.zeros(1, dtype=int)
        self._fast_ready = wp.empty(1, dtype=int)
        self._original_ready = wp.empty(1, dtype=int)
        self._fast_factor = wp.full(1, 0.5, dtype=wp.float64)
        self._fast_scratch = allocate_scratch(self.retry_chunk, 128, self.device)
        self._fast_gt = SimpleNamespace(**vars(self._gt))
        self._fast_gt.make_cells = self._make_cells
        self._recording_fast = False
        self._pass_capacity = 0

    def _make_cells(self, capacity):
        self._pass_capacity = capacity
        if capacity == 128:
            return make_cells(capacity)
        # The proxy is used only while self._gt points at it.
        from ._graph_tetrahedra import make_cells as original

        return original(capacity)

    def _record(self):
        self.attempted.zero_()
        self.repaired_cells.zero_()
        super()._record()

    def _record_tetrahedra(self):
        self._distinct.zero_()
        self._launch(count_distinct, self.n, [self._hierarchy.owners, self._distinct])
        self._launch(
            choose, 1, [self.enabled, self._distinct, self._fast_ready, self._original_ready]
        )
        wp.capture_if(self._fast_ready, self._record_hybrid)
        wp.capture_if(self._original_ready, super()._record_tetrahedra)

    def _record_hybrid(self):
        self.attempted.fill_(1)
        previous, scratch, factor = self._gt, self._robust[128], self._factor
        self._gt, self._robust[128], self._factor = (
            self._fast_gt,
            self._fast_scratch,
            self._fast_factor,
        )
        self._recording_fast = True
        try:
            super()._record_tetrahedra()
        finally:
            self._gt, self._robust[128], self._factor = previous, scratch, factor
            self._recording_fast = False

    def _failures(self, status):
        super()._failures(status)
        if self._recording_fast and self._pass_capacity == 128:
            wp.copy(self.repaired_cells, self._job_count)
