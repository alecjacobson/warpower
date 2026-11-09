"""Experiment #271: bounded continuation of current-input cells across capacities.

Production constructors never import this module. Saved prefixes are cleared at
EVERY box expansion and fresh construction, not reused after input changes.
"""

from functools import lru_cache
from types import SimpleNamespace

import warp as wp

from warpower import _bvh, _graph_tetrahedra
from warpower._cell import Cell64, Scratch64, clip, init_cell, plane_at
from warpower._tetrahedra import box_distance2, may_clip, sorted4

wp.set_module_options({"enable_backward": False, "fast_math": False})


@wp.struct
class Continuations:
    entries: wp.array[int]
    used: wp.array[int]
    cells: wp.array[Cell64]
    sizes: wp.array[int]
    pending: wp.array[int]
    offsets: wp.array[int]
    nodes: wp.array2d[int]
    distances: wp.array2d[wp.float64]
    planes: wp.array2d[wp.vec4d]
    neighbors: wp.array2d[int]
    vertices: wp.array2d[wp.uint32]
    positions: wp.array2d[wp.vec4d]
    cartesian: wp.array2d[wp.vec3d]
    statistics: wp.array[int]
    capacity: int


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
        scratch: Scratch64,
        status: wp.array[int],
        candidates: wp.array2d[int],
        candidate_count: wp.array[int],
        cache: Continuations,
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
        p = wp.vec3d(points[i])
        wi = wp.float64(weights[i])
        cell = init_cell(scratch, slot, points[i], bounds[0], bounds[1], Cell64())
        nodes = wp.zeros(shape=queue_size, dtype=int)
        distances = wp.zeros(shape=queue_size, dtype=wp.float64)
        nodes[0] = 1
        distances[0] = wp.float64(0.0)
        size = int(1)
        pending = int(-1)
        offset = int(0)
        entry = cache.entries[i]
        if entry >= 0:
            saved = cache.cells[entry]
            # The destination's capacity/layout changed; all plane IDs remain local.
            saved.base = cell.base
            cell = saved
            for k in range(cell.np):
                at = cell.base + 32 * k
                scratch.planes[at] = cache.planes[entry, k]
                scratch.neighbors[at] = cache.neighbors[entry, k]
            for k in range(cell.nv):
                at = cell.base + 32 * k
                scratch.vertices[at] = cache.vertices[entry, k]
                scratch.positions[at] = cache.positions[entry, k]
                scratch.cartesian[at] = cache.cartesian[entry, k]
            size = cache.sizes[entry]
            for k in range(size):
                nodes[k] = cache.nodes[entry, k]
                distances[k] = cache.distances[entry, k]
            pending = cache.pending[entry]
            offset = cache.offsets[entry]
            wp.atomic_add(cache.statistics, 1, 1)
        while (size > 0 or pending >= 0) and cell.status == 0:
            node = pending
            begin = offset
            if pending < 0:
                nearest = int(0)
                for k in range(1, size):
                    if distances[k] < distances[nearest]:
                        nearest = k
                node = nodes[nearest]
                size -= 1
                nodes[nearest] = nodes[size]
                distances[nearest] = distances[size]
                begin = tree.start[node]
            pending = -1
            if not may_clip(tree, node, p, wi, scratch, cell):
                continue
            if tree.count[node] > 0:
                for k in range(begin, tree.start[node] + tree.count[node]):
                    j = tree.ids[k]
                    if tree.owners[j] == j:
                        # Pause before mutating a nearly full cell. This is a
                        # routing heuristic, never acceptance of partial geometry.
                        promote = cell.nv + 2 > scratch.capacity or cell.np == scratch.capacity
                        if promote and scratch.capacity <= cache.capacity and entry != -2:
                            if entry < 0:
                                entry = wp.atomic_add(cache.used, 0, 1)
                                if entry >= cache.cells.shape[0]:
                                    entry = -2
                                    wp.atomic_add(cache.statistics, 2, 1)
                                cache.entries[i] = entry
                            if entry >= 0:
                                cache.cells[entry] = cell
                                cache.sizes[entry] = size
                                cache.pending[entry] = node
                                cache.offsets[entry] = k
                                for h in range(size):
                                    cache.nodes[entry, h] = nodes[h]
                                    cache.distances[entry, h] = distances[h]
                                for h in range(cell.np):
                                    at = cell.base + 32 * h
                                    cache.planes[entry, h] = scratch.planes[at]
                                    cache.neighbors[entry, h] = scratch.neighbors[at]
                                for h in range(cell.nv):
                                    at = cell.base + 32 * h
                                    cache.vertices[entry, h] = scratch.vertices[at]
                                    cache.positions[entry, h] = scratch.positions[at]
                                    cache.cartesian[entry, h] = scratch.cartesian[at]
                                wp.atomic_add(cache.statistics, 0, 1)
                                status[i] = 2
                                return
                        cell = clip(
                            scratch,
                            cell,
                            i,
                            j,
                            wp.vec3d(points[j]) - p,
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


def install(workspace, *, slots=64):
    """Patch one benchmark instance, preserving certified fallbacks and defaults."""
    if slots < 1:
        raise ValueError("Continuation slots must be positive")
    base = workspace._base
    capacities = base._tet_capacities
    if len(capacities) < 2:
        raise ValueError("Continuation needs at least two capacity passes")
    capacity = max(capacities[:-1])
    with wp.ScopedDevice(workspace.device):
        cache = Continuations()
        cache.entries = wp.full(workspace.n, -1, dtype=int)
        cache.used = wp.zeros(1, dtype=int)
        cache.cells = wp.empty(slots, dtype=Cell64)
        cache.sizes = wp.empty(slots, dtype=int)
        cache.pending = wp.empty(slots, dtype=int)
        cache.offsets = wp.empty(slots, dtype=int)
        cache.nodes = wp.empty((slots, capacity), dtype=int)
        cache.distances = wp.empty((slots, capacity), dtype=wp.float64)
        cache.planes = wp.empty((slots, capacity), dtype=wp.vec4d)
        cache.neighbors = wp.empty((slots, capacity), dtype=int)
        cache.vertices = wp.empty((slots, capacity), dtype=wp.uint32)
        cache.positions = wp.empty((slots, capacity), dtype=wp.vec4d)
        cache.cartesian = wp.empty((slots, capacity), dtype=wp.vec3d)
        cache.statistics = wp.zeros(3, dtype=int)
        cache.capacity = capacity
        launch, iteration, record = base._launch, base._tet_iteration, workspace._record
        replacements = {_graph_tetrahedra.make_cells(c): make_cells(c) for c in capacities}

        def launch_continuation(kernel, dim, args, **kwargs):
            if kernel in replacements:
                kernel, args = replacements[kernel], [*args, cache]
            launch(kernel, dim, args, **kwargs)

        def current_box():
            cache.entries.fill_(-1)
            cache.used.zero_()
            iteration()

        def current_input():
            cache.statistics.zero_()
            record()

        base._launch = launch_continuation
        base._tet_iteration = current_box
        workspace._record = current_input
        with wp.ScopedCapture() as captured:
            base._record()
        base.graph = captured.graph
        with wp.ScopedCapture() as captured:
            workspace._record()
        workspace.graph = captured.graph
        wp.synchronize()
    arrays = [getattr(cache, name) for name in Continuations.vars if name != "capacity"]
    workspace.continuation_cache = SimpleNamespace(
        data=cache,
        arrays=arrays,
        bytes=sum(a.size * wp.types.type_size_in_bytes(a.dtype) for a in arrays),
    )
    return workspace.continuation_cache
