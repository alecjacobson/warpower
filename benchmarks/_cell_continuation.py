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
    integers: wp.array[int]
    scalars: wp.array[wp.float64]
    n: int
    slots: int
    capacity: int


@wp.func(inline=True)
def integer_base(cache: Continuations, entry: int):
    return cache.n + 4 + entry * (3 * cache.capacity + 5)


@wp.func(inline=True)
def scalar_base(cache: Continuations, entry: int):
    return entry * (12 * cache.capacity + 7)


@wp.func(inline=False)
def save_geometry(cache: Continuations, scratch: Scratch64, cell: Cell64, entry: int):
    ib = integer_base(cache, entry)
    db = scalar_base(cache, entry)
    cap = cache.capacity
    cache.integers[ib] = cell.nv
    cache.integers[ib + 1] = cell.np
    for axis in range(3):
        cache.scalars[db + axis] = cell.lo[axis]
        cache.scalars[db + 3 + axis] = cell.hi[axis]
    cache.scalars[db + 6] = cell.r2
    for k in range(cell.np):
        at = cell.base + 32 * k
        value = scratch.planes[at]
        for axis in range(4):
            cache.scalars[db + 7 + 4 * k + axis] = value[axis]
        cache.integers[ib + 5 + k] = scratch.neighbors[at]
    for k in range(cell.nv):
        at = cell.base + 32 * k
        cache.integers[ib + 5 + cap + k] = int(scratch.vertices[at])
        value = scratch.positions[at]
        for axis in range(4):
            cache.scalars[db + 7 + 4 * cap + 4 * k + axis] = value[axis]
        point = scratch.cartesian[at]
        for axis in range(3):
            cache.scalars[db + 7 + 8 * cap + 3 * k + axis] = point[axis]


@wp.func(inline=False)
def restore_geometry(cache: Continuations, scratch: Scratch64, base: int, entry: int):
    ib = integer_base(cache, entry)
    db = scalar_base(cache, entry)
    cap = cache.capacity
    cell = Cell64()
    cell.base = base
    cell.nv = cache.integers[ib]
    cell.np = cache.integers[ib + 1]
    cell.status = 0
    for axis in range(3):
        cell.lo[axis] = cache.scalars[db + axis]
        cell.hi[axis] = cache.scalars[db + 3 + axis]
    cell.r2 = cache.scalars[db + 6]
    for k in range(cell.np):
        at = base + 32 * k
        value = wp.vec4d(0.0)
        for axis in range(4):
            value[axis] = cache.scalars[db + 7 + 4 * k + axis]
        scratch.planes[at] = value
        scratch.neighbors[at] = cache.integers[ib + 5 + k]
    for k in range(cell.nv):
        at = base + 32 * k
        scratch.vertices[at] = wp.uint32(cache.integers[ib + 5 + cap + k])
        value = wp.vec4d(0.0)
        for axis in range(4):
            value[axis] = cache.scalars[db + 7 + 4 * cap + 4 * k + axis]
        scratch.positions[at] = value
        point = wp.vec3d(0.0)
        for axis in range(3):
            point[axis] = cache.scalars[db + 7 + 8 * cap + 3 * k + axis]
        scratch.cartesian[at] = point
    return cell


@lru_cache(maxsize=None)
def make_cells(queue_capacity=128, resuming=True):
    queue_size = wp.constant(queue_capacity)
    can_resume = wp.constant(resuming)

    @wp.kernel(
        enable_backward=False,
        module="unique",
        module_options={"fast_math": False},
    )
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
        entry = int(-1)
        if wp.static(can_resume):
            entry = cache.integers[i]
            if entry >= 0:
                cell = restore_geometry(cache, scratch, cell.base, entry)
                ib = integer_base(cache, entry)
                db = scalar_base(cache, entry)
                size = cache.integers[ib + 2]
                for k in range(size):
                    nodes[k] = cache.integers[ib + 5 + 2 * cache.capacity + k]
                    distances[k] = cache.scalars[db + 7 + 11 * cache.capacity + k]
                pending = cache.integers[ib + 3]
                offset = cache.integers[ib + 4]
                wp.atomic_add(cache.integers, cache.n + 2, 1)
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
                                entry = wp.atomic_add(cache.integers, cache.n, 1)
                                if entry >= cache.slots:
                                    entry = -2
                                    wp.atomic_add(cache.integers, cache.n + 3, 1)
                                cache.integers[i] = entry
                            if entry >= 0:
                                save_geometry(cache, scratch, cell, entry)
                                ib = integer_base(cache, entry)
                                db = scalar_base(cache, entry)
                                cache.integers[ib + 2] = size
                                cache.integers[ib + 3] = node
                                cache.integers[ib + 4] = k
                                for h in range(size):
                                    cache.integers[ib + 5 + 2 * cache.capacity + h] = nodes[h]
                                    cache.scalars[db + 7 + 11 * cache.capacity + h] = distances[h]
                                wp.atomic_add(cache.integers, cache.n + 1, 1)
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


def original_schedule(workspace):
    """Explicitly retain the experiment's 128-slot overflow/resume workload."""
    base = workspace._base
    capacities = tuple(sorted(base._robust))
    if base._tet_capacities != capacities:
        base._tet_capacities = capacities
        with wp.ScopedDevice(workspace.device):
            with wp.ScopedCapture() as captured:
                base._record()
            base.graph = captured.graph
            with wp.ScopedCapture() as captured:
                workspace._record()
            workspace.graph = captured.graph


def install(workspace, *, slots=64):
    """Patch one benchmark instance, preserving certified fallbacks and defaults."""
    if slots < 1:
        raise ValueError("Continuation slots must be positive")
    original_schedule(workspace)
    base = workspace._base
    capacities = base._tet_capacities
    if len(capacities) < 2:
        raise ValueError("Continuation needs at least two capacity passes")
    capacity = max(capacities[:-1])
    with wp.ScopedDevice(workspace.device):
        cache = Continuations()
        cache.n, cache.slots, cache.capacity = workspace.n, slots, capacity
        cache.integers = wp.empty(workspace.n + 4 + slots * (3 * capacity + 5), dtype=int)
        cache.scalars = wp.empty(slots * (12 * capacity + 7), dtype=wp.float64)
        entries = cache.integers[: workspace.n]
        used = cache.integers[workspace.n : workspace.n + 1]
        statistics = cache.integers[workspace.n + 1 : workspace.n + 4]
        launch, iteration, record = base._launch, base._tet_iteration, workspace._record
        replacements = {
            _graph_tetrahedra.make_cells(c): make_cells(c, c != capacities[0]) for c in capacities
        }

        def launch_continuation(kernel, dim, args, **kwargs):
            if kernel in replacements:
                kernel, args = replacements[kernel], [*args, cache]
            launch(kernel, dim, args, **kwargs)

        def current_box():
            entries.fill_(-1)
            used.zero_()
            iteration()

        def current_input():
            statistics.zero_()
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
    arrays = [cache.integers, cache.scalars]
    workspace.continuation_cache = SimpleNamespace(
        data=SimpleNamespace(statistics=statistics, entries=entries, used=used),
        storage=cache,
        arrays=arrays,
        bytes=sum(a.size * wp.types.type_size_in_bytes(a.dtype) for a in arrays),
    )
    return workspace.continuation_cache
