"""Experimental previous-neighbor clipping; no public constructors use this module."""

from functools import lru_cache

import warp as wp

from warpower import _bvh
from warpower._cell import Cell64, Scratch64, clip, init_cell, plane_at
from warpower._tetrahedra import (
    box_distance2,
    may_clip,
    sorted4,
)

wp.set_module_options({"enable_backward": False, "fast_math": False})


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
        hints: wp.array2d[int],
        hint_counts: wp.array[int],
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
        # Recompute every plane from current positions; the old neighbor set
        # only tightens a fresh box before the complete current-input search.
        for h in range(hint_counts[i]):
            j = hints[i, h]
            if j >= 0 and j < points.shape[0]:
                j = tree.owners[j]
                cell = clip(
                    scratch,
                    cell,
                    i,
                    j,
                    wp.vec3d(points[j]) - p,
                    weights[i],
                    weights[j],
                    True,
                )
            if cell.status != 0:
                break
        nodes = wp.zeros(shape=queue_size, dtype=int)
        distances = wp.zeros(shape=queue_size, dtype=wp.float64)
        nodes[0] = 1
        distances[0] = wp.float64(0.0)
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
            if not may_clip(tree, node, p, wi, scratch, cell):
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
                            wp.vec3d(points[j]) - p,
                            weights[i],
                            weights[j],
                            True,
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


@wp.kernel
def collect_neighbors(
    tetrahedra: wp.array2d[int],
    count: wp.array[int],
    error: wp.array[int],
    table: wp.array2d[int],
    overflows: wp.array[int],
):
    task = wp.tid()
    tet, edge = task // 12, task % 12
    if error[0] != 0 or tet >= count[0]:
        return
    a, b = edge // 3, edge % 3
    if b >= a:
        b += 1
    i, j = tetrahedra[tet, a], tetrahedra[tet, b]
    slot = int((wp.uint32(j) * wp.uint32(2654435761)) & wp.uint32(table.shape[1] - 1))
    for probe in range(table.shape[1]):
        previous = wp.atomic_cas(table, i, slot, -1, j)
        if previous == -1 or previous == j:
            return
        slot = (slot + 1) & (table.shape[1] - 1)
    # Truncated hints are safe: the complete spatial search follows them.
    wp.atomic_add(overflows, 0, 1)


@wp.kernel
def order_neighbors(
    points: wp.array[wp.vec3],
    table: wp.array2d[int],
    ordered: wp.array2d[int],
    counts: wp.array[int],
):
    i = wp.tid()
    n = int(0)
    p = wp.vec3d(points[i])
    for slot in range(table.shape[1]):
        j = table[i, slot]
        if j >= 0:
            distance = wp.length_sq(wp.vec3d(points[j]) - p)
            at = n
            while at > 0:
                other = ordered[i, at - 1]
                previous = wp.length_sq(wp.vec3d(points[other]) - p)
                if previous < distance or (previous == distance and other < j):
                    break
                ordered[i, at] = other
                at -= 1
            ordered[i, at] = j
            n += 1
    counts[i] = n


def install(workspace, *, hint_capacity=128, starting_capacity=128):
    """Patch this experimental instance only; return cache and reset handles.

    The base original-coordinate builder receives hints. Normalized/symbolic
    recovery and the exact output certificate are unchanged. Each successful
    result supplies next-run neighbors, including a result produced by fallback.
    """
    from types import SimpleNamespace

    from warpower import _graph_tetrahedra

    if hint_capacity < 1 or hint_capacity & (hint_capacity - 1):
        raise ValueError("hint_capacity must be a positive power of two")
    base = workspace._base
    capacities = tuple(sorted(base._robust))
    if starting_capacity not in capacities:
        raise ValueError("Unsupported starting capacity")
    base._tet_capacities = tuple(c for c in capacities if c >= starting_capacity)
    with wp.ScopedDevice(workspace.device):
        table = wp.full((workspace.n, hint_capacity), -1, dtype=int)
        ordered = wp.empty_like(table)
        counts = wp.zeros(workspace.n, dtype=int)
        overflows = wp.zeros(1, dtype=int)
        original_launch, original_record = base._launch, workspace._record
        replacements = {
            _graph_tetrahedra.make_cells(c): make_cells(c) for c in base._tet_capacities
        }

        def launch(kernel, dim, args, **kwargs):
            if kernel in replacements:
                kernel = replacements[kernel]
                args = [*args, ordered, counts]
            original_launch(kernel, dim, args, **kwargs)

        def record():
            wp.launch(
                order_neighbors, workspace.n, inputs=[workspace.points, table, ordered, counts]
            )
            original_record()
            table.fill_(-1)
            overflows.zero_()
            wp.launch(
                collect_neighbors,
                len(workspace.tetrahedra) * 12,
                inputs=[
                    workspace.tetrahedra,
                    workspace.tetrahedra_count,
                    workspace.error,
                    table,
                    overflows,
                ],
            )

        base._launch = launch
        workspace._record = record
        with wp.ScopedCapture() as capture:
            workspace._record()
        workspace.graph = capture.graph
        wp.synchronize()
    # Retain every array and closure as long as the captured graph is used.
    workspace.neighbor_cache = SimpleNamespace(
        table=table,
        ordered=ordered,
        counts=counts,
        overflows=overflows,
        clear=lambda: table.fill_(-1),
        bytes=table.size * 4 + ordered.size * 4 + counts.size * 4 + 4,
    )
    return workspace.neighbor_cache
