"""Bounded GPU recovery queue, with one 32-thread block per difficult cell.

Threads share clipping and node-rejection work. They keep identical traversal
control flow; only the leader emits the completed neighbor row.
"""

from functools import lru_cache

import warp as wp

from . import _retry
from ._bvh import Tree
from ._construct import node_distance, node_radius
from ._cooperative_cell import Cell64, Scratch64, clip, emit, init_cell


@wp.func
def vertex_may_clip(k: int, value: wp.vec3d, nv: int, lower: wp.vec3, upper: wp.vec3, delta: float):
    if k >= nv:
        return 0
    x = wp.vec3(value)
    q = wp.min(wp.max(x, lower), upper)
    q2 = wp.dot(q, q)
    products = wp.cw_mul(x, q)
    distance = q2 - 2.0 * (products[0] + products[1] + products[2]) + delta
    scale = (
        q2 + 2.0 * (wp.abs(products[0]) + wp.abs(products[1]) + wp.abs(products[2])) + wp.abs(delta)
    )
    return int(distance <= 1.0e-5 * (scale + 1.0e-20))


@wp.func
def node_may_clip(
    tree: Tree, node: int, p: wp.vec3, wi: float, s: Scratch64, cell: Cell64, weighted: bool
):
    lower = tree.lower[node] - p
    upper = tree.upper[node] - p
    delta = float(0.0)
    if weighted:
        delta = wi - tree.weight[node]
    positions = wp.tile_load(s.cartesian, shape=128, offset=cell.base)
    indices = wp.tile_arange(128, dtype=int)
    decisions = wp.tile_map(vertex_may_clip, indices, positions, cell.nv, lower, upper, delta)
    return wp.tile_extract(wp.tile_max(decisions), 0) != 0


@lru_cache(maxsize=None)
def make_precision_kernel(weighted=False, queue_capacity=128, robust=True, lane_stride=32):
    if queue_capacity != 128 or not robust or lane_stride != 32:
        raise ValueError("cooperative recovery requires queue=128, FP64, and 32 lanes")
    weighted_mode = wp.constant(weighted)
    queue_size = wp.constant(queue_capacity)
    vector = wp.vec3d
    scratch_type = Scratch64
    cell_type = Cell64
    stride = wp.constant(lane_stride)

    @wp.kernel(enable_backward=False, module="unique", module_options={"fast_math": True})
    def construct(
        points: wp.array[wp.vec3],
        weights: wp.array[float],
        tree: Tree,
        jobs: wp.array[int],
        chunk_offset: int,
        job_count: wp.array[int],
        row_map: wp.array[int],
        remapped: bool,
        output_by_job: bool,
        scratch: scratch_type,
        neighbors: wp.array2d[int],
        counts: wp.array[int],
        status: wp.array[int],
        seed_neighbors: wp.array2d[int],
        seed_by_job: bool,
        hints: wp.array[int],
        hint_offsets: wp.array[int],
        use_hints: bool,
        retry_states: wp.array[wp.vec2i],
    ):
        slot = wp.tid() // stride
        job = chunk_offset + slot
        if job >= job_count[0]:
            return
        i = jobs[job]
        output = i
        if output_by_job:
            output = i
            if remapped:
                output = row_map[i]
            if not _retry.cooperative_eligible(retry_states[i], status[output]):
                return
        if tree.owners[i] != i:
            if wp.tid() % 32 == 0:
                counts[output] = 0
                status[output] = 3
            return
        p = points[i]
        wi = float(0.0)
        if wp.static(weighted_mode):
            wi = weights[i]
        lower = tree.lower[1] - wp.vec3(1.00001)
        upper = tree.upper[1] + wp.vec3(1.00001)
        scratch.lane = wp.tid() % 32
        cell = init_cell(scratch, slot, p, lower, upper, cell_type())
        if output_by_job:
            seed_count = seed_neighbors.shape[1]
            for k in range(seed_count):
                j = seed_neighbors[output, k]
                if j >= 0:
                    wj = float(0.0)
                    if wp.static(weighted_mode):
                        wj = weights[j]
                    cell = clip(scratch, cell, i, j, vector(points[j]) - vector(p), wi, wj, True)
                    if cell.status != 0:
                        break
        if use_hints:
            for h in range(hint_offsets[i], hint_offsets[i + 1]):
                j = hints[h]
                if j >= 0 and j < tree.n:
                    j = tree.owners[j]
                    wj = float(0.0)
                    if wp.static(weighted_mode):
                        wj = weights[j]
                    cell = clip(scratch, cell, i, j, vector(points[j]) - vector(p), wi, wj, True)
                if cell.status != 0:
                    break

        queue_nodes = wp.zeros(shape=queue_size, dtype=int)
        queue_dist = wp.zeros(shape=queue_size, dtype=float)
        size = int(0)
        node = int(1)
        # Start with the nearest leaf before directional priorities. With the
        # initial world box, a distant direction can have the largest radius;
        # prioritizing d^2-r^2 immediately would clip far-away sites first.
        if not use_hints:
            while tree.count[node] == 0:
                left = tree.left[node]
                right = tree.right[node]
                dl = node_distance(tree, left, p, wi, wp.static(weighted_mode))
                dr = node_distance(tree, right, p, wi, wp.static(weighted_mode))
                if size == queue_size:
                    cell.status = 4
                    break
                if dl < dr:
                    node = left
                    queue_nodes[size] = right
                    queue_dist[size] = dr
                else:
                    node = right
                    queue_nodes[size] = left
                    queue_dist[size] = dl
                size += 1
        while node >= 0 and cell.status == 0:
            while tree.count[node] == 0:
                left = tree.left[node]
                right = tree.right[node]
                dl = node_distance(tree, left, p, wi, wp.static(weighted_mode))
                dr = node_distance(tree, right, p, wi, wp.static(weighted_mode))
                rl = node_radius(tree, left, p, cell.lo, cell.hi)
                rr = node_radius(tree, right, p, cell.lo, cell.hi)
                vl = dl <= rl
                vr = dr <= rr
                if not vl and not vr:
                    node = int(-1)
                    break
                far = int(-1)
                far_d = float(0.0)
                if vl and (not vr or dl - rl < dr - rr):
                    node = left
                    if vr:
                        far = right
                        far_d = dr
                else:
                    node = right
                    if vl:
                        far = left
                        far_d = dl
                if far >= 0:
                    if size == queue_size:
                        kept = int(0)
                        for k in range(size):
                            queued = queue_nodes[k]
                            if queue_dist[k] <= node_radius(
                                tree, queued, p, cell.lo, cell.hi
                            ) and node_may_clip(
                                tree, queued, p, wi, scratch, cell, wp.static(weighted_mode)
                            ):
                                queue_nodes[kept] = queued
                                queue_dist[kept] = queue_dist[k]
                                kept += 1
                        size = kept
                        if size == queue_size:
                            cell.status = 4
                            break
                    queue_nodes[size] = far
                    queue_dist[size] = far_d
                    size += 1
            if cell.status != 0:
                break
            if node >= 0:
                start = tree.start[node]
                end = start + tree.count[node]
                for k in range(start, end):
                    j = tree.ids[k]
                    if tree.owners[j] != j:
                        continue
                    wj = float(0.0)
                    if wp.static(weighted_mode):
                        wj = weights[j]
                    cell = clip(
                        scratch,
                        cell,
                        i,
                        j,
                        vector(points[j]) - vector(p),
                        wi,
                        wj,
                        use_hints or output_by_job,
                    )
                    if cell.status != 0:
                        break
            node = int(-1)
            while size > 0 and node < 0 and cell.status == 0:
                nearest = int(0)
                best = queue_dist[0]
                for k in range(1, size):
                    if queue_dist[k] < best:
                        best = queue_dist[k]
                        nearest = k
                candidate = queue_nodes[nearest]
                size -= 1
                queue_nodes[nearest] = queue_nodes[size]
                queue_dist[nearest] = queue_dist[size]
                if best <= node_radius(tree, candidate, p, cell.lo, cell.hi) and node_may_clip(
                    tree, candidate, p, wi, scratch, cell, wp.static(weighted_mode)
                ):
                    node = candidate
        if wp.tid() % 32 == 0:
            emit(scratch, cell, output, neighbors, counts, status)
            retry_states[i] = _retry.after_attempt(
                retry_states[i],
                status[output],
                128,
                True,
                True,
                _retry.capacity_failure(cell.status),
            )

    return construct
