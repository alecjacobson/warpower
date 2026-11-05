"""Fused best-first traversal and clipping kernels."""

from functools import lru_cache
from typing import Any

import warp as wp

from ._bvh import Tree, directional_radius_squared, distance_squared
from ._cell import Cell, Cell64, Scratch, Scratch64, clip, emit, init_cell, position

wp.set_module_options({"enable_backward": False})


@wp.func(inline=True)
def node_distance(tree: Tree, node: int, p: wp.vec3, weight: float, weighted: bool):
    d2 = distance_squared(p, tree.lower[node], tree.upper[node])
    if not weighted:
        return d2 * 0.24999975
    delta = wp.min(weight - tree.weight[node], 0.0)
    numerator = d2 + delta
    if d2 < 1.0e-20 or numerator <= 0.0:
        return 0.0
    return numerator * numerator / (4.0 * d2) * 0.999999


@wp.func(inline=True)
def node_radius(tree: Tree, node: int, p: wp.vec3, lo: Any, hi: Any):
    return (
        directional_radius_squared(p, tree.lower[node], tree.upper[node], wp.vec3(lo), wp.vec3(hi))
        * 1.000002
    )


@wp.func(inline=True)
def node_may_clip(tree: Tree, node: int, p: wp.vec3, wi: float, s: Any, cell: Any, weighted: bool):
    lower = tree.lower[node] - p
    upper = tree.upper[node] - p
    delta = float(0.0)
    if weighted:
        delta = wi - tree.weight[node]
    # For each cell vertex x, minimize |q|^2-2*x.q over the node's box.
    # This is |x-clamp(x,box)|^2-|x|^2. If every vertex remains on the
    # seed's side, no site's bisector in the node can cut this convex cell.
    for k in range(cell.nv):
        x = wp.vec3(position(s, cell.base + 32 * k))
        q = wp.min(wp.max(x, lower), upper)
        q2 = wp.dot(q, q)
        products = wp.cw_mul(x, q)
        value = q2 - 2.0 * (products[0] + products[1] + products[2]) + delta
        scale = (
            q2
            + 2.0 * (wp.abs(products[0]) + wp.abs(products[1]) + wp.abs(products[2]))
            + wp.abs(delta)
        )
        if value <= 1.0e-5 * (scale + 1.0e-20):
            return True
    return False


@lru_cache(maxsize=None)
def make_construct_kernel(weighted=False, queue_capacity=64, robust=False, lane_stride=1):
    weighted_mode = wp.constant(weighted)
    queue_size = wp.constant(queue_capacity)
    vector = wp.vec3d if robust else wp.vec3
    scratch_type = Scratch64 if robust else Scratch
    cell_type = Cell64 if robust else Cell
    stride = wp.constant(lane_stride)

    @wp.kernel(enable_backward=False, module="unique", module_options={"fast_math": True})
    def construct(
        points: wp.array[wp.vec3],
        weights: wp.array[float],
        tree: Tree,
        jobs: wp.array[int],
        chunk_offset: int,
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
    ):
        if wp.tid() % stride != 0:
            return
        slot = wp.tid() // stride
        job = chunk_offset + slot
        i = jobs[job]
        output = i
        if output_by_job:
            output = job
            if status[output] == 0 or status[output] == 3:
                return
        if tree.owners[i] != i:
            counts[output] = 0
            status[output] = 3
            return
        p = points[i]
        wi = float(0.0)
        if wp.static(weighted_mode):
            wi = weights[i]
        lower = tree.lower[1] - wp.vec3(1.00001)
        upper = tree.upper[1] + wp.vec3(1.00001)
        cell = init_cell(scratch, slot, p, lower, upper, cell_type())
        if output_by_job:
            seed_row = i
            if seed_by_job:
                seed_row = job
            for k in range(seed_neighbors.shape[1]):
                j = seed_neighbors[seed_row, k]
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
        emit(scratch, cell, output, neighbors, counts, status)

    return construct


@wp.kernel(enable_backward=False)
def collect_failures(status: wp.array[int], ids: wp.array[int], count: wp.array[int]):
    i = wp.tid()
    if status[i] != 0 and status[i] != 3:
        at = wp.atomic_add(count, 0, 1)
        ids[at] = i


@wp.kernel(enable_backward=False)
def merge_retry(
    jobs: wp.array[int],
    retry_counts: wp.array[int],
    retry_status: wp.array[int],
    counts: wp.array[int],
    status: wp.array[int],
    source: wp.array[int],
):
    j = wp.tid()
    i = jobs[j]
    counts[i] = retry_counts[j]
    status[i] = retry_status[j]
    source[i] = j
