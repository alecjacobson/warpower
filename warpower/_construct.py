"""Fused best-first traversal and clipping kernels."""

from functools import lru_cache

import warp as wp

from ._bvh import Tree, directional_radius_squared, distance_squared
from ._cell import Scratch, clip, emit, init_cell

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
def node_radius(tree: Tree, node: int, p: wp.vec3, lo: wp.vec3, hi: wp.vec3):
    return directional_radius_squared(p, tree.lower[node], tree.upper[node], lo, hi) * 1.000002


@lru_cache(maxsize=None)
def make_construct_kernel(weighted=False, queue_capacity=64):
    weighted_mode = wp.constant(weighted)
    queue_size = wp.constant(queue_capacity)

    @wp.kernel(enable_backward=False, module="unique")
    def construct(
        points: wp.array[wp.vec3],
        weights: wp.array[float],
        tree: Tree,
        jobs: wp.array[int],
        chunk_offset: int,
        output_by_job: bool,
        scratch: Scratch,
        neighbors: wp.array2d[int],
        counts: wp.array[int],
        status: wp.array[int],
        hints: wp.array[int],
        hint_offsets: wp.array[int],
        use_hints: bool,
    ):
        slot = wp.tid()
        job = chunk_offset + slot
        i = jobs[job]
        p = points[i]
        wi = float(0.0)
        if wp.static(weighted_mode):
            wi = weights[i]
        lower = tree.lower[1] - wp.vec3(1.00001)
        upper = tree.upper[1] + wp.vec3(1.00001)
        cell = init_cell(scratch, slot, p, lower, upper)
        if use_hints:
            for h in range(hint_offsets[i], hint_offsets[i + 1]):
                j = hints[h]
                if j >= 0 and j < tree.n:
                    wj = float(0.0)
                    if wp.static(weighted_mode):
                        wj = weights[j]
                    cell = clip(scratch, cell, i, j, points[j] - p, wi, wj)
                if cell.status != 0:
                    break

        queue_nodes = wp.zeros(shape=queue_size, dtype=int)
        queue_dist = wp.zeros(shape=queue_size, dtype=float)
        size = int(0)
        node = int(1)
        while node >= 0 and cell.status == 0:
            while node < tree.leaf_base:
                left = 2 * node
                right = left + 1
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
                        cell.status = 4
                        break
                    queue_nodes[size] = far
                    queue_dist[size] = far_d
                    size += 1
            if cell.status != 0:
                break
            if node >= tree.leaf_base:
                start = (node - tree.leaf_base) * tree.leaf_size
                end = wp.min(start + tree.leaf_size, tree.n)
                for k in range(start, end):
                    j = tree.ids[k]
                    wj = float(0.0)
                    if wp.static(weighted_mode):
                        wj = weights[j]
                    cell = clip(scratch, cell, i, j, points[j] - p, wi, wj)
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
                if best <= node_radius(tree, candidate, p, cell.lo, cell.hi):
                    node = candidate
        output = i
        if output_by_job:
            output = job
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
