"""Spatial ordering for RadFoam's contiguous-leaf AABB tree, computed in Warp."""

import torch
import warp as wp
from warp.utils import radix_sort_pairs

from warpower._bvh import block_bounds, merge_bounds, morton_keys


@wp.kernel(enable_backward=False)
def reorder_csr(
    permutation: wp.array[wp.int64],
    inverse: wp.array[wp.int64],
    adjacency: wp.array[int],
    offsets: wp.array[int],
    new_offsets: wp.array[int],
    new_adjacency: wp.array[int],
):
    i = wp.tid()
    old = int(permutation[i])
    for j in range(offsets[old + 1] - offsets[old]):
        neighbor = adjacency[offsets[old] + j]
        new_adjacency[new_offsets[i] + j] = int(inverse[neighbor])


@torch.no_grad()
def permute_graph(adjacency, offsets, permutation):
    """Relabel exactly the same directed graph, preserving each row's order."""
    n = len(permutation)
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(n, device=permutation.device)
    old_offsets = offsets.view(torch.int32)
    counts = old_offsets[1:] - old_offsets[:-1]
    new_offsets = torch.cat(
        [
            torch.zeros(1, device=permutation.device, dtype=torch.int64),
            torch.cumsum(counts[permutation], 0),
        ]
    ).int()
    new_adjacency = torch.empty_like(adjacency.view(torch.int32))
    with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream())):
        wp.launch(
            reorder_csr,
            n,
            inputs=[
                wp.from_torch(permutation),
                wp.from_torch(inverse),
                wp.from_torch(adjacency.view(torch.int32)),
                wp.from_torch(old_offsets),
                wp.from_torch(new_offsets),
                wp.from_torch(new_adjacency),
            ],
            device="cuda:0",
        )
    return new_adjacency.view(torch.uint32), new_offsets.view(torch.uint32)


@torch.no_grad()
def morton_permutation(points):
    """Return sorted-row -> input-row indices without moving coordinates to CPU."""
    p = wp.from_torch(points.detach(), dtype=wp.vec3)
    n, device = len(p), p.device
    with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(points.device))):
        groups = (n + 255) // 256
        lower = wp.empty(groups, dtype=wp.vec3, device=device)
        upper = wp.empty(groups, dtype=wp.vec3, device=device)
        bounds = wp.array([[1e30] * 3, [-1e30] * 3], dtype=wp.vec3, device=device)
        wp.launch_tiled(
            block_bounds, dim=groups, inputs=[p, lower, upper], block_dim=256, device=device
        )
        wp.launch(merge_bounds, groups, inputs=[lower, upper, bounds], device=device)
        keys = wp.empty(2 * n, dtype=wp.uint64, device=device)
        indices = wp.empty(2 * n, dtype=int, device=device)
        wp.launch(morton_keys, n, inputs=[p, bounds, keys, indices], device=device)
        radix_sort_pairs(keys, indices, n, end_bit=63)
        return wp.to_torch(indices[:n]).long()
