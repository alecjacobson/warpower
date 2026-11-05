"""Morton-sorted binary hierarchy with explicit, Warp-owned node arrays."""

from dataclasses import dataclass

import warp as wp
from warp.utils import radix_sort_pairs

wp.set_module_options({"enable_backward": False})


@wp.struct
class Tree:
    lower: wp.array[wp.vec3]
    upper: wp.array[wp.vec3]
    weight: wp.array[float]
    ids: wp.array[int]
    leaf_base: int
    leaf_size: int
    n: int


@wp.kernel(enable_backward=False)
def block_bounds(points: wp.array[wp.vec3], lower: wp.array[wp.vec3], upper: wp.array[wp.vec3]):
    block, lane = wp.tid()
    i = block * wp.block_dim() + lane
    lo = wp.vec3(1.0e30)
    hi = wp.vec3(-1.0e30)
    if i < points.shape[0]:
        lo = points[i]
        hi = lo
    lx = wp.tile_min(wp.tile(lo[0]))[0]
    ly = wp.tile_min(wp.tile(lo[1]))[0]
    lz = wp.tile_min(wp.tile(lo[2]))[0]
    hx = wp.tile_max(wp.tile(hi[0]))[0]
    hy = wp.tile_max(wp.tile(hi[1]))[0]
    hz = wp.tile_max(wp.tile(hi[2]))[0]
    if lane == 0:
        lower[block] = wp.vec3(lx, ly, lz)
        upper[block] = wp.vec3(hx, hy, hz)


@wp.kernel(enable_backward=False)
def merge_bounds(lower: wp.array[wp.vec3], upper: wp.array[wp.vec3], out: wp.array[wp.vec3]):
    i = wp.tid()
    wp.atomic_min(out, 0, lower[i])
    wp.atomic_max(out, 1, upper[i])


@wp.func(inline=True)
def expand_bits(value: wp.uint64):
    x = value & wp.uint64(0x1FFFFF)
    x = (x | (x << wp.uint64(32))) & wp.uint64(0x1F00000000FFFF)
    x = (x | (x << wp.uint64(16))) & wp.uint64(0x1F0000FF0000FF)
    x = (x | (x << wp.uint64(8))) & wp.uint64(0x100F00F00F00F00F)
    x = (x | (x << wp.uint64(4))) & wp.uint64(0x10C30C30C30C30C3)
    x = (x | (x << wp.uint64(2))) & wp.uint64(0x1249249249249249)
    return x


@wp.kernel(enable_backward=False)
def morton_keys(
    points: wp.array[wp.vec3],
    bounds: wp.array[wp.vec3],
    keys: wp.array[wp.uint64],
    ids: wp.array[int],
):
    i = wp.tid()
    extent = wp.max(bounds[1] - bounds[0], wp.vec3(1.0e-20))
    q = wp.cw_div(points[i] - bounds[0], extent) * 2097151.0
    q = wp.min(wp.max(q, wp.vec3(0.0)), wp.vec3(2097151.0))
    keys[i] = (
        expand_bits(wp.uint64(q[0]))
        | (expand_bits(wp.uint64(q[1])) << wp.uint64(1))
        | (expand_bits(wp.uint64(q[2])) << wp.uint64(2))
    )
    ids[i] = i


@wp.kernel(enable_backward=False)
def make_leaves(points: wp.array[wp.vec3], weights: wp.array[float], tree: Tree):
    leaf = wp.tid()
    lo = wp.vec3(1.0e30)
    hi = wp.vec3(-1.0e30)
    weight = float(-1.0e30)
    for j in range(leaf * tree.leaf_size, wp.min((leaf + 1) * tree.leaf_size, tree.n)):
        i = tree.ids[j]
        p = points[i]
        lo = wp.min(lo, p)
        hi = wp.max(hi, p)
        weight = wp.max(weight, weights[i])
    node = tree.leaf_base + leaf
    tree.lower[node] = lo
    tree.upper[node] = hi
    tree.weight[node] = weight


@wp.kernel(enable_backward=False)
def merge_level(tree: Tree, first: int):
    node = first + wp.tid()
    left = 2 * node
    right = left + 1
    tree.lower[node] = wp.min(tree.lower[left], tree.lower[right])
    tree.upper[node] = wp.max(tree.upper[left], tree.upper[right])
    tree.weight[node] = wp.max(tree.weight[left], tree.weight[right])


@dataclass
class Hierarchy:
    tree: Tree
    bounds: wp.array
    # Keep key buffers and reduction temporaries alive until queued work ends.
    temporaries: tuple


def build(points, weights, leaf_size=17):
    n = len(points)
    device = points.device
    groups = (n + 255) // 256
    partial_lo = wp.empty(groups, dtype=wp.vec3, device=device)
    partial_hi = wp.empty(groups, dtype=wp.vec3, device=device)
    bounds = wp.array([[1e30] * 3, [-1e30] * 3], dtype=wp.vec3, device=device)
    wp.launch_tiled(
        block_bounds,
        dim=groups,
        inputs=[points, partial_lo, partial_hi],
        block_dim=256,
        device=device,
    )
    wp.launch(merge_bounds, groups, inputs=[partial_lo, partial_hi, bounds], device=device)
    keys = wp.empty(2 * n, dtype=wp.uint64, device=device)
    ids = wp.empty(2 * n, dtype=int, device=device)
    wp.launch(morton_keys, n, inputs=[points, bounds, keys, ids], device=device)
    radix_sort_pairs(keys, ids, n, end_bit=63)
    tree = Tree()
    tree.leaf_base = 1 << (((n + leaf_size - 1) // leaf_size - 1).bit_length())
    tree.leaf_size = leaf_size
    tree.n = n
    tree.ids = ids
    tree.lower = wp.empty(2 * tree.leaf_base, dtype=wp.vec3, device=device)
    tree.upper = wp.empty(2 * tree.leaf_base, dtype=wp.vec3, device=device)
    tree.weight = wp.empty(2 * tree.leaf_base, dtype=float, device=device)
    wp.launch(make_leaves, tree.leaf_base, inputs=[points, weights, tree], device=device)
    level = tree.leaf_base // 2
    while level:
        wp.launch(merge_level, level, inputs=[tree, level], device=device)
        level //= 2
    return Hierarchy(tree, bounds, (keys, partial_lo, partial_hi))


@wp.func(inline=True)
def distance_squared(p: wp.vec3, lo: wp.vec3, hi: wp.vec3):
    d = wp.max(wp.max(lo - p, p - hi), wp.vec3(0.0))
    return wp.dot(d, d)


@wp.func(inline=True)
def directional_radius_squared(
    p: wp.vec3, node_lo: wp.vec3, node_hi: wp.vec3, cell_lo: wp.vec3, cell_hi: wp.vec3
):
    # Each axis independently chooses the largest squared corner coordinate
    # among signs occupied by this node. Equivalent to the eight-octant max.
    result = float(0.0)
    for axis in range(3):
        v = float(0.0)
        if node_lo[axis] <= p[axis]:
            v = cell_lo[axis] * cell_lo[axis]
        if node_hi[axis] >= p[axis]:
            v = wp.max(v, cell_hi[axis] * cell_hi[axis])
        result += v
    return result
