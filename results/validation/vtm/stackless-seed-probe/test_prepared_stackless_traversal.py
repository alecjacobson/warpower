"""Independent recursive coverage oracle and current segment containment."""
import numpy as np
import pytest
import warp as wp

from prepared_stackless_queries import successor
from tests.test_grouped_query_bounds import produce
from warpower import _bvh


@wp.kernel
def walk(prune: wp.array2d[int], base: int, n: int, nodes: wp.array2d[int],
         ranks: wp.array2d[int], unfinished: wp.array[int]):
    t = wp.tid()
    case, lane = t // 32, t % 32
    node = int(1)
    visits = int(0)
    while node > 0 and visits < 2 * base - 1:
        current = node
        visits += 1
        if lane == 0:
            nodes[case, current] += 1
        if prune[case, current] != 0:
            node = successor(current)
            continue
        if current < base:
            node = 2 * current
            continue
        node = successor(current)
        rank = 32 * (current - base) + lane
        if rank < n:
            ranks[case, rank] += 1
    if lane == 0:
        unfinished[case] = node


@pytest.mark.parametrize('n', [1, 4, 31, 32, 33, 63, 64, 65, 257, 1000, 4096, 4097])
@pytest.mark.parametrize('block', [32, 256])
def test_stackless_coverage_matches_recursive_pruning(n, block):
    base = max(1, (1 << (n - 1).bit_length()) // 32)
    rng = np.random.default_rng(177 + n)
    patterns = rng.integers(0, 2, (69, 2 * base), dtype=np.int32)
    patterns[0] = 0
    patterns[1] = 1
    patterns[2:5] = 0
    patterns[2, 1] = 1
    patterns[3, base:] = 1
    patterns[4, -1] = 1
    expected_nodes = np.zeros_like(patterns)
    expected_ranks = np.zeros((len(patterns), n), np.int32)
    for case, prune in enumerate(patterns):
        def recurse(node):
            expected_nodes[case, node] += 1
            if prune[node]:
                return
            if node < base:
                recurse(2 * node)
                recurse(2 * node + 1)
            else:
                start = 32 * (node - base)
                expected_ranks[case, start:min(start + 32, n)] += 1
        recurse(1)
    with wp.ScopedDevice('cuda:0'):
        prune = wp.array(patterns, dtype=int)
        nodes = wp.empty_like(prune)
        ranks = wp.empty(expected_ranks.shape, dtype=int)
        unfinished = wp.empty(len(patterns), dtype=int)
        wp.load_module(module=__name__, device='cuda:0', block_dim=block)
        with wp.ScopedCapture() as cap:
            nodes.zero_()
            ranks.zero_()
            unfinished.fill_(-1)
            wp.launch(walk, 32 * len(patterns), inputs=[prune, base, n, nodes, ranks, unfinished], block_dim=block)
        for _ in range(2):
            wp.capture_launch(cap.graph)
            np.testing.assert_array_equal(nodes.numpy(), expected_nodes)
            np.testing.assert_array_equal(ranks.numpy(), expected_ranks)
            np.testing.assert_array_equal(unfinished.numpy(), 0)


@pytest.mark.parametrize('n', [1, 4, 31, 32, 33, 65, 257, 4097])
@pytest.mark.parametrize('block', [32, 256])
def test_segment_ancestors_enclose_current_groups_and_extreme_points(n, block):
    rng = np.random.default_rng(177177 + n)
    bits = rng.integers(0, 2**32, (2, n, 3), dtype=np.uint32)
    bad = (bits & np.uint32(0x7F800000)) == np.uint32(0x7F800000)
    bits[bad] ^= np.uint32(0x00800000)
    frames = bits.view(np.float32)
    frames[0, 0] = [np.finfo(np.float32).max, -np.finfo(np.float32).max, np.nextafter(np.float32(0), np.float32(1))]
    frames[1, 0] = [-0.0, 0.0, -np.nextafter(np.float32(0), np.float32(1))]
    permutations = np.stack([rng.permutation(n), rng.permutation(n)]).astype(np.int32)
    with wp.ScopedDevice('cuda:0'):
        source = wp.array(frames, dtype=wp.vec3)
        perms = wp.array(permutations, dtype=int)
        points = wp.empty(n, dtype=wp.vec3)
        weights = wp.zeros(n, dtype=float)
        tree = _bvh.Tree()
        tree.n, tree.leaf_size, tree.leaf_base = n, 1, 1 << (n - 1).bit_length()
        tree.ids = wp.empty(n, dtype=int)
        tree.lower = wp.empty(2 * tree.leaf_base, dtype=wp.vec3)
        tree.upper = wp.empty_like(tree.lower)
        tree.weight = wp.empty(2 * tree.leaf_base, dtype=float)
        phase = wp.zeros(1, dtype=int)
        wp.load_module(module=produce.module.name, device='cuda:0', block_dim=block)
        wp.load_module(module=_bvh, device='cuda:0', block_dim=block)
        with wp.ScopedCapture() as cap:
            wp.launch(produce, n, inputs=[source, perms, phase, points, tree.ids], block_dim=block)
            tree.lower.fill_(wp.vec3(wp.nan))
            tree.upper.fill_(wp.vec3(wp.nan))
            wp.launch(_bvh.make_leaves, tree.leaf_base, inputs=[points, weights, tree], block_dim=block)
            level = tree.leaf_base // 2
            while level:
                wp.launch(_bvh.merge_level, level, inputs=[tree, level], block_dim=block)
                level //= 2
        base = max(1, tree.leaf_base // 32)
        for f in [0, 1, 0]:
            phase.fill_(f)
            wp.capture_launch(cap.graph)
            lo, hi = tree.lower.numpy(), tree.upper.numpy()
            for group in range((n + 31) // 32):
                p = frames[f, permutations[f, 32 * group:min(32 * group + 32, n)]]
                node = base + group
                while node:
                    assert np.all(lo[node] <= p) and np.all(hi[node] >= p)
                    node //= 2
