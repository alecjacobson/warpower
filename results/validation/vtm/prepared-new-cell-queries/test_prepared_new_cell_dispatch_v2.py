"""Dispatch masks, including unsigned lane 31 and exhaustive rejected cells."""

from itertools import combinations, product

import numpy as np
import pytest
import warp as wp

from tests.test_prepared_query_masks import positive_canonical, rational_mask
from warpower import _bvh
from warpower._site_bvh import SiteHierarchy
from prepared_new_cell_v2 import PreparedUnprovedQueries
from warpower._proved_completion import check_unproved_spheres


@pytest.mark.parametrize(
    "mode", ["none", "lane31-rejected", "all", "overflow-first-only", "overflow-second-rejects"]
)
@pytest.mark.parametrize("block", [32, 256])
@pytest.mark.parametrize("initial_error", [0, 64])
def test_masks_reset_and_exact_dispatch(mode, block, initial_error):
    if mode == "all":
        p = np.asarray(
            [q for q in product(range(-7, 8), repeat=3) if sum(x * x for x in q) == 50], np.float32
        )
        t = positive_canonical(p, np.asarray(list(combinations(range(8), 4)), np.int32))[:1]
        want = np.uint32(0xFFFFFFFF)
    else:
        p = np.full((68, 3), 20, np.float32)
        p[:, 0] = np.arange(68, dtype=np.float32) + 20
        p[:4] = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
        if mode.startswith("overflow"):
            p[:4] = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 1]]
            p[4] = [0, 1, 1]
            if mode == "overflow-second-rejects":
                p[36] = [1, 0, 0]
        else:
            p[4] = [0.25, 0.25, 0.25]  # Rejection must not hide lane 31's tie.
        if mode == "lane31-rejected":
            p[31] = [1, 1, 1]
        t = positive_canonical(p, np.asarray([[0, 1, 2, 3]], np.int32))
        want = np.uint32(
            16 if mode.startswith("overflow") else 0x80000000 if mode == "lane31-rejected" else 0
        )
    expected = rational_mask(p, t)
    if mode.startswith("overflow"):
        assert expected.tolist() == [int(mode == "overflow-first-only")]
    with wp.ScopedDevice("cuda:0"):
        points = wp.array(p, dtype=wp.vec3)
        cells = np.full((5, 4), -1, np.int32)
        cells[:1] = t
        tets = wp.array(cells, dtype=int)
        count = wp.ones(1, dtype=int)
        hierarchy = SiteHierarchy(points)
        tree = hierarchy.tree
        tree.owners = wp.array(np.arange(len(p), dtype=np.int32), dtype=int)
        identity = wp.array(np.arange(len(p), dtype=np.int32), dtype=int)
        wp.copy(tree.ids, identity, count=len(p))
        segment = hierarchy._segment
        weights = wp.zeros(len(p), dtype=float)
        wp.load_module(module=_bvh, device="cuda:0", block_dim=block)
        wp.launch(
            _bvh.make_leaves, segment.leaf_base, inputs=[points, weights, segment], block_dim=block
        )
        level = segment.leaf_base // 2
        while level:
            wp.launch(_bvh.merge_level, level, inputs=[segment, level], block_dim=block)
            level //= 2
        error = wp.empty(1, dtype=int)
        statistics = wp.empty(1, dtype=int)
        proof_statistics = wp.empty(2, dtype=int)
        proved = wp.zeros(5, dtype=int)
        original = [points, tets, count, tree, error, statistics, proved, proof_statistics]
        split = PreparedUnprovedQueries(*original, segment=segment, block_dim=block)
        first_error = wp.empty_like(error)
        first_statistics = wp.empty_like(statistics)
        first_proofs = wp.empty_like(proof_statistics)
        wp.load_module(module="warpower._proved_completion", device="cuda:0", block_dim=block)
        with wp.ScopedCapture() as cap:
            error.fill_(initial_error)
            statistics.fill_(7)
            proof_statistics.fill_(11)
            wp.launch(check_unproved_spheres, 32 * 5, inputs=original, block_dim=block)
            wp.copy(first_error, error)
            wp.copy(first_statistics, statistics)
            wp.copy(first_proofs, proof_statistics)
            error.fill_(initial_error)
            statistics.fill_(7)
            proof_statistics.fill_(11)
            split.pending.fill_(wp.uint32(0xFFFFFFFF))
            split.unresolved.fill_(-233)
            split.run()
        for active in [1, 0, 1]:
            count.fill_(active)
            wp.capture_launch(cap.graph)
            np.testing.assert_array_equal(error.numpy(), first_error.numpy())
            np.testing.assert_array_equal(statistics.numpy(), first_statistics.numpy())
            np.testing.assert_array_equal(proof_statistics.numpy(), first_proofs.numpy())
            np.testing.assert_array_equal(split.pending.numpy()[active:], 0)
            if active:
                assert error.numpy()[0] == initial_error | (0 if expected[0] else 8)
                assert split.pending.numpy()[0] == want
                saved = split.unresolved.numpy()[0]
                if mode == "lane31-rejected":
                    assert saved[31] == 31
                elif mode == "all":
                    np.testing.assert_array_equal(saved, -2)
                elif mode.startswith("overflow"):
                    assert saved[4] == (4 if mode == "overflow-first-only" else -2)
            else:
                assert error.numpy()[0] == initial_error
                assert statistics.numpy()[0] == 7
                np.testing.assert_array_equal(proof_statistics.numpy(), 11)
