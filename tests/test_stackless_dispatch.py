"""Dispatch masks, including unsigned lane 31 and exhaustive rejected cells."""

from itertools import combinations, product

import numpy as np
import pytest
import warp as wp

from tests.test_prepared_query_masks import positive_canonical, rational_mask
from warpower import _bvh
from warpower._facet_seeds import reset_live
from warpower._grouped_seeds import GroupedSeedQueries
from warpower._site_bvh import SiteHierarchy
from warpower._stackless_seeds import StacklessSeedQueries


@pytest.mark.parametrize(
    "mode", ["none", "lane31-rejected", "all", "overflow-first-only", "overflow-second-rejects"]
)
@pytest.mark.parametrize("block", [32, 256])
def test_masks_reset_and_exact_dispatch(mode, block):
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
        live = wp.empty(5, dtype=int)
        error = wp.zeros(1, dtype=int)
        current = GroupedSeedQueries(points, tets, count, tree, live, error, block_dim=block)
        split = StacklessSeedQueries(
            points, tets, count, tree, live, error, segment=segment, block_dim=block
        )
        first = wp.empty_like(live)
        first_error = wp.empty_like(error)
        wp.load_module(module="warpower._facet_seeds", device="cuda:0", block_dim=block)
        with wp.ScopedCapture() as cap:
            live.fill_(-99)
            error.zero_()
            wp.launch(reset_live, 5, inputs=[count, live], block_dim=block)
            current.run()
            wp.copy(first, live)
            wp.copy(first_error, error)
            live.fill_(-175)
            error.zero_()
            split.pending.fill_(wp.uint32(0xFFFFFFFF))
            split.unresolved.fill_(-177)
            wp.launch(reset_live, 5, inputs=[count, live], block_dim=block)
            split.run()
        for active in [1, 0, 1]:
            count.fill_(active)
            wp.capture_launch(cap.graph)
            np.testing.assert_array_equal(error.numpy(), 0)
            np.testing.assert_array_equal(first_error.numpy(), 0)
            np.testing.assert_array_equal(live.numpy(), first.numpy())
            np.testing.assert_array_equal(live.numpy()[active:], 0)
            np.testing.assert_array_equal(split.pending.numpy()[active:], 0)
            if active:
                np.testing.assert_array_equal(live.numpy()[:1], expected)
                assert split.pending.numpy()[0] == want
                saved = split.unresolved.numpy()[0]
                if mode == "lane31-rejected":
                    assert saved[31] == 31
                elif mode == "all":
                    np.testing.assert_array_equal(saved, -2)
                elif mode.startswith("overflow"):
                    assert saved[4] == (4 if mode == "overflow-first-only" else -2)
