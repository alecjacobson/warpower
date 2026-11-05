import numpy as np
import warp as wp

from warpower._bvh import build


def test_hierarchy_bounds_weights_and_permutation():
    rng = np.random.default_rng(42)
    p = rng.normal(size=(1031, 3)).astype(np.float32)
    w = rng.normal(size=1031).astype(np.float32)
    points = wp.array(p, dtype=wp.vec3, device="cuda:0")
    weights = wp.array(w, device="cuda:0")
    h = build(points, weights, leaf_size=7)
    ids = h.tree.ids.numpy()[: len(p)]
    assert np.array_equal(np.sort(ids), np.arange(len(p)))
    lo, hi, weights = h.tree.lower.numpy(), h.tree.upper.numpy(), h.tree.weight.numpy()
    np.testing.assert_array_equal(lo[1], p.min(axis=0))
    np.testing.assert_array_equal(hi[1], p.max(axis=0))
    assert weights[1] == w.max()
    for node in range(1, h.tree.leaf_base):
        np.testing.assert_array_equal(lo[node], np.minimum(lo[2 * node], lo[2 * node + 1]))
        np.testing.assert_array_equal(hi[node], np.maximum(hi[2 * node], hi[2 * node + 1]))
        assert weights[node] == max(weights[2 * node], weights[2 * node + 1])
