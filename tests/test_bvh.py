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
    left, right = h.tree.left.numpy(), h.tree.right.numpy()
    for node in range(1, len(p)):
        a, b = left[node], right[node]
        np.testing.assert_array_equal(lo[node], np.minimum(lo[a], lo[b]))
        np.testing.assert_array_equal(hi[node], np.maximum(hi[a], hi[b]))
        assert weights[node] == max(weights[a], weights[b])
    start, count = h.tree.start.numpy(), h.tree.count.numpy()
    covered = []
    stack = [1]
    while stack:
        node = stack.pop()
        if count[node]:
            assert 1 <= count[node] <= 7
            covered.extend(range(start[node], start[node] + count[node]))
        else:
            stack.extend([left[node], right[node]])
    assert sorted(covered) == list(range(len(p)))


def test_coincident_radix_keys():
    points = wp.zeros(83, dtype=wp.vec3, device="cuda:0")
    h = build(points, wp.zeros(83, device="cuda:0"), leaf_size=7)
    start, count = h.tree.start.numpy(), h.tree.count.numpy()
    left, right = h.tree.left.numpy(), h.tree.right.numpy()
    stack, covered = [1], []
    while stack:
        node = stack.pop()
        if count[node]:
            covered.extend(range(start[node], start[node] + count[node]))
        else:
            stack.extend([left[node], right[node]])
    assert sorted(covered) == list(range(83))
