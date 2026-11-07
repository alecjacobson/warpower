"""Float32 min/max grouping is exact across finite ranges and partial groups."""

import numpy as np
import pytest
import warp as wp

from warpower._bvh import Tree
from warpower._grouped_seeds import group_bounds


@wp.kernel
def produce(
    frames: wp.array2d[wp.vec3],
    permutations: wp.array2d[int],
    phase: wp.array[int],
    points: wp.array[wp.vec3],
    ids: wp.array[int],
):
    i = wp.tid()
    points[i] = frames[phase[0], i]
    ids[i] = permutations[phase[0], i]


@pytest.mark.parametrize("n", [1, 4, 31, 32, 33, 65, 257])
@pytest.mark.parametrize("block", [32, 256])
def test_extreme_ranges_permutations_partial_groups_and_poison(n, block):
    rng = np.random.default_rng(172175 + n)
    bits = rng.integers(0, 2**32, (2, n, 3), dtype=np.uint32)
    bad = (bits & np.uint32(0x7F800000)) == np.uint32(0x7F800000)
    bits[bad] ^= np.uint32(0x00800000)
    frames = bits.view(np.float32)
    frames[0, 0] = [
        np.finfo(np.float32).max,
        -np.finfo(np.float32).max,
        np.nextafter(np.float32(0), np.float32(1)),
    ]
    frames[1, 0] = [-0.0, 0.0, -np.nextafter(np.float32(0), np.float32(1))]
    ids = np.stack([rng.permutation(n), rng.permutation(n)]).astype(np.int32)
    with wp.ScopedDevice("cuda:0"):
        source = wp.array(frames, dtype=wp.vec3)
        perms = wp.array(ids, dtype=int)
        points = wp.empty(n, dtype=wp.vec3)
        tree = Tree()
        tree.ids = wp.empty(n, dtype=int)
        phase = wp.zeros(1, dtype=int)
        lower = wp.empty((n + 31) // 32, dtype=wp.vec3)
        upper = wp.empty_like(lower)
        wp.load_module(module=__name__, device="cuda:0", block_dim=block)
        wp.load_module(module="warpower._grouped_seeds", device="cuda:0", block_dim=block)
        with wp.ScopedCapture() as cap:
            wp.launch(produce, n, inputs=[source, perms, phase, points, tree.ids], block_dim=block)
            lower.fill_(wp.vec3(wp.nan))
            upper.fill_(wp.vec3(wp.nan))
            wp.launch(
                group_bounds, len(lower), inputs=[points, tree, lower, upper], block_dim=block
            )
        for f in [0, 1, 0]:
            phase.fill_(f)
            wp.capture_launch(cap.graph)
            ordered = frames[f, ids[f]]
            lo = np.asarray([ordered[i : i + 32].min(axis=0) for i in range(0, n, 32)])
            hi = np.asarray([ordered[i : i + 32].max(axis=0) for i in range(0, n, 32)])
            np.testing.assert_array_equal(lower.numpy(), lo)
            np.testing.assert_array_equal(upper.numpy(), hi)
