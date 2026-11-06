"""The scalar filter must also be conservative in an FMA-enabled caller."""

from itertools import permutations
from pathlib import Path

import numpy as np
import warp as wp

from tests.test_sphere_bounds import exact_sphere
from tests.test_stable_center import rounded_fraction
from warpower._stable_center import circumcenter

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": True})


@wp.kernel
def evaluate(points: wp.array2d[wp.vec3], centers: wp.array[wp.vec3], valid: wp.array[int]):
    i = wp.tid()
    result = circumcenter(points[i, 0], points[i, 1], points[i, 2], points[i, 3])
    centers[i] = result.point
    valid[i] = result.valid


def test_fma_enabled_caller_matches_exact_rounding():
    sites = np.load(Path(__file__).parent / "data/hand-fit256.npz")["points"]
    hand = sites[[14, 50, 96, 233]]
    cases = np.concatenate(
        [
            [hand[list(p)] for p in permutations(range(4))],
            np.random.default_rng(2109).normal(size=(128, 4, 3)).astype(np.float32),
        ]
    )
    points = wp.array(cases, dtype=wp.vec3, device="cuda:0")
    centers = wp.empty(len(cases), dtype=wp.vec3, device=points.device)
    valid = wp.empty(len(cases), dtype=int, device=points.device)
    wp.launch(evaluate, len(cases), inputs=[points, centers, valid])
    np.testing.assert_array_equal(valid.numpy(), 1)
    reference = [[rounded_fraction(c) for c in exact_sphere(p)[1]] for p in cases]
    np.testing.assert_array_equal(centers.numpy(), reference)
