"""Midpoint rounding against neighboring representable-value oracles."""

import numpy as np
import pytest
import warp as wp

from warpower._boundary_rounding import rounded_boundary

wp.set_module_options({"enable_backward": False, "fast_math": False})


@wp.kernel
def evaluate(
    p: wp.array[wp.vec3],
    requested: wp.array[wp.vec3d],
    target: wp.array[wp.vec3],
    cap: wp.array[wp.float64],
    out: wp.array[wp.vec3],
    count: wp.array[int],
):
    i = wp.tid()
    r = rounded_boundary(p[i], requested[i], target[i], cap[i])
    out[i] = r.point
    count[i] = r.changed


def check(p, target, q, caps, expected, counts):
    args = [
        wp.array(x, dtype=t)
        for x, t in [(p, wp.vec3), (q, wp.vec3d), (target, wp.vec3), (caps, wp.float64)]
    ]
    output = wp.empty(len(p), dtype=wp.vec3)
    changes = wp.empty(len(p), dtype=int)
    wp.launch(evaluate, len(p), inputs=[*args, output, changes])
    np.testing.assert_array_equal(output.numpy(), expected)
    np.testing.assert_array_equal(changes.numpy(), counts)


@pytest.mark.parametrize("fuse", [False, True])
@pytest.mark.parametrize("exponent", [-140, -120, -60, -1, 0, 1, 60, 120])
def test_midpoints_neighbors_caps_and_backtracking(exponent, fuse):
    wp.set_module_options({"fuse_fp": fuse}, module=__name__)
    rng = np.random.default_rng(117)
    p = (rng.uniform(-1.7, 1.7, (1024, 3)) * 2.0**exponent).astype(np.float32)
    target = np.nextafter(
        p, np.where(rng.uniform(size=p.shape) > 0.5, np.float32(np.inf), np.float32(-np.inf))
    )
    p64, t64 = p.astype(float), target.astype(float)
    midpoint = (p64 + t64) * 0.5
    # A NumPy nextafter oracle verifies that selected targets really are the
    # adjacent representable endpoint, independently of the GPU midpoint test.
    old = midpoint.astype(np.float32)
    stalled = (old == p) & (target != p)
    assert np.any(stalled)
    expected = old.copy()
    expected[stalled] = target[stalled]
    caps = np.linalg.norm(t64 - p64, axis=1) * 1.01
    counts = np.sum(stalled, axis=1).astype(np.int32)
    check(p, target, midpoint, caps, expected, counts)
    # Below the midpoint, the ordinary rounding remains unchanged. Backtracking
    # to a quarter step cannot force the old one-ulp endpoint move.
    quarter = p64 + 0.25 * (t64 - p64)
    check(p, target, quarter, caps, quarter.astype(np.float32), np.zeros(len(p), np.int32))
    for destination in [p64, t64]:
        q = np.nextafter(midpoint, destination)
        check(p, target, q, caps, q.astype(np.float32), np.zeros(len(p), np.int32))
    # An insufficient total vector cap rejects all additional tie corrections.
    caps = np.linalg.norm(t64 - p64, axis=1) * 0.49
    check(p, target, midpoint, caps, old, np.zeros(len(p), np.int32))
    # A target many ulps away is not a license to snap a tiny damped update.
    far = (p64 + 100 * (t64 - p64)).astype(np.float32)
    check(p, far, midpoint, np.full(len(p), 1e100), old, np.zeros(len(p), np.int32))


@pytest.mark.parametrize("fuse", [False, True])
def test_nonfinite_zero_and_single_axis_exact_cap(fuse):
    wp.set_module_options({"fuse_fp": fuse}, module=__name__)
    p = np.array([[1, 0, 0], [-1, 0, 0], [0, 0, 0], [-0.0, 0, 0]], np.float32)
    target = p.copy()
    target[:, 0] = np.nextafter(p[:, 0], np.float32(np.inf))
    q = (p.astype(float) + target.astype(float)) * 0.5
    old = q.astype(np.float32)
    stalled = (old == p) & (target != p)
    caps = np.abs(target[:, 0].astype(float) - p[:, 0])
    expected = old.copy()
    expected[stalled] = target[stalled]
    check(p, target, q, caps, expected, stalled.sum(axis=1))
    check(p, target, q, np.nextafter(caps, 0), old, np.zeros(4, np.int32))
    for cap in [0.0, -1.0, np.nan, np.inf]:
        check(p, target, q, np.full(4, cap), old, np.zeros(4, np.int32))
    for value in [np.nan, np.inf, -np.inf]:
        for mode in ["point", "target", "request"]:
            pp, tt, qq = p.copy(), target.copy(), q.copy()
            if mode == "point":
                pp[:, 0] = value
            if mode == "target":
                tt[:, 0] = value
            if mode == "request":
                qq[:, 0] = value
            check(pp, tt, qq, np.full(4, 1e100), qq.astype(np.float32), np.zeros(4, np.int32))
