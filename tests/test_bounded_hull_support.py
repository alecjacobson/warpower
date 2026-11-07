"""Fresh support masks against old traversal and exact rational plane signs."""

import numpy as np
import pytest
import warp as wp

from tests.test_facet_seeds import fixture
from tests.test_support_reuse import support_oracle
from warpower._canonical import insert, resolve
from warpower._delaunay_certificate import check_faces
from warpower._proved_completion import record_face_support as candidate


@pytest.mark.parametrize(
    "kind",
    [
        "unit",
        "subnormal",
        "slanted",
        "limit",
        "above_limit",
        "tiny",
        "huge",
        "translated",
        "signed_zero",
    ],
)
@pytest.mark.parametrize("depth", [0, 5])
@pytest.mark.parametrize("block_dim", [32, 256])
def test_masks_after_motion_ties_promotion_empty_prefix_and_restore(
    kind, depth, block_dim, monkeypatch
):
    points = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0.25, 0.25, 0.125],
            [0.25, 0.25, 0.125],
            [0.125, 0.125, 0.125],
            [0, 0, 0],
        ],
        np.float32,
    )
    eps = (
        np.nextafter(np.float32(0), np.float32(1)) if kind == "subnormal" else np.float32(2.0**-20)
    )
    frames = []
    for z in [0.125, 0, -eps, -0.25, -0.25, 0.125]:
        p = points.copy()
        p[4, 2] = z
        p[5] = p[4]
        frames.append(p)
    # Old owner4 moves inside; unchanged duplicate5 becomes the outside witness.
    frames[3][4] = [0.125, 0.125, 0.125]
    for p in frames:
        if kind == "slanted":
            p[:] = p @ np.array([[1, 0.5, 0.25], [0.25, 1, 0.5], [0.5, 0.25, 1]], np.float32)
        elif kind == "limit":
            p[:] = p * np.float32(32) - np.float32(16)
        elif kind == "above_limit":
            p[:] = p * np.float32(32) - np.float32(16)
            p[1, 0] = np.nextafter(np.float32(16), np.float32(np.inf))
        elif kind == "tiny":
            p[:] *= np.float32(2.0**-60)
        elif kind == "huge":
            p[:] *= np.float32(2.0**60)
        elif kind == "translated":
            p[:] += np.array([8, -4, 2], np.float32)
        elif kind == "signed_zero":
            p[7] = np.array([-0.0, 0.0, -0.0], np.float32)
    tets = np.array([[0, 1, 2, 3]], np.int32)
    seed, _, retained = fixture(frames[0], tets)
    hierarchy, owners, (table, buckets, bits), weights = retained
    device = seed.points.device
    faces = np.tile(np.array([1, 3, 0, 2], np.int32), 7)
    face_array = wp.array(faces, dtype=int, device=device)
    count = wp.full(1, len(faces), dtype=int, device=device)
    masks = [wp.empty(len(faces) + 3, dtype=int, device=device) for _ in range(1)]
    errors = wp.empty(2, dtype=int, device=device)
    workers = 2
    for kernel in [insert, resolve, check_faces, candidate]:
        wp.load_module(module=kernel.module, device=device, block_dim=block_dim)

    def forbidden(*args, **kwargs):
        raise AssertionError("host work during captured support checks")

    with monkeypatch.context() as patch:
        for name in [
            "empty",
            "zeros",
            "ones",
            "empty_like",
            "clone",
            "synchronize",
            "synchronize_device",
            "synchronize_stream",
        ]:
            patch.setattr(wp, name, forbidden)
        patch.setattr(wp.array, "numpy", forbidden)
        with wp.ScopedCapture() as capture:
            table.fill_(-1)
            wp.launch(
                insert,
                len(seed.points),
                inputs=[seed.points, weights, bits, table, buckets],
                device=device,
                block_dim=block_dim,
            )
            wp.launch(
                resolve,
                len(seed.points),
                inputs=[table, buckets, owners],
                device=device,
                block_dim=block_dim,
            )
            hierarchy.rebuild()
            errors.zero_()
            for mask in masks:
                mask.fill_(1)
            for i, kernel in enumerate([check_faces, candidate]):
                args = [
                    seed.points,
                    seed.source,
                    face_array,
                    count,
                    seed.tree,
                    errors[i : i + 1],
                    depth,
                    workers,
                ]
                if i:
                    args.append(masks[i - 1])
                wp.launch(
                    kernel, workers * (1 << depth), inputs=args, block_dim=block_dim, device=device
                )
    for phase, p in enumerate(frames):
        active = 0 if phase == 4 else len(faces)
        seed.points.assign(p)
        count.fill_(active)
        for mask in masks:
            mask.zero_()
        errors.fill_(127)
        wp.capture_launch(capture.graph)
        expected = np.ones(len(faces) + 3, np.int32)
        expected[:active] = support_oracle(p, tets, faces[:active])
        for mask in masks:
            np.testing.assert_array_equal(mask.numpy(), expected)
        np.testing.assert_array_equal(errors.numpy(), [64 if np.any(expected == 0) else 0] * 2)
        if phase == 3:
            assert owners.numpy()[5] == 5
