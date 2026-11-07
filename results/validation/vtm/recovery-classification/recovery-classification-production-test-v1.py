"""Private selection preserves geometry, errors and exterior distance signs."""

import numpy as np
import pytest
import warp as wp
from scipy.spatial import Delaunay

import warpower
from tests.test_domain import combine, cube, torus
from tests.test_extraction import orient
from tests.test_wide_center import bridge, needle
from warpower import _extraction
from warpower._extraction import classify_selection
from warpower._surface_bvh import SurfaceTree, closest_point, material_side

wp.set_module_options({"enable_backward": False, "fast_math": False})


@wp.kernel
def center_comparison(
    tree: SurfaceTree,
    centers: wp.array[wp.vec3d],
    tolerance: wp.float64,
    metadata: wp.array2d[int],
    distances: wp.array2d[wp.float64],
):
    i = wp.tid()
    center = centers[i]
    status = _extraction.selection_center(tree, center, wp.float64(1.0), tolerance)
    metadata[i, 1] = status.error
    metadata[i, 3] = status.retained
    distances[i, 1] = status.distance_sign
    hit = closest_point(tree, center, wp.float64(1.0e150))
    if hit.face < 0 or not wp.isfinite(hit.distance2):
        metadata[i, 0] = 16
        return
    side = int(1)
    if hit.distance2 > wp.float64(0.0):
        side = material_side(tree, center)
    metadata[i, 4] = side
    if side != 0 and side != 1:
        metadata[i, 0] = 16
        return
    distance = wp.sqrt(hit.distance2)
    distances[i, 0] = wp.where(side == 1, -distance, distance)
    metadata[i, 2] = int(side == 1 or distance < tolerance)


@pytest.mark.parametrize("shape", ["cube", "torus", "nested-positive"])
@pytest.mark.parametrize("scale", [1e-30, 1.0, 1e30])
@pytest.mark.parametrize("mode", ["ordinary", "missing-witness", "all-missing", "empty"])
def test_synthetic_centers_preserve_search_limit_and_error_semantics(shape, scale, mode):
    if shape == "cube":
        p, f = cube()
    elif shape == "torus":
        p, f = torus(12, 6)
    else:
        # Both shells point outward: the origin has winding count two, which
        # must retain the full classifier's error instead of becoming inside.
        p, f = combine(cube(), cube(0.4))
    p = (p * scale).astype(np.float32)
    domain = warpower.SurfaceDomain(p, f)
    assert domain.error.numpy()[0] == 0
    if mode in ["missing-witness", "all-missing"]:
        first = int(domain.tree.ids.numpy()[0])
        p = p.copy()
        p[f[first]] = np.nan
        if mode == "all-missing":
            p[:] = np.nan
        wp.copy(domain.tree.points, wp.array(p, dtype=wp.vec3, device=domain.device))
        assert np.isnan(domain.tree.points.numpy()[f[first]]).all()
    elif mode == "empty":
        domain.tree.n = 0
    ordinary = (
        np.array(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1], [0.25, 0.5, 0.3], [1.5, 0, 0]],
            np.float64,
        )
        * scale
    )
    extremes = np.array(
        [
            [np.nextafter(1e140, 0), 0, 0],
            [1e140, 0, 0],
            [np.nextafter(1e140, np.inf), 0, 0],
            [9e149, 0, 0],
            [np.nextafter(1e150, 0), 0, 0],
            [1e150, 0, 0],
            [np.nextafter(1e150, np.inf), 0, 0],
            [1e160, 0, 0],
            [np.finfo(np.float64).max, 0, 0],
            [np.inf, 0, 0],
            [np.nan, 0, 0],
        ],
        np.float64,
    )
    centers = wp.array(np.vstack([ordinary, extremes]), dtype=wp.vec3d, device=domain.device)
    metadata = wp.zeros((len(centers), 5), dtype=int, device=domain.device)
    distances = wp.zeros((len(centers), 2), dtype=wp.float64, device=domain.device)
    wp.launch(
        center_comparison,
        len(centers),
        inputs=[domain.tree, centers, 0.4, metadata, distances],
        device=domain.device,
        block_dim=32,
    )
    m, d = metadata.numpy(), distances.numpy()
    np.testing.assert_array_equal(m[:, 0], m[:, 1])
    good = m[:, 0] == 0
    np.testing.assert_array_equal(m[good, 2], m[good, 3])
    np.testing.assert_array_equal(d[good, 0] <= 0, d[good, 1] <= 0)
    outside = good & (d[:, 0] > 0)
    np.testing.assert_array_equal(d[outside, 0], d[outside, 1])
    np.testing.assert_array_equal(m[-6:, 0], 16)
    if mode in ["all-missing", "empty"]:
        np.testing.assert_array_equal(m[:, 0], 16)
    elif mode == "ordinary" and shape == "nested-positive":
        assert m[0, 4] == 2 and m[0, 0] == 16


def compare(domain, p, t, tolerance, initial_error=0):
    device = domain.device
    n = len(t)
    capacity = n + 8
    points = wp.array(p, dtype=wp.vec3, device=device)
    padded = np.full((capacity, 4), -7, np.int32)
    padded[:n] = t
    cells = wp.array(padded, dtype=int, device=device)
    count = wp.full(1, n, dtype=int, device=device)
    results = []
    for kernel in [_extraction.classify, classify_selection]:
        error = wp.full(1, initial_error, dtype=int, device=device)
        centers = wp.full(capacity, wp.vec3d(77), dtype=wp.vec3d, device=device)
        radius = wp.full(capacity, 77, dtype=wp.float64, device=device)
        distance = wp.full(capacity, 77, dtype=wp.float64, device=device)
        keep = wp.full(capacity, 77, dtype=int, device=device)
        volume = wp.full(capacity, 77, dtype=wp.float64, device=device)
        quality = wp.full(capacity, 77, dtype=wp.float64, device=device)
        wp.launch(
            kernel,
            capacity,
            inputs=[
                domain.tree,
                points,
                cells,
                count,
                tolerance,
                error,
                centers,
                radius,
                distance,
                keep,
                volume,
                quality,
            ],
            device=device,
            block_dim=32,
        )
        results.append(
            {
                k: v.numpy()
                for k, v in dict(
                    error=error,
                    centers=centers,
                    radius=radius,
                    distance=distance,
                    keep=keep,
                    volume=volume,
                    quality=quality,
                ).items()
            }
        )
    a, b = results
    np.testing.assert_array_equal(a["error"], b["error"])
    if a["error"][0] == 0:
        for k in ["centers", "radius", "keep", "volume", "quality"]:
            np.testing.assert_array_equal(a[k], b[k])
        np.testing.assert_array_equal(a["distance"] <= 0, b["distance"] <= 0)
        outside = a["distance"] > 0
        np.testing.assert_array_equal(a["distance"][outside], b["distance"][outside])
    for r in results:
        np.testing.assert_array_equal(r["keep"][n:], 0)
        for k in ["centers", "radius", "distance", "volume", "quality"]:
            np.testing.assert_array_equal(r[k][n:], 77)
    return a, b


@pytest.mark.parametrize("shape", ["cube", "torus", "cavity-components"])
@pytest.mark.parametrize("scale", [1e-30, 1.0, 1e30])
@pytest.mark.parametrize("tolerance", [0.0, 1e-100, 0.4, 0.99])
def test_random_scaled_clouds(shape, scale, tolerance):
    if shape == "cube":
        p, f = cube()
    elif shape == "torus":
        p, f = torus(16, 8)
    else:
        inner = cube(0.4)
        p, f = combine(cube(), (inner[0], inner[1][:, ::-1].copy()), cube(0.5, (4, 0, 0)))
    domain = warpower.SurfaceDomain((p * scale).astype(np.float32), f)
    assert domain.error.numpy()[0] == 0
    points = np.random.default_rng(194).uniform(-3, 5, (100, 3)).astype(np.float32)
    tets = orient(points, Delaunay(points).simplices)
    a, b = compare(domain, (points * scale).astype(np.float32), tets, tolerance)
    assert a["error"][0] == 0


@pytest.mark.parametrize("tolerance", [0.0, 1e-100, 0.4, 0.99])
def test_exact_face_edge_corner_and_interiors(tolerance):
    offsets = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], np.float32) / 8
    centers = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 0, 0]], np.float32)
    p = (centers[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    t = orient(p, np.arange(16).reshape(-1, 4))
    a, b = compare(warpower.SurfaceDomain(*cube()), p, t, tolerance)
    assert a["error"][0] == 0
    np.testing.assert_array_equal(a["keep"][:4], 1)
    assert b["distance"][3] == -1


def test_strict_outside_tie_and_neighbors():
    offsets = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], np.float64)
    p = (offsets * (0.4 / np.sqrt(3)) + [1.1, 0, 0]).astype(np.float32)
    t = orient(p, [[0, 1, 2, 3]])
    domain = warpower.SurfaceDomain(*cube())
    a, _ = compare(domain, p, t, 0.4)
    ratio = a["distance"][0] / a["radius"][0]
    for tolerance, expected in [
        (np.nextafter(ratio, -np.inf), 0),
        (ratio, 0),
        (np.nextafter(ratio, np.inf), 1),
    ]:
        a, b = compare(domain, p, t, float(tolerance))
        assert a["keep"][0] == expected


@pytest.mark.parametrize("geometry", ["needle", "bridge", "large-bridge"])
def test_exact_geometry_fallback(geometry):
    p = needle() if geometry == "needle" else bridge()
    if geometry == "large-bridge":
        p = (p * 2.0**120).astype(np.float32)
    a, b = compare(
        warpower.SurfaceDomain(*cube()), p, np.arange(4, dtype=np.int32).reshape(1, 4), 0.4
    )
    assert a["error"][0] == 0
    if geometry == "large-bridge":
        assert np.max(np.abs(a["centers"][0])) > np.finfo(np.float32).max


@pytest.mark.parametrize(
    "mode",
    [
        "missing-witness",
        "all-missing",
        "nonfinite-point",
        "invalid-index",
        "reversed",
        "degenerate",
        "initial-error",
    ],
)
def test_unavailable_witness_and_errors(mode):
    p, f = cube()
    domain = warpower.SurfaceDomain(p, f)
    points = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]], np.float32)
    t = np.arange(4, dtype=np.int32).reshape(1, 4)
    if mode in ["missing-witness", "all-missing"]:
        first = int(domain.tree.ids.numpy()[0])
        p = p.copy()
        p[f[first]] = np.nan
        if mode == "all-missing":
            p[:] = np.nan
        wp.copy(domain.tree.points, wp.array(p, dtype=wp.vec3, device=domain.device))
        assert np.isnan(domain.tree.points.numpy()[f[first]]).all()
    if mode == "nonfinite-point":
        points[0] = np.nan
    if mode == "invalid-index":
        t[0, 0] = -1
    if mode == "reversed":
        t[0, :2] = t[0, 1::-1]
    if mode == "degenerate":
        points[:] = 0
    a, b = compare(domain, points, t, 0.4, 4 if mode == "initial-error" else 0)
    if mode == "all-missing":
        assert a["error"][0] == 16
    if mode in ["nonfinite-point", "invalid-index", "initial-error"]:
        assert a["error"][0] == 4
    if mode in ["reversed", "degenerate"]:
        assert a["error"][0] == 8
