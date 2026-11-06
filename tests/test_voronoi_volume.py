from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp
from scipy.spatial import ConvexHull, HalfspaceIntersection, Voronoi

import warpower
from warpower._voronoi_volume import BoundedVoronoiVolumes


def source(points, sizing=None):
    mesh = warpower.DelaunayWorkspace(points).run()
    assert mesh.error.numpy()[0] == 0
    return SimpleNamespace(
        n=len(points),
        points=mesh.points,
        full_tetrahedra=mesh.tetrahedra,
        full_tetrahedra_count=mesh.tetrahedra_count,
        error=mesh.error,
        mesh=mesh,
        device=mesh.device,
        active_count=None,
        sizing=sizing,
    )


def oracle(points):
    """Independent halfspace cells in normalized, site-relative coordinates."""
    p = points.astype(np.float64)
    scale = np.ptp(p, axis=0).max()
    p = (p - p.mean(axis=0)) / scale
    diagram = Voronoi(p)
    volume = np.zeros(len(p))
    bounded = np.zeros(len(p), bool)
    for i, r in enumerate(diagram.point_region):
        if -1 in diagram.regions[r] or not diagram.regions[r]:
            continue
        bounded[i] = True
        delta = p - p[i]
        delta = delta[np.any(delta != 0, axis=1)]
        halfspaces = np.column_stack((2 * delta, -np.sum(delta * delta, axis=1)))
        vertices = HalfspaceIntersection(halfspaces, np.zeros(3)).intersections
        volume[i] = ConvexHull(vertices).volume * scale**3
    return volume, bounded


@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e12])
def test_bounded_dual_volumes_match_independent_halfspace_cells(scale):
    points = ((np.random.default_rng(2026).normal(size=(80, 3)) + [8, -3, 2]) * scale).astype(
        np.float32
    )
    core = source(points)
    cells = BoundedVoronoiVolumes(core).run()
    assert cells.error.numpy()[0] == 0 and cells.invalid_cells.numpy()[0] == 0
    expected, bounded = oracle(points)
    assert bounded.sum() > 50
    np.testing.assert_array_equal(cells.valid.numpy(), bounded)
    np.testing.assert_array_equal(cells.hull.numpy(), ~bounded)
    np.testing.assert_allclose(cells.volume.numpy(), expected, rtol=1e-10, atol=0)
    np.testing.assert_array_equal(cells.volume.numpy(), cells.normalized.numpy())
    assert cells.count.numpy()[0] == bounded.sum()


def test_cospherical_lattice_has_unit_bounded_cell():
    from itertools import product

    points = np.array(list(product([-1, 0, 1], repeat=3)), np.float32)
    core = source(points)
    cells = BoundedVoronoiVolumes(core).run()
    assert cells.error.numpy()[0] == 0 and cells.count.numpy()[0] == 1
    assert cells.valid.numpy()[13] == 1
    np.testing.assert_allclose(cells.volume.numpy()[13], 1.0, rtol=1e-12)
    np.testing.assert_array_equal(cells.volume.numpy()[np.arange(27) != 13], 0)


def test_graded_normalization_uses_local_sizing_cubed():
    points = np.random.default_rng(82).normal(size=(70, 3)).astype(np.float32)
    values = wp.array([0.2, 0.4], dtype=float, device="cuda:0")
    locations = np.array([[-1, 0, 0], [1, 0, 0]], np.float32)
    field = warpower.SizingField(locations, values, gradation=0.3)
    core = source(points, field)
    cells = BoundedVoronoiVolumes(core).run()
    assert cells.error.numpy()[0] == 0

    expected, bounded = oracle(points)
    mu = (
        values.numpy()[None, :]
        + 0.3 * np.linalg.norm(points.astype(np.float64)[:, None, :] - locations, axis=2)
    ).min(axis=1)
    np.testing.assert_allclose(
        cells.normalized.numpy()[bounded], expected[bounded] / mu[bounded] ** 3, rtol=1e-10
    )
    field.error.fill_(1)
    cells.run()
    assert cells.error.numpy()[0] == 1 and cells.count.numpy()[0] == 0
    np.testing.assert_array_equal(cells.valid.numpy(), 0)
    field.refit()
    cells.run()
    assert cells.error.numpy()[0] == 0


def test_duplicate_sites_and_inactive_nan_tail_have_no_donor_volume():
    unique = np.random.default_rng(20).normal(size=(30, 3)).astype(np.float32)
    points = np.repeat(unique, 2, axis=0)
    safe = np.concatenate((points, np.repeat(points[:1], 8, axis=0)))
    core = source(safe)
    actual = safe.copy()
    actual[60:] = np.nan
    core.points = wp.array(actual, dtype=wp.vec3, device=core.device)
    core.active_count = wp.full(1, 60, dtype=int, device=core.device)
    cells = BoundedVoronoiVolumes(core).run()
    assert cells.error.numpy()[0] == 0
    expected, bounded = oracle(unique)
    np.testing.assert_allclose(cells.volume.numpy()[:60:2], expected, rtol=1e-10, atol=0)
    np.testing.assert_array_equal(cells.valid.numpy()[:60:2], bounded)
    np.testing.assert_array_equal(cells.volume.numpy()[1:60:2], 0)
    np.testing.assert_array_equal(cells.volume.numpy()[60:], 0)
    core.active_count.fill_(69)
    cells.run()
    assert cells.error.numpy()[0] == 1 and cells.count.numpy()[0] == 0


def test_current_connectivity_rebuilds_hull_even_with_stale_certificate_tables():
    core = source(np.random.default_rng(53).normal(size=(70, 3)).astype(np.float32))
    cells = BoundedVoronoiVolumes(core).run()
    saved = cells.volume.numpy()
    cells._table.fill_(-1)
    cells._occurrences.fill_(99)
    cells.hull.fill_(1)
    # This reduction owns its face table and never consumes certificate scratch.
    core.mesh._certificate._table.fill_(-1)
    cells.run()
    assert cells.error.numpy()[0] == 0
    np.testing.assert_allclose(cells.volume.numpy(), saved, rtol=1e-12)


def test_invalid_connectivity_and_source_count_clear_outputs_and_recover():
    core = source(np.random.default_rng(915).normal(size=(30, 3)).astype(np.float32))
    cells = BoundedVoronoiVolumes(core).run()
    saved = wp.clone(core.full_tetrahedra)
    count = int(core.full_tetrahedra_count.numpy()[0])
    core.full_tetrahedra.fill_(-1)
    cells.run()
    assert cells.error.numpy()[0] == 2 and cells.count.numpy()[0] == 0
    np.testing.assert_array_equal(cells.volume.numpy(), 0)
    wp.copy(core.full_tetrahedra, saved)
    core.full_tetrahedra_count.fill_(len(core.full_tetrahedra) + 1)
    cells.run()
    assert cells.error.numpy()[0] == 1 and cells.count.numpy()[0] == 0
    core.full_tetrahedra_count.fill_(count)
    cells.run()
    assert cells.error.numpy()[0] == 0 and cells.count.numpy()[0] > 0


def test_volume_reduction_captures_changing_points_without_host_work(monkeypatch, tmp_path):
    from tests.test_delaunay_recovery import produce

    core = source(np.random.default_rng(37).normal(size=(40, 3)).astype(np.float32))
    original = wp.clone(core.points)
    phase = wp.zeros(1, dtype=int, device=core.device)
    cells = BoundedVoronoiVolumes(core)
    wp.load_module(module="tests.test_delaunay_recovery", device=core.device)

    def forbidden(*args, **kwargs):
        raise AssertionError("Host allocation, readback, or synchronization")

    with monkeypatch.context() as patch:
        for name in (
            "empty",
            "zeros",
            "ones",
            "full",
            "empty_like",
            "clone",
            "synchronize",
            "synchronize_device",
            "synchronize_stream",
        ):
            patch.setattr(wp, name, forbidden)
        patch.setattr(wp.array, "numpy", forbidden)
        with wp.ScopedCapture() as cap:
            wp.launch(produce, core.n, inputs=[original, core.points, phase])
            core.mesh.run()
            cells.run()
        wp.capture_launch(cap.graph)
    for value in [0, 1, -1, 0]:
        phase.fill_(value)
        wp.capture_launch(cap.graph)
        assert cells.error.numpy()[0] == 0
        expected, bounded = oracle(core.points.numpy())
        np.testing.assert_array_equal(cells.valid.numpy(), bounded)
        np.testing.assert_allclose(cells.volume.numpy(), expected, rtol=1e-10, atol=0)
    path = tmp_path / "volumes.dot"
    wp.capture_debug_dot_print(cap.graph, str(path), verbose=True)
    assert all(node not in path.read_text() for node in ("MEM_ALLOC", "MEM_FREE", "HOST"))
