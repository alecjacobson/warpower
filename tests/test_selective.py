import numpy as np
import pytest
import warp as wp

import warpower
from tests.test_domain import brute_distances, combine, cube, solid_angle_inside, torus
from tests.test_extraction import orient
from tests.test_tetrahedra import check_mesh
from tests.test_vtm_core import consume, torus_sites, valid_core
from warpower._vtm import VTMWorkspace


def initial(n=128, iterations=12, sizing=None):
    domain = warpower.SurfaceDomain(*torus(24, 12), surface_subdivisions=2)
    core = VTMWorkspace(domain, torus_sites(n), max_iterations=iterations, sizing=sizing).run()
    assert core.error.numpy()[0] == 0
    return core


def retained_oracle(core, tolerance=0.4):
    points = core.points.numpy().astype(np.float64)
    tets = core.full_tetrahedra.numpy()[: core.full_tetrahedra_count.numpy()[0]]
    cells = points[tets]
    edges = cells[:, 1:] - cells[:, :1]
    offset = np.linalg.solve(2 * edges, np.sum(edges * edges, axis=2)[..., None])[..., 0]
    centers = cells[:, 0] + offset
    radii = np.linalg.norm(offset, axis=1)
    p = core.domain.points.numpy()
    f = core.domain._indices.numpy().reshape(-1, 3)
    keep = solid_angle_inside(p, f, centers) | (brute_distances(p, f, centers) / radii < tolerance)
    area = np.zeros(len(tets))
    for opposite in range(4):
        tri = cells[:, [k for k in range(4) if k != opposite]]
        area += np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
    quality = 1.5 * np.linalg.det(edges) / (area * radii)
    return tets, keep, quality


def test_selective_mask_matches_exactly_one_retained_neighbor_ring():
    source = initial()
    tets, keep, quality = retained_oracle(source)
    threshold = float(quality[keep].min() * 1.001)
    core = VTMWorkspace(
        source.domain, source.points, max_iterations=0, selective_threshold=threshold
    ).run()
    core.selection.select()
    bad = np.zeros(core.n, bool)
    bad[tets[keep & (quality < threshold)].ravel()] = True
    mask = np.zeros_like(bad)
    for tet in tets[keep]:
        if bad[tet].any():
            mask[tet] = True
    assert bad.any() and mask.sum() > bad.sum() and not mask.all()
    np.testing.assert_array_equal(core.selection.bad_vertices.numpy(), bad)
    np.testing.assert_array_equal(core.selection.mask.numpy(), mask)
    assert core.selection.selected_count.numpy()[0] == mask.sum()
    np.testing.assert_allclose(core.selection.minimum.numpy()[0], quality[keep].min(), rtol=1e-11)


def test_unselected_positions_stay_fixed_and_retained_minimum_does_not_decrease():
    source = initial()
    _, keep, quality = retained_oracle(source)
    threshold = float(quality[keep].min() * 1.001)
    before = source.points.numpy()
    core = VTMWorkspace(
        source.domain, source.points, max_iterations=1, selective_threshold=threshold
    ).run()
    assert core.error.numpy()[0] == 0
    mask = core.selection.mask.numpy().astype(bool)
    assert not mask.all()
    np.testing.assert_array_equal(
        core.points.numpy()[~mask].view(np.uint32), before[~mask].view(np.uint32)
    )
    _, after_keep, after_quality = retained_oracle(core)
    assert after_quality[after_keep].min() >= quality[keep].min() * (1 - 1e-9)
    valid_core(core)


def test_selective_updates_improve_a_fitted_torus_worst_element():
    source = initial(256, 20)
    _, keep, quality = retained_oracle(source)
    before = quality[keep].min()
    boundary_before = source.boundary.energy.numpy()
    core = VTMWorkspace(
        source.domain, source.points, max_iterations=8, selective_threshold=0.3
    ).run()
    assert core.error.numpy()[0] == 0
    _, keep, quality = retained_oracle(core)
    assert quality[keep].min() > before * 1.05
    assert core.iterations.numpy()[0] > 0
    assert core.boundary.energy.numpy()[0] <= boundary_before[0] * (1 + 1e-9)
    valid_core(core)


def test_no_poor_elements_stops_without_an_update():
    source = initial()
    before = source.points.numpy()
    core = VTMWorkspace(
        source.domain, source.points, max_iterations=10, selective_threshold=1e-9
    ).run()
    assert core.error.numpy()[0] == 0
    assert core.termination.numpy()[0] == 3
    assert core.iterations.numpy()[0] == core.selection.selected_count.numpy()[0] == 0
    np.testing.assert_array_equal(core.points.numpy(), before)
    valid_core(core)


def test_rejected_quality_updates_restore_original_geometry_and_statistics():
    source = initial(64, 5)
    before = source.points.numpy()
    core = VTMWorkspace(
        source.domain, source.points, max_iterations=2, max_backtracks=2, selective_threshold=1.0
    )
    real_evaluate = core.selection.evaluate

    def reject_quality():
        real_evaluate()
        core.selection.minimum.zero_()

    # Only trial checks are replaced; initial selection retains the real minimum.
    def check():
        from warpower._selective import guard_quality

        reject_quality()
        wp.launch(
            guard_quality,
            1,
            inputs=[
                core.selection.error,
                core.selection.retained_count,
                core.selection.minimum,
                core.selection.previous_minimum,
                core._accepted,
                core.selection.rejections,
                core.selection.last_error,
            ],
        )

    core.selection.check = check
    with wp.ScopedCapture() as cap:
        core.run()
    wp.capture_launch(cap.graph)
    assert core.error.numpy()[0] == 0
    assert core.iterations.numpy()[0] == 0
    assert core.termination.numpy()[0] == 2
    assert core.selection.rejections.numpy()[0] == 3
    assert core.selection.last_error.numpy()[0] == 4
    np.testing.assert_array_equal(core.points.numpy(), before)
    np.testing.assert_allclose(
        core.energy.numpy()[0], core.energy_history.numpy()[0, 0], rtol=1e-12
    )
    valid_core(core)


@wp.kernel
def produce(points: wp.array[wp.vec3], phase: wp.array[int]):
    i = wp.tid()
    base = wp.vec3(2.0, 0.0, 0.0)
    if phase[0] == 1:
        base = wp.vec3(10.0)
    if i > 0:
        base[i - 1] += 0.1
    points[i] = base


@wp.kernel
def advance(phase: wp.array[int]):
    phase[0] += 1


def test_empty_retained_domain_fails_and_recovers_in_device_producer_loop(monkeypatch, tmp_path):
    points = wp.array(
        [[2, 0, 0], [2.1, 0, 0], [2, 0.1, 0], [2, 0, 0.1]], dtype=wp.vec3, device="cuda:0"
    )
    core = VTMWorkspace(
        warpower.SurfaceDomain(*torus(24, 12)),
        points,
        max_iterations=2,
        selective_threshold=0.3,
        selective_tolerance=0,
    )
    phase = wp.zeros(1, dtype=int, device="cuda:0")
    volume = wp.zeros(1, dtype=wp.float64, device="cuda:0")
    wp.load_module(module=__name__, device="cuda:0")
    wp.load_module(module="tests.test_vtm_core", device="cuda:0")

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
            wp.launch(produce, 4, inputs=[points, phase])
            core.run()
            volume.zero_()
            wp.launch(
                consume,
                len(core.full_tetrahedra),
                inputs=[
                    points,
                    core.full_tetrahedra,
                    core.full_tetrahedra_count,
                    core.error,
                    volume,
                ],
            )
            wp.launch(advance, 1, inputs=[phase])
        wp.capture_launch(cap.graph)
    for step in range(3):
        if step:
            wp.capture_launch(cap.graph)
        if step == 1:
            assert core.error.numpy()[0] == 16
            assert core.full_tetrahedra_count.numpy()[0] == volume.numpy()[0] == 0
        else:
            assert core.error.numpy()[0] == 0
            assert core.termination.numpy()[0] == 3
            assert core.full_tetrahedra_count.numpy()[0] == 1
            np.testing.assert_allclose(volume.numpy(), core.volume.numpy(), rtol=1e-12)
    path = tmp_path / "selective.dot"
    wp.capture_debug_dot_print(cap.graph, str(path), verbose=True)
    assert all(node not in path.read_text() for node in ("MEM_ALLOC", "MEM_FREE", "HOST"))


def test_graded_selective_active_prefix_preserves_tail_and_corner_anchors():
    domain = warpower.SurfaceDomain(*cube(), surface_subdivisions=6)
    points = np.full((96, 3), np.nan, np.float32)
    points[:64] = np.random.default_rng(174).uniform(-0.9, 0.9, (64, 3))
    count = wp.array([64], dtype=int, device="cuda:0")
    field = warpower.SizingField(
        np.array([[0, 0, 0]], np.float32),
        wp.array([0.3], dtype=float, device="cuda:0"),
        gradation=0.4,
    )
    core = VTMWorkspace(
        domain, points, active_count=count, sizing=field, max_iterations=3, selective_threshold=0.3
    ).run()
    assert core.error.numpy()[0] == 0
    actual = core.points.numpy()
    assert np.isnan(actual[64:]).all()
    assert not core.selection.mask.numpy()[64:].any()
    corners = core.boundary.corner_sites.numpy()[:8]
    np.testing.assert_array_equal(actual[corners], domain.corner_points.numpy()[:8])
    tets = core.full_tetrahedra.numpy()[: core.full_tetrahedra_count.numpy()[0]]
    assert tets.max() < 64
    check_mesh(actual[:64], tets)


@wp.kernel
def moving_input(source: wp.array[wp.vec3], points: wp.array[wp.vec3], phase: wp.array[int]):
    i = wp.tid()
    points[i] = source[i] + wp.vec3(0.0001 * float(phase[0]), 0.0, 0.0)


def test_captured_selective_updates_and_quality_guard_execute_without_host_work(monkeypatch):
    fitted = initial(256, 20)
    domain, source = fitted.domain, fitted.points
    points = wp.clone(source)
    core = VTMWorkspace(domain, points, max_iterations=2, max_backtracks=8, selective_threshold=0.3)
    phase = wp.zeros(1, dtype=int, device="cuda:0")
    volume = wp.zeros(1, dtype=wp.float64, device="cuda:0")
    wp.load_module(module=__name__, device="cuda:0")
    wp.load_module(module="tests.test_vtm_core", device="cuda:0")

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
            wp.launch(moving_input, len(points), inputs=[source, points, phase])
            core.run()
            volume.zero_()
            wp.launch(
                consume,
                len(core.full_tetrahedra),
                inputs=[
                    points,
                    core.full_tetrahedra,
                    core.full_tetrahedra_count,
                    core.error,
                    volume,
                ],
            )
            wp.launch(advance, 1, inputs=[phase])
        wp.capture_launch(cap.graph)
    for step in range(3):
        if step:
            wp.capture_launch(cap.graph)
        assert core.error.numpy()[0] == core.selection.error.numpy()[0] == 0
        assert core.selection.selected_count.numpy()[0] > 0
        assert core.iterations.numpy()[0] > 0
        np.testing.assert_allclose(volume.numpy(), core.volume.numpy(), rtol=1e-12)
        valid_core(core)


def test_selection_configuration_is_validated():
    source = initial(16, 0)
    from warpower._selective import SelectiveUpdate

    for threshold in [0, -0.1, 1.1, np.nan]:
        with pytest.raises(ValueError, match="threshold"):
            SelectiveUpdate(source, threshold)
    for tolerance in [-0.1, 1.0, 1.1, np.nan, np.inf]:
        with pytest.raises(ValueError, match="tolerance"):
            SelectiveUpdate(source, 0.3, tolerance)


def compare_retention(domain, points, tets, tolerance):
    from warpower._selective import classify_retained

    points = wp.array(points, dtype=wp.vec3, device="cuda:0")
    tets = wp.array(tets, dtype=int, device="cuda:0")
    count = wp.array([len(tets)], dtype=int, device="cuda:0")
    error = wp.zeros(1, dtype=int, device="cuda:0")
    keep = wp.zeros(len(tets), dtype=int, device="cuda:0")
    quality = wp.zeros(len(tets), dtype=wp.float64, device="cuda:0")
    full = warpower.ExtractionWorkspace(domain, points, tets, count, tolerance=tolerance).run()
    wp.launch(
        classify_retained,
        len(tets),
        inputs=[
            domain.tree,
            points,
            tets,
            count,
            tolerance,
            error,
            keep,
            quality,
        ],
    )
    np.testing.assert_array_equal(error.numpy(), full.error.numpy())
    if error.numpy()[0] == 0:
        np.testing.assert_array_equal(keep.numpy(), full.selected.numpy())
        selected = keep.numpy().astype(bool)
        np.testing.assert_allclose(
            quality.numpy()[selected], full.radius_ratio.numpy()[: selected.sum()], rtol=1e-12
        )
    return full


@pytest.mark.parametrize("case", ["torus", "cavity-and-components"])
@pytest.mark.parametrize("tolerance", [0.0, 1e-100, 0.4, 0.99])
def test_retention_only_query_matches_full_distance_classifier(case, tolerance):
    from scipy.spatial import Delaunay

    if case == "torus":
        boundary = torus(16, 8)
    else:
        inner = cube(0.4)
        boundary = combine(cube(), (inner[0], inner[1][:, ::-1].copy()), cube(0.5, (4, 0, 0)))
    domain = warpower.SurfaceDomain(*boundary)
    points = np.random.default_rng(218).uniform(-3, 5, (100, 3)).astype(np.float32)
    tets = orient(points, Delaunay(points).simplices)
    compare_retention(domain, points, tets, tolerance)


@pytest.mark.parametrize("tolerance", [0.0, 0.4])
def test_retention_only_query_keeps_exact_boundary_face_edge_and_corner_centers(tolerance):
    offsets = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], np.float32) / 8
    centers = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]], np.float32)
    points = (centers[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    tets = orient(points, np.arange(12).reshape(-1, 4))
    full = compare_retention(warpower.SurfaceDomain(*cube()), points, tets, tolerance)
    np.testing.assert_array_equal(full.selected.numpy(), 1)


def test_retention_search_expansion_preserves_strict_threshold_and_neighbors():
    offsets = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], np.float64)
    points = (offsets * (0.4 / np.sqrt(3)) + [1.1, 0, 0]).astype(np.float32)
    tets = orient(points, [[0, 1, 2, 3]])
    domain = warpower.SurfaceDomain(*cube())
    full = warpower.ExtractionWorkspace(domain, points, tets).run()
    ratio = full.signed_distance.numpy()[0] / full.circumradii.numpy()[0]
    for tolerance, retained in [
        (np.nextafter(ratio, -np.inf), 0),
        (ratio, 0),
        (np.nextafter(ratio, np.inf), 1),
    ]:
        full = compare_retention(domain, points, tets, float(tolerance))
        assert full.selected.numpy()[0] == retained
