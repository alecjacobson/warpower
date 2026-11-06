import numpy as np
import pytest
import warp as wp

import warpower
from tests.test_delaunay_recovery import exact_empty
from tests.test_domain import cube
from tests.test_jitter import fail_geometry
from tests.test_tetrahedra import check_mesh, reference
from tests.test_voronoi_volume import oracle
from tests.test_vtm_core import consume
from warpower._teleport import InteriorImprovementWorkspace
from warpower._vtm import VTMWorkspace


def initial(seed=2026, sizing=None):
    domain = warpower.SurfaceDomain(*cube(), surface_subdivisions=6)
    p = np.full((136, 3), np.nan, np.float32)
    p[:128] = np.random.default_rng(seed).uniform(-0.9, 0.9, (128, 3))
    active = wp.full(1, 128, dtype=int, device="cuda:0")
    core = VTMWorkspace(domain, p, active_count=active, sizing=sizing, max_iterations=5).run()
    assert core.error.numpy()[0] == 0
    return core


def valid(core):
    assert core.error.numpy()[0] == core.mesh.error.numpy()[0] == 0
    n = core.active_count.numpy()[0]
    p = core.points.numpy()[:n]
    t = core.full_tetrahedra.numpy()[: core.full_tetrahedra_count.numpy()[0]]
    assert t.max() < n and core.mesh.certificate_error.numpy()[0] == 0
    check_mesh(p, t)
    expected = reference(p)
    different = np.array([tet for tet in t if tuple(sorted(tet)) not in expected], np.int32)
    if len(different):
        exact_empty(p, different)
    return p, t


@pytest.mark.parametrize("grow", [False, True])
def test_interior_improvement_selects_centroid_and_dense_donor_or_inserts(grow):
    core = initial()
    before = core.points.numpy()
    roles = core.boundary.roles.numpy()
    move = InteriorImprovementWorkspace(core, target_count=129 if grow else None)
    e = move.extraction.run()
    tets = e.tetrahedra.numpy()[: e.tetrahedra_count.numpy()[0]]
    qualities = e.radius_ratio.numpy()[: len(tets)]
    eligible = (roles[tets] == 0).all(axis=1) & (qualities < 0.3)
    rows = np.flatnonzero(eligible)
    chosen = min(rows, key=lambda row: (qualities[row], tuple(sorted(tets[row]))))
    selected = np.sort(tets[chosen])
    center = before[selected].astype(np.float64).mean(axis=0).astype(np.float32)
    move.run()
    assert move.error.numpy()[0] == move.status.numpy()[0] == 0
    assert move.accepted.numpy()[0] == move.attempts.numpy()[0] == 1
    assert move.inserted.numpy()[0] == int(grow)
    assert core.active_count.numpy()[0] == move.point_count.numpy()[0] == 128 + int(grow)
    assert move.source_tetrahedron.numpy()[0] == chosen
    np.testing.assert_array_equal(move.canonical_tetrahedron.numpy(), selected)
    p, _ = valid(core)
    site = move.site.numpy()[0]
    np.testing.assert_array_equal(p[site], center)
    if grow:
        assert site == 128
        np.testing.assert_array_equal(p[:128], before[:128])
    else:
        volumes, bounded = oracle(before[:128])
        donors = bounded & (roles[:128] == 0)
        donors[selected] = False
        candidates = np.flatnonzero(donors)
        expected_donor = min(candidates, key=lambda i: (volumes[i], i))
        assert site == expected_donor and roles[site] == 0
        np.testing.assert_array_equal(np.flatnonzero(np.any(p != before[:128], axis=1)), [site])
    tail = 128 + int(grow)
    np.testing.assert_array_equal(
        core.points.numpy()[tail:].view(np.uint32), before[tail:].view(np.uint32)
    )
    np.testing.assert_array_equal(
        p[np.flatnonzero(roles[:128] != 0)], before[np.flatnonzero(roles[:128] != 0)]
    )
    assert move.metrics.numpy()[0] >= move.previous_metrics.numpy()[0] * (1 - 1e-10)
    assert move.metrics.numpy()[1] < 0.9 * move.previous_metrics.numpy()[1]
    np.testing.assert_array_equal(move.topology.numpy(), move.previous_topology.numpy())


@pytest.mark.parametrize("grow", [False, True])
def test_unhelpful_interior_move_restores_active_prefix_tail_mesh_and_history(grow):
    core = initial(174)
    before = core.points.numpy()
    tets = core.full_tetrahedra.numpy()
    history = core.energy_history.numpy()
    move = InteriorImprovementWorkspace(core, target_count=129 if grow else None).run()
    assert move.error.numpy()[0] == 0 and move.status.numpy()[0] == 2
    assert move.accepted.numpy()[0] == move.inserted.numpy()[0] == 0
    assert move.rejection_reason.numpy()[0] & 4
    assert core.active_count.numpy()[0] == move.point_count.numpy()[0] == 128
    np.testing.assert_array_equal(core.points.numpy().view(np.uint32), before.view(np.uint32))
    np.testing.assert_array_equal(core.full_tetrahedra.numpy(), tets)
    np.testing.assert_array_equal(core.energy_history.numpy(), history)
    np.testing.assert_allclose(move.metrics.numpy(), move.previous_metrics.numpy(), rtol=1e-12)
    valid(core)


def test_insertion_geometry_failure_rolls_back_then_recovers_in_capture():
    core = initial()
    before = core.points.numpy()
    move = InteriorImprovementWorkspace(core, target_count=129)
    flag = wp.ones(1, dtype=int, device=core.device)
    wp.load_module(module="tests.test_jitter", device=core.device)
    real = core.mesh.run

    def failing(stream=None):
        real(stream)
        wp.launch(fail_geometry, 1, inputs=[flag, core.mesh.error, core.full_tetrahedra_count])

    core.mesh.run = failing
    with wp.ScopedCapture() as cap:
        move.run()
    wp.capture_launch(cap.graph)
    assert move.error.numpy()[0] == 0 and move.status.numpy()[0] == 2
    assert move.last_errors.numpy()[0] == 2
    assert core.active_count.numpy()[0] == 128 and move.inserted.numpy()[0] == 0
    np.testing.assert_array_equal(core.points.numpy().view(np.uint32), before.view(np.uint32))
    valid(core)
    flag.zero_()
    wp.capture_launch(cap.graph)
    assert move.error.numpy()[0] == 0 and move.inserted.numpy()[0] == 1
    assert core.active_count.numpy()[0] == 129
    valid(core)


def test_invalid_target_zeroes_counts_without_modifying_source():
    core = initial()
    target = wp.full(1, 127, dtype=int, device=core.device)
    before = core.points.numpy()
    move = InteriorImprovementWorkspace(core, target_count=target).run()
    assert move.error.numpy()[0] == 32 and move.status.numpy()[0] == 3
    assert move.point_count.numpy()[0] == move.tetrahedra_count.numpy()[0] == 0
    np.testing.assert_array_equal(core.points.numpy().view(np.uint32), before.view(np.uint32))
    target.fill_(137)
    move.run()
    assert move.error.numpy()[0] == 32
    target.fill_(129)
    move.run()
    assert move.error.numpy()[0] == 0 and move.inserted.numpy()[0] == 1


def test_no_eligible_tet_is_valid_noop():
    core = initial()
    before = core.points.numpy()
    move = InteriorImprovementWorkspace(core, threshold=1e-12).run()
    assert move.error.numpy()[0] == 0 and move.status.numpy()[0] == 1
    assert move.attempts.numpy()[0] == 0
    assert move.point_count.numpy()[0] == 128 and move.tetrahedra_count.numpy()[0] > 0
    np.testing.assert_array_equal(core.points.numpy().view(np.uint32), before.view(np.uint32))


def test_graded_donor_is_selected_by_volume_over_local_sizing_cubed():
    locations = np.array([[-0.8, 0, 0], [0.8, 0, 0]], np.float32)
    values = wp.array([0.2, 0.5], dtype=float, device="cuda:0")
    field = warpower.SizingField(locations, values, gradation=0.3)
    fitted = initial()
    core = VTMWorkspace(
        fitted.domain,
        fitted.points,
        active_count=fitted.active_count,
        sizing=field,
        max_iterations=0,
    ).run()
    p = core.points.numpy()[:128]
    roles = core.boundary.roles.numpy()[:128]
    move = InteriorImprovementWorkspace(core, threshold=1.0).run()
    assert move.error.numpy()[0] == 0 and move.attempts.numpy()[0] == 1
    volume, bounded = oracle(p)
    mu = (
        values.numpy()[None, :]
        + 0.3 * np.linalg.norm(p.astype(np.float64)[:, None, :] - locations, axis=2)
    ).min(axis=1)
    donors = bounded & (roles == 0)
    donors[move.canonical_tetrahedron.numpy()] = False
    expected = min(np.flatnonzero(donors), key=lambda i: (volume[i] / mu[i] ** 3, i))
    assert move.site.numpy()[0] == expected


def test_volume_failure_is_explicit_and_leaves_source_untouched():
    core = initial()
    before = core.points.numpy()
    move = InteriorImprovementWorkspace(core)
    real = move.cells.run

    def failing(stream=None):
        real(stream)
        move.cells.error.fill_(8)

    move.cells.run = failing
    with wp.ScopedCapture() as cap:
        move.run()
    wp.capture_launch(cap.graph)
    assert move.error.numpy()[0] == 16 and move.status.numpy()[0] == 3
    assert move.attempts.numpy()[0] == move.point_count.numpy()[0] == 0
    np.testing.assert_array_equal(core.points.numpy().view(np.uint32), before.view(np.uint32))


def test_target_switches_insert_and_teleport_in_device_pipeline(monkeypatch, tmp_path):
    fitted = initial()
    source = wp.clone(fitted.points)
    active = wp.full(1, 128, dtype=int, device=fitted.device)
    core = VTMWorkspace(fitted.domain, wp.clone(source), active_count=active, max_iterations=0)
    target = wp.full(1, 129, dtype=int, device=core.device)
    move = InteriorImprovementWorkspace(core, target_count=target)
    volume = wp.zeros(1, dtype=wp.float64, device=core.device)
    wp.load_module(module="tests.test_vtm_core", device=core.device)

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
            wp.copy(core.points, source)
            active.fill_(128)
            core.run()
            move.run()
            volume.zero_()
            wp.launch(
                consume,
                move._m,
                inputs=[core.points, move.tetrahedra, move.tetrahedra_count, move.error, volume],
            )
        wp.capture_launch(cap.graph)
    for value in [129, 128, 127, 129]:
        target.fill_(value)
        wp.capture_launch(cap.graph)
        if value == 127:
            assert move.error.numpy()[0] == 32 and volume.numpy()[0] == 0
        else:
            assert move.error.numpy()[0] == 0 and move.accepted.numpy()[0] == 1
            assert move.point_count.numpy()[0] == value
            np.testing.assert_allclose(volume.numpy(), move.extraction.volume.numpy(), rtol=1e-12)
            valid(core)
    path = tmp_path / "interior.dot"
    wp.capture_debug_dot_print(cap.graph, str(path), verbose=True)
    assert all(node not in path.read_text() for node in ("MEM_ALLOC", "MEM_FREE", "HOST"))
