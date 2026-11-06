import numpy as np
import pytest
import warp as wp

import warpower
from tests.test_delaunay_recovery import exact_empty
from tests.test_domain import cube
from tests.test_extraction import independent_topology
from tests.test_selective import initial
from tests.test_tetrahedra import check_mesh, reference
from tests.test_vtm_core import consume, valid_core
from warpower._jitter import BoundaryJitterWorkspace
from warpower._vtm import VTMWorkspace


def test_jitter_improves_worst_element_with_one_bounded_tangential_move():
    core = initial(256, 20)
    before = core.points.numpy()
    roles, edges = core.boundary.roles.numpy(), core._edge.numpy()
    jitter = BoundaryJitterWorkspace(core).run()
    assert jitter.error.numpy()[0] == jitter.status.numpy()[0] == 0
    assert jitter.accepted.numpy()[0] == jitter.attempts.numpy()[0] == 1
    p, _ = valid_core(core)
    site = jitter.site.numpy()[0]
    assert roles[site] == 1
    np.testing.assert_array_equal(np.flatnonzero(np.any(p != before, axis=1)), [site])
    displacement = p[site].astype(np.float64) - before[site]
    assert 0 < np.linalg.norm(displacement) <= 0.1 * edges[site]
    normal = jitter.proposed_normal.numpy()[0]
    assert abs(displacement @ normal) < 2 * np.spacing(np.abs(before[site])).max()
    current, old = jitter.metrics.numpy(), jitter.previous_metrics.numpy()
    assert current[0] > 2 * old[0]
    assert current[1] < old[1]
    np.testing.assert_array_equal(jitter.topology.numpy(), jitter.previous_topology.numpy())
    retained = jitter.tetrahedra.numpy()[: jitter.tetrahedra_count.numpy()[0]]
    _, components, chi = independent_topology(p, retained)
    assert components == jitter.topology.numpy()[0] and chi == jitter.topology.numpy()[3]
    assert core.iterations.numpy()[0] == core.rejected_steps.numpy()[0] == 0
    np.testing.assert_allclose(core.energy_history.numpy()[0, 0], core.energy.numpy()[0])
    assert np.isnan(core.energy_history.numpy()[1:]).all()


def test_no_poor_tets_returns_original_valid_mesh_without_trials():
    core = initial(64, 5)
    before = core.points.numpy()
    history = core.energy_history.numpy()
    jitter = BoundaryJitterWorkspace(core, threshold=1e-12).run()
    assert jitter.error.numpy()[0] == 0
    assert jitter.status.numpy()[0] == 1
    assert jitter.attempts.numpy()[0] == jitter.accepted.numpy()[0] == 0
    assert jitter.source_tetrahedron.numpy()[0] == -1
    assert jitter.point_count.numpy()[0] == 64 and jitter.tetrahedra_count.numpy()[0] > 0
    np.testing.assert_array_equal(core.points.numpy(), before)
    np.testing.assert_array_equal(core.energy_history.numpy(), history)


def test_rejected_quality_restores_points_mesh_statistics_and_history():
    core = initial(256, 20)
    before = core.points.numpy()
    tets = core.full_tetrahedra.numpy()
    count = core.full_tetrahedra_count.numpy()
    history = core.energy_history.numpy()
    stats = [a.numpy() for a in (core.energy, core.volume, core._mass, core._moment, core._edge)]
    jitter = BoundaryJitterWorkspace(core, max_attempts=3)
    real = jitter._complete_quality

    def reject():
        real()
        jitter.accepted.zero_()

    jitter._complete_quality = reject
    with wp.ScopedCapture() as cap:
        jitter.run()
    wp.capture_launch(cap.graph)
    assert jitter.error.numpy()[0] == 0
    assert jitter.status.numpy()[0] == 2
    assert jitter.attempts.numpy()[0] == 3
    assert jitter.accepted.numpy()[0] == 0
    np.testing.assert_array_equal(core.points.numpy(), before)
    np.testing.assert_array_equal(core.full_tetrahedra.numpy(), tets)
    np.testing.assert_array_equal(core.full_tetrahedra_count.numpy(), count)
    np.testing.assert_array_equal(core.energy_history.numpy(), history)
    for a, saved in zip((core.energy, core.volume, core._mass, core._moment, core._edge), stats):
        np.testing.assert_allclose(a.numpy(), saved, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(jitter.metrics.numpy(), jitter.previous_metrics.numpy(), rtol=1e-12)
    valid_core(core)


@wp.kernel
def fail_geometry(flag: wp.array[int], error: wp.array[int], count: wp.array[int]):
    if flag[0] != 0:
        error[0] = 2
        count[0] = 0


def test_geometry_failure_rolls_back_and_recovers_on_same_graph():
    core = initial(256, 20)
    before = core.points.numpy()
    jitter = BoundaryJitterWorkspace(core, max_attempts=2)
    flag = wp.ones(1, dtype=int, device=core.device)
    wp.load_module(module=__name__, device=core.device)
    real = core.mesh.run

    def failing(stream=None):
        real(stream)
        wp.launch(fail_geometry, 1, inputs=[flag, core.mesh.error, core.full_tetrahedra_count])

    core.mesh.run = failing
    with wp.ScopedCapture() as cap:
        jitter.run()
    wp.capture_launch(cap.graph)
    assert jitter.error.numpy()[0] == 0 and jitter.status.numpy()[0] == 2
    assert jitter.last_errors.numpy()[0] == 2
    assert jitter.rejection_reason.numpy()[0] == 1
    np.testing.assert_array_equal(core.points.numpy(), before)
    valid_core(core)
    flag.zero_()
    wp.capture_launch(cap.graph)
    assert jitter.error.numpy()[0] == jitter.status.numpy()[0] == 0
    assert jitter.accepted.numpy()[0] == 1
    valid_core(core)


def test_invalid_source_publishes_zero_counts_then_recovers():
    core = initial(64, 5)
    jitter = BoundaryJitterWorkspace(core, threshold=1e-12).run()
    assert jitter.tetrahedra_count.numpy()[0] > 0
    before = core.points.numpy()
    core.error.fill_(4)
    jitter.run()
    assert jitter.error.numpy()[0] == 1 and jitter.status.numpy()[0] == 3
    for a in (jitter.point_count, jitter.full_tetrahedra_count, jitter.tetrahedra_count):
        assert a.numpy()[0] == 0
    np.testing.assert_array_equal(core.points.numpy(), before)
    core.error.zero_()
    jitter.run()
    assert jitter.error.numpy()[0] == 0 and jitter.tetrahedra_count.numpy()[0] > 0


def test_duplicate_or_rounded_noop_proposals_do_not_reconstruct():
    core = initial(64, 5)
    before = core.points.numpy()
    jitter = BoundaryJitterWorkspace(core, threshold=1.0, max_attempts=3)

    def duplicate():
        jitter.site.fill_(1)
        wp.copy(jitter.proposed_position, core.points, count=1)
        jitter.proposal_valid.fill_(1)

    jitter._propose = duplicate
    real_build = jitter._build

    def track_build():
        real_build()
        jitter.last_errors.fill_(123)

    jitter._build = track_build
    with wp.ScopedCapture() as cap:
        jitter.run()
    wp.capture_launch(cap.graph)
    assert jitter.error.numpy()[0] == 0 and jitter.status.numpy()[0] == 2
    assert jitter.invalid_proposals.numpy()[0] == jitter.attempts.numpy()[0] == 3
    assert jitter.rejection_reason.numpy()[0] == 128
    np.testing.assert_array_equal(jitter.last_errors.numpy(), 0)
    np.testing.assert_array_equal(core.points.numpy(), before)
    valid_core(core)


def test_empty_retained_domain_fails_without_moving_sites():
    from tests.test_domain import torus

    points = np.array([[10, 10, 10], [10.1, 10, 10], [10, 10.1, 10], [10, 10, 10.1]], np.float32)
    core = VTMWorkspace(warpower.SurfaceDomain(*torus(24, 12)), points, max_iterations=0).run()
    assert core.error.numpy()[0] == 0
    jitter = BoundaryJitterWorkspace(core, extraction_tolerance=0).run()
    assert jitter.error.numpy()[0] == 2 and jitter.status.numpy()[0] == 3
    assert jitter.attempts.numpy()[0] == jitter.point_count.numpy()[0] == 0
    assert jitter.tetrahedra_count.numpy()[0] == jitter.full_tetrahedra_count.numpy()[0] == 0
    np.testing.assert_array_equal(core.points.numpy(), points)


def test_graded_active_prefix_preserves_inactive_tail_corners_and_creases():
    domain = warpower.SurfaceDomain(*cube(), surface_subdivisions=6)
    points = np.full((96, 3), np.nan, np.float32)
    points[:64] = np.random.default_rng(174).uniform(-0.9, 0.9, (64, 3))
    active = wp.array([64], dtype=int, device="cuda:0")
    sizing = warpower.SizingField(
        np.array([[0, 0, 0]], np.float32), wp.array([0.3], dtype=float, device="cuda:0")
    )
    core = VTMWorkspace(domain, points, active_count=active, sizing=sizing, max_iterations=3).run()
    assert core.error.numpy()[0] == 0
    before, roles = core.points.numpy(), core.boundary.roles.numpy()
    assert np.any(roles == 1) and np.any(roles == 2) and np.any(roles == 3)
    jitter = BoundaryJitterWorkspace(core, threshold=1.0, max_attempts=2).run()
    assert jitter.error.numpy()[0] == 0
    after = core.points.numpy()
    protected = roles != 1
    np.testing.assert_array_equal(
        after[protected].view(np.uint32), before[protected].view(np.uint32)
    )
    assert jitter.point_count.numpy()[0] == active.numpy()[0] == 64
    tets = core.full_tetrahedra.numpy()[: core.full_tetrahedra_count.numpy()[0]]
    assert tets.max() < 64
    check_mesh(after[:64], tets)
    # Exact coplanar/cospherical cube sites admit different valid diagonals.
    expected = reference(after[:64])
    differing = np.array([tet for tet in tets if tuple(sorted(tet)) not in expected], np.int32)
    if len(differing):
        exact_empty(after[:64], differing)
    assert core.mesh.certificate_error.numpy()[0] == 0


def test_canonical_poor_tet_selection_is_independent_of_rows_and_vertex_order():
    from warpower._jitter import find_quality, high_key, low_key, select_tet

    core = initial(16, 0)
    jitter = BoundaryJitterWorkspace(core)
    rows = np.array([[3, 5, 8, 9], [7, 2, 8, 3], [1, 6, 9, 4], [0, 1, 2, 3]], np.int32)
    # The last tetrahedron is worse but has no movable surface vertex.
    roles = wp.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 0], dtype=int, device=core.device)
    qualities = np.array([0.1, 0.2, 0.1, 0.01])
    for order in (np.arange(4), np.array([3, 2, 0, 1])):
        tets = wp.array(rows[order, ::-1].copy(), dtype=int, device=core.device)
        q = wp.array(qualities[order], dtype=wp.float64, device=core.device)
        count = wp.full(1, 4, dtype=int, device=core.device)
        jitter._minimum.fill_(1)
        jitter._high.fill_(0xFFFFFFFFFFFFFFFF)
        jitter._low.fill_(0xFFFFFFFFFFFFFFFF)
        wp.launch(
            find_quality,
            4,
            inputs=[jitter._state, tets, count, q, roles, jitter._eligible, jitter._minimum],
        )
        wp.launch(high_key, 4, inputs=[tets, q, jitter._eligible, jitter._minimum, jitter._high])
        wp.launch(
            low_key,
            4,
            inputs=[tets, q, jitter._eligible, jitter._minimum, jitter._high, jitter._low],
        )
        wp.launch(
            select_tet,
            4,
            inputs=[
                tets,
                q,
                jitter._eligible,
                jitter._minimum,
                jitter._high,
                jitter._low,
                jitter.source_tetrahedron,
                jitter.canonical_tetrahedron,
            ],
        )
        np.testing.assert_array_equal(jitter.canonical_tetrahedron.numpy(), [1, 4, 6, 9])
        assert order[jitter.source_tetrahedron.numpy()[0]] == 2


def test_device_seed_and_producer_consumer_capture_without_host_work(monkeypatch, tmp_path):
    fitted = initial(256, 20)
    source = wp.clone(fitted.points)
    core = VTMWorkspace(fitted.domain, wp.clone(source), max_iterations=0)
    seed = wp.full(1, 2026, dtype=int, device=core.device)
    jitter = BoundaryJitterWorkspace(core, seed=seed)
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
            core.run()
            jitter.run()
            volume.zero_()
            wp.launch(
                consume,
                jitter._m,
                inputs=[
                    core.points,
                    jitter.tetrahedra,
                    jitter.tetrahedra_count,
                    jitter.error,
                    volume,
                ],
            )
        wp.capture_launch(cap.graph)
    expected = core.points.numpy()
    expected_proposal = jitter.proposed_position.numpy()
    for value in (2026, 17, 2026):
        seed.fill_(value)
        wp.capture_launch(cap.graph)
        assert jitter.error.numpy()[0] == 0
        np.testing.assert_allclose(volume.numpy(), jitter.extraction.volume.numpy(), rtol=1e-12)
        if value == 2026:
            assert jitter.accepted.numpy()[0] == 1
            np.testing.assert_array_equal(core.points.numpy(), expected)
            np.testing.assert_array_equal(jitter.proposed_position.numpy(), expected_proposal)
        else:
            assert not np.array_equal(jitter.proposed_position.numpy(), expected_proposal)
        valid_core(core)
    path = tmp_path / "jitter.dot"
    wp.capture_debug_dot_print(cap.graph, str(path), verbose=True)
    assert all(node not in path.read_text() for node in ("MEM_ALLOC", "MEM_FREE", "HOST"))


def test_jitter_configuration_is_validated():
    core = initial(16, 0)
    for name, bad in [
        ("step_fraction", 0),
        ("step_fraction", np.inf),
        ("threshold", 1.1),
        ("boundary_tolerance", -1),
        ("volume_tolerance", np.nan),
        ("max_attempts", 0),
        ("max_attempts", 1025),
        ("seed", 2**31),
    ]:
        with pytest.raises(ValueError):
            BoundaryJitterWorkspace(core, **{name: bad})
