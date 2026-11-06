import numpy as np
import pytest
import warp as wp

import warpower
from tests.test_active_sites import advance_phase, validate_active
from tests.test_domain import cube, solid_angle_inside, torus
from tests.test_sizing import oracle as sizing_oracle
from tests.test_tetrahedra import check_mesh
from tests.test_vtm_core import consume, torus_sites
from warpower._refinement import RefinementWorkspace
from warpower._vtm import VTMWorkspace


def make_core(n=40, capacity=80, *, sizing=None, iterations=0, seed=213):
    domain = warpower.SurfaceDomain(*torus(24, 12))
    points = np.full((capacity, 3), np.nan, np.float32)
    points[:n] = torus_sites(n, seed)
    active = wp.array([n], dtype=int, device="cuda:0")
    core = VTMWorkspace(
        domain, points, active_count=active, sizing=sizing, max_iterations=iterations
    ).run()
    assert core.error.numpy()[0] == 0
    return core


def ranked_oracle(core, sizing=None):
    """Independent linear solves, all-triangle winding, and Python sorting."""
    n = int(core.active_count.numpy()[0])
    points = core.points.numpy()[:n].astype(np.float64)
    tets = core.full_tetrahedra.numpy()[: core.full_tetrahedra_count.numpy()[0]]
    canonical = np.sort(tets, axis=1)
    vertices = points[canonical]
    edges = vertices[:, 1:] - vertices[:, :1]
    offsets = np.linalg.solve(2 * edges, np.sum(edges * edges, axis=2)[..., None])[..., 0]
    centers = (vertices[:, 0] + offsets).astype(np.float32)
    radii = np.linalg.norm(offsets, axis=1)
    inside = solid_angle_inside(
        core.domain.points.numpy(), core.domain._indices.numpy().reshape(-1, 3), centers
    )
    sizes = np.ones(len(centers))
    if sizing is not None:
        samples, values, gradation = sizing
        sizes = sizing_oracle(samples, values, centers, gradation)
    order = sorted(range(len(tets)), key=lambda i: (-radii[i] / sizes[i], tuple(canonical[i])))
    seen = set(map(tuple, points))
    ranked = []
    for i in order:
        key = tuple(centers[i])
        if inside[i] and key not in seen:
            ranked.append(i)
            seen.add(key)
    return np.asarray(ranked, dtype=int), centers, canonical


@pytest.mark.parametrize("graded", [False, True])
def test_ranked_insertion_matches_independent_uniform_and_graded_oracles(graded):
    field = None
    sizing = None
    if graded:
        samples = np.array([[2, 0, 0], [-2, 0, 0]], np.float32)
        values = np.array([0.15, 0.7], np.float32)
        field = warpower.SizingField(samples, wp.array(values, device="cuda:0"), gradation=0.3)
        sizing = samples, values, 0.3
    core = make_core(sizing=field)
    ranked, centers, _ = ranked_oracle(core, sizing)
    before = core.points.numpy()
    refinement = RefinementWorkspace(core, 80).run()
    assert refinement.error.numpy()[0] == 0
    assert refinement.requested.numpy()[0] == refinement.inserted.numpy()[0] == 20
    assert refinement.eligible.numpy()[0] == len(ranked)
    np.testing.assert_array_equal(refinement.selected_tetrahedra.numpy()[:20], ranked[:20])
    points, _ = validate_active(core, 60)
    np.testing.assert_allclose(points[40:], centers[ranked[:20]], rtol=1e-7, atol=2e-7)
    np.testing.assert_array_equal(points[:40], before[:40])
    np.testing.assert_array_equal(
        core.points.numpy()[60:].view(np.uint32), before[60:].view(np.uint32)
    )
    np.testing.assert_allclose(
        core.energy_history.numpy()[0, 0], core.energy.numpy()[0], rtol=1e-12
    )
    assert core.iterations.numpy()[0] == 0


def test_nonconvex_growth_reaches_target_and_preserves_inactive_storage():
    core = make_core(48, 128, iterations=2)
    refinement = RefinementWorkspace(core, 128)
    for count, inserted in [(72, 24), (108, 36), (128, 20), (128, 0)]:
        refinement.run()
        assert refinement.error.numpy()[0] == 0
        assert refinement.point_count.numpy()[0] == count
        assert refinement.inserted.numpy()[0] == inserted
        assert refinement.status.numpy()[0] == int(count == 128)
        points, _ = validate_active(core, count)
        assert len(np.unique(points, axis=0)) == count
        if inserted:
            assert solid_angle_inside(*torus(24, 12), points[-inserted:]).all()
        assert np.isnan(core.points.numpy()[count:]).all()
        core.run()


def test_cospherical_cube_deduplicates_centers_and_reports_stagnation():
    domain = warpower.SurfaceDomain(*cube())
    points = np.full((16, 3), np.nan, np.float32)
    points[:8] = cube()[0]
    active = wp.array([8], dtype=int, device="cuda:0")
    core = VTMWorkspace(domain, points, active_count=active, max_iterations=0).run()
    source = core.full_tetrahedra.numpy()[: core.full_tetrahedra_count.numpy()[0]]
    first = min(range(len(source)), key=lambda i: tuple(sorted(source[i])))
    refinement = RefinementWorkspace(core, 16).run()
    assert refinement.error.numpy()[0] == 0
    assert refinement.requested.numpy()[0] == 4
    assert refinement.eligible.numpy()[0] == refinement.inserted.numpy()[0] == 1
    assert refinement.selected_tetrahedra.numpy()[0] == first
    assert active.numpy()[0] == 9
    np.testing.assert_array_equal(core.points.numpy()[8], [0, 0, 0])
    tets = core.full_tetrahedra.numpy()[: core.full_tetrahedra_count.numpy()[0]]
    check_mesh(core.points.numpy()[:9], tets)
    refinement.run()
    assert refinement.error.numpy()[0] == 0
    assert refinement.status.numpy()[0] == 2
    assert refinement.eligible.numpy()[0] == refinement.inserted.numpy()[0] == 0
    assert refinement.point_count.numpy()[0] == 9


def test_narrow_plate_uses_only_eligible_centers_and_preserves_corners():
    vertices, faces = cube()
    vertices[:, 2] *= 0.15
    domain = warpower.SurfaceDomain(vertices, faces, surface_subdivisions=4)
    points = np.full((128, 3), np.nan, np.float32)
    points[:64] = np.random.default_rng(217).uniform(-0.95, 0.95, (64, 3)) * [1, 1, 0.15]
    active = wp.array([64], dtype=int, device="cuda:0")
    core = VTMWorkspace(domain, points, active_count=active, max_iterations=1).run()
    ranked, centers, _ = ranked_oracle(core)
    refinement = RefinementWorkspace(core, 128).run()
    assert refinement.error.numpy()[0] == 0
    assert refinement.eligible.numpy()[0] == len(ranked)
    inserted = min(32, len(ranked))
    assert refinement.inserted.numpy()[0] == inserted
    np.testing.assert_array_equal(
        refinement.selected_tetrahedra.numpy()[:inserted], ranked[:inserted]
    )
    actual = core.points.numpy()
    np.testing.assert_allclose(
        actual[64 : 64 + inserted], centers[ranked[:inserted]], rtol=1e-7, atol=2e-7
    )
    corners = core.boundary.corner_sites.numpy()[:8]
    np.testing.assert_array_equal(actual[corners], domain.corner_points.numpy()[:8])
    check_mesh(
        actual[: 64 + inserted],
        core.full_tetrahedra.numpy()[: core.full_tetrahedra_count.numpy()[0]],
    )


def test_priority_ties_and_centers_are_invariant_to_connectivity_order():
    results = []
    for permute in [False, True]:
        core = make_core()
        count = int(core.full_tetrahedra_count.numpy()[0])
        tets = core.full_tetrahedra.numpy()
        if permute:
            # Two swaps preserve positive orientation.
            tets[:count] = tets[:count][::-1][:, [1, 0, 3, 2]]
            core.full_tetrahedra.assign(tets)
        refinement = RefinementWorkspace(core, 60).run()
        assert refinement.error.numpy()[0] == 0
        chosen = refinement.selected_tetrahedra.numpy()[:20]
        results.append((core.points.numpy()[40:60], np.sort(tets[chosen], axis=1)))
    np.testing.assert_array_equal(results[0][0], results[1][0])
    np.testing.assert_array_equal(results[0][1], results[1][1])


def test_device_target_validation_and_failure_recovery():
    core = make_core()
    target = wp.array([39], dtype=int, device="cuda:0")
    refinement = RefinementWorkspace(core, target)
    before = core.points.numpy()
    for bad in [39, -1, 81]:
        target.fill_(bad)
        refinement.run()
        assert refinement.error.numpy()[0] == 1
        assert refinement.status.numpy()[0] == 3
        assert refinement.point_count.numpy()[0] == refinement.tetrahedra_count.numpy()[0] == 0
        assert core.active_count.numpy()[0] == 40
        np.testing.assert_array_equal(core.points.numpy(), before)
    target.fill_(40)
    refinement.run()
    assert refinement.error.numpy()[0] == 0
    assert refinement.status.numpy()[0] == 1
    assert refinement.inserted.numpy()[0] == 0
    target.fill_(45)
    refinement.run()
    assert refinement.error.numpy()[0] == 0
    assert refinement.inserted.numpy()[0] == 5
    validate_active(core, 45)


def test_invalid_source_connectivity_and_upstream_field_fail_without_insertion():
    values = wp.ones(1, device="cuda:0")
    field = warpower.SizingField(np.array([[2, 0, 0]], np.float32), values)
    core = make_core(sizing=field)
    refinement = RefinementWorkspace(core, 60)
    original = core.full_tetrahedra.numpy()
    invalid = original.copy()
    invalid[0, 0] = -1
    core.full_tetrahedra.assign(invalid)
    refinement.run()
    assert refinement.error.numpy()[0] == 2
    assert refinement.point_count.numpy()[0] == 0
    assert core.active_count.numpy()[0] == 40
    core.full_tetrahedra.assign(original)
    values.fill_(-1)
    field.refit()
    refinement.run()
    assert refinement.error.numpy()[0] == 1
    assert refinement.point_count.numpy()[0] == 0
    values.fill_(1)
    field.refit()
    core.run()
    refinement.run()
    assert refinement.error.numpy()[0] == 0
    validate_active(core, 60)


@wp.kernel
def force_rebuild_error(flag: wp.array[int], error: wp.array[int], count: wp.array[int]):
    if flag[0] != 0:
        error[0] = 2
        count[0] = 0


def test_rejected_insertion_restores_points_active_count_connectivity_and_statistics():
    core = make_core(iterations=1)
    old_points = core.points.numpy()
    old_tets = core.full_tetrahedra.numpy()
    old_count = core.full_tetrahedra_count.numpy()
    old_history = core.energy_history.numpy()
    old_energy = core.energy.numpy()
    old_volume = core.volume.numpy()
    old_iterations = core.iterations.numpy()
    flag = wp.ones(1, dtype=int, device="cuda:0")
    real_run = core.mesh.run
    wp.load_module(module=__name__, device="cuda:0")

    def failing_run(stream=None):
        real_run(stream)
        wp.launch(
            force_rebuild_error, 1, inputs=[flag, core.mesh.error, core.full_tetrahedra_count]
        )

    core.mesh.run = failing_run
    refinement = RefinementWorkspace(core, 60).run()
    assert refinement.error.numpy()[0] == 4
    assert refinement.status.numpy()[0] == 3
    assert refinement.point_count.numpy()[0] == refinement.tetrahedra_count.numpy()[0] == 0
    assert refinement.inserted.numpy()[0] == 0 and refinement.attempted.numpy()[0] == 20
    assert refinement.last_errors.numpy()[0] == 2
    np.testing.assert_array_equal(core.points.numpy().view(np.uint32), old_points.view(np.uint32))
    np.testing.assert_array_equal(core.full_tetrahedra.numpy(), old_tets)
    np.testing.assert_array_equal(core.full_tetrahedra_count.numpy(), old_count)
    np.testing.assert_array_equal(core.energy_history.numpy(), old_history)
    np.testing.assert_array_equal(core.iterations.numpy(), old_iterations)
    np.testing.assert_allclose(core.energy.numpy(), old_energy, rtol=1e-12)
    np.testing.assert_allclose(core.volume.numpy(), old_volume, rtol=1e-12)
    validate_active(core, 40)
    flag.zero_()
    refinement.run()
    assert refinement.error.numpy()[0] == 0
    assert np.isnan(core.energy_history.numpy()[1:]).all()
    validate_active(core, 60)


@wp.kernel
def produce_refinement(
    source: wp.array[wp.vec3],
    points: wp.array[wp.vec3],
    active: wp.array[int],
    target: wp.array[int],
    phase: wp.array[int],
):
    i = wp.tid()
    if i == 0:
        active[0] = 32
        target[0] = 40 + 4 * phase[0]
        if phase[0] == 1:
            target[0] = 0
    points[i] = wp.vec3(wp.nan)
    if i < 32:
        points[i] = source[i] + wp.vec3(0.0001 * float(phase[0]), 0.0, 0.0)


def test_captured_producer_refinement_consumer_recovers_and_has_no_host_work(monkeypatch, tmp_path):
    core = make_core(32, 64, iterations=1)
    target = wp.array([40], dtype=int, device="cuda:0")
    refinement = RefinementWorkspace(core, target)
    source = wp.clone(core.points)
    phase = wp.zeros(1, dtype=int, device="cuda:0")
    volume = wp.zeros(1, dtype=wp.float64, device="cuda:0")
    wp.load_module(module=__name__, device="cuda:0")
    wp.load_module(module="tests.test_vtm_core", device="cuda:0")
    wp.load_module(module="tests.test_active_sites", device="cuda:0")
    storage = [core.points, core.active_count, core.full_tetrahedra, refinement._keys]
    pointers = [a.ptr for a in storage]

    def forbidden(*args, **kwargs):
        raise AssertionError("Host allocation/readback/synchronization during captured work")

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
        with wp.ScopedCapture() as capture:
            wp.launch(
                produce_refinement,
                64,
                inputs=[source, core.points, core.active_count, target, phase],
            )
            core.run()
            refinement.run()
            volume.zero_()
            wp.launch(
                consume,
                len(core.full_tetrahedra),
                inputs=[
                    core.points,
                    core.full_tetrahedra,
                    refinement.tetrahedra_count,
                    refinement.error,
                    volume,
                ],
            )
            wp.launch(advance_phase, 1, inputs=[phase])
        wp.capture_launch(capture.graph)
    for step in range(3):
        if step:
            wp.capture_launch(capture.graph)
        assert [a.ptr for a in storage] == pointers
        if step == 1:
            assert refinement.error.numpy()[0] == 1
            assert refinement.tetrahedra_count.numpy()[0] == 0
            assert volume.numpy()[0] == 0
        else:
            assert refinement.error.numpy()[0] == 0
            validate_active(core, 40 + 4 * step)
            np.testing.assert_allclose(volume.numpy(), core.volume.numpy(), rtol=1e-12)
    path = tmp_path / "refinement.dot"
    wp.capture_debug_dot_print(capture.graph, str(path), verbose=True)
    graph = path.read_text()
    assert all(node not in graph for node in ("MEM_ALLOC", "MEM_FREE", "HOST"))


def test_setup_requires_active_core_and_valid_options():
    core = make_core()
    for fraction in [0, -0.1, 1.1, np.nan]:
        with pytest.raises(ValueError, match="fraction"):
            RefinementWorkspace(core, fraction=fraction)
    for target in [3, 81]:
        with pytest.raises(ValueError, match="Target count"):
            RefinementWorkspace(core, target)
    with pytest.raises(ValueError, match="Target count"):
        RefinementWorkspace(core, wp.ones(1, dtype=float, device="cuda:0"))
    core.active_count = None
    with pytest.raises(ValueError, match="active-prefix"):
        RefinementWorkspace(core)
