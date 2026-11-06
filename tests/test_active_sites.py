import numpy as np
import warp as wp

import warpower
from tests.test_domain import cube, torus
from tests.test_tetrahedra import check_mesh, reference, tet_set
from tests.test_vtm_core import consume, fail_after_initial, torus_sites
from warpower._vtm import VTMWorkspace


def validate_active(workspace, active):
    assert workspace.error.numpy()[0] == 0, workspace.error.numpy()
    assert workspace.mesh.error.numpy()[0] == 0
    assert workspace.boundary.error.numpy()[0] == 0
    points = workspace.points.numpy()[:active]
    count = int(workspace.full_tetrahedra_count.numpy()[0])
    tets = workspace.full_tetrahedra.numpy()[:count]
    assert count > 0 and tets.min() >= 0 and tets.max() < active
    check_mesh(points, tets)
    assert tet_set(tets) == reference(points)
    scratch = workspace.mesh.points.numpy()
    np.testing.assert_array_equal(scratch[:active], points)
    np.testing.assert_array_equal(scratch[active:], np.tile(points[0], (len(scratch) - active, 1)))
    np.testing.assert_array_equal(workspace.boundary.roles.numpy()[active:], 0)
    assert workspace.boundary.role_counts.numpy().sum() == active
    return points, tets


def test_active_prefix_grows_and_shrinks_without_consuming_or_changing_inactive_storage():
    domain = warpower.SurfaceDomain(*torus(24, 12), surface_subdivisions=2)
    source = torus_sites(96, 421)
    points = wp.array(source, dtype=wp.vec3, device="cuda:0")
    active = wp.array([24], dtype=int, device="cuda:0")
    workspace = VTMWorkspace(domain, points, active_count=active, max_iterations=2)
    for n in [24, 70, 40, 96, 32]:
        data = source.copy()
        data[n:] = np.array([np.nan, np.inf, -1e30])
        points.assign(data)
        active.fill_(n)
        workspace.run()
        validate_active(workspace, n)
        np.testing.assert_array_equal(points.numpy()[n:].view(np.uint32), data[n:].view(np.uint32))


def test_full_active_budget_agrees_with_existing_fixed_budget_optimization():
    domain = warpower.SurfaceDomain(*torus(24, 12), surface_subdivisions=2)
    source = torus_sites(80, 120)
    fixed = VTMWorkspace(domain, source.copy(), max_iterations=4).run()
    count = wp.array([len(source)], dtype=int, device="cuda:0")
    active = VTMWorkspace(domain, source.copy(), active_count=count, max_iterations=4).run()
    points, tets = validate_active(active, len(source))
    assert fixed.error.numpy()[0] == 0
    assert fixed.iterations.numpy()[0] == active.iterations.numpy()[0] == 4
    np.testing.assert_allclose(points, fixed.points.numpy(), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(
        active.energy_history.numpy(), fixed.energy_history.numpy(), rtol=1e-7
    )
    assert tet_set(tets) == tet_set(
        fixed.full_tetrahedra.numpy()[: fixed.full_tetrahedra_count.numpy()[0]]
    )


def test_invalid_device_counts_publish_no_mesh_and_recover():
    source = torus_sites(48, 84)
    domain = warpower.SurfaceDomain(*torus(24, 12))
    active = wp.array([48], dtype=int, device="cuda:0")
    workspace = VTMWorkspace(domain, source, active_count=active, max_iterations=1)
    for n in [48, -1, 0, 1, 3, 49, 24]:
        active.fill_(n)
        before = workspace.points.numpy()
        workspace.run()
        if n < 4 or n > 48:
            assert workspace.error.numpy()[0] == 8
            assert workspace.full_tetrahedra_count.numpy()[0] == 0
            assert workspace.iterations.numpy()[0] == 0
            assert workspace.energy.numpy()[0] == workspace.volume.numpy()[0] == 0
            assert np.all(np.isnan(workspace.energy_history.numpy()))
            np.testing.assert_array_equal(workspace.points.numpy(), before)
        else:
            validate_active(workspace, n)


def test_active_sites_respect_corner_budget_and_preserve_exact_anchors():
    domain = warpower.SurfaceDomain(*cube(), surface_subdivisions=4)
    source = np.random.default_rng(483).uniform(-0.8, 0.8, (40, 3)).astype(np.float32)
    active = wp.array([16], dtype=int, device="cuda:0")
    workspace = VTMWorkspace(domain, source, active_count=active, max_iterations=1)
    for n in [16, 4, 40]:
        workspace.points.assign(source)
        active.fill_(n)
        workspace.run()
        if n == 4:
            assert workspace.error.numpy()[0] & 2
            assert workspace.boundary.error.numpy()[0] & 4
            assert workspace.full_tetrahedra_count.numpy()[0] == 0
        else:
            assert workspace.error.numpy()[0] == 0
            points = workspace.points.numpy()
            corners = workspace.boundary.corner_sites.numpy()[:8]
            assert np.all(corners < n) and len(np.unique(corners)) == 8
            np.testing.assert_array_equal(points[corners], domain.corner_points.numpy()[:8])
            tets = workspace.full_tetrahedra.numpy()[: workspace.full_tetrahedra_count.numpy()[0]]
            assert tets.max() < n
            check_mesh(points[:n], tets)
            np.testing.assert_array_equal(points[n:], source[n:])


@wp.kernel
def produce_prefix(
    source: wp.array[wp.vec3],
    points: wp.array[wp.vec3],
    active: wp.array[int],
    phase: wp.array[int],
):
    i = wp.tid()
    count = 24 + 16 * phase[0]
    if phase[0] == 2:
        count = 0
    if i == 0:
        active[0] = count
    p = wp.vec3(wp.nan, wp.inf, -1.0e30)
    if i < count:
        p = source[i] + wp.vec3(0.001 * float(phase[0]), 0.0, 0.0)
    points[i] = p


@wp.kernel
def advance_phase(phase: wp.array[int]):
    phase[0] += 1


def test_captured_device_producer_controls_counts_and_consumer_recovers(monkeypatch, tmp_path):
    source = wp.array(torus_sites(80, 41), dtype=wp.vec3, device="cuda:0")
    points = wp.clone(source)
    active = wp.array([24], dtype=int, device="cuda:0")
    phase = wp.zeros(1, dtype=int, device="cuda:0")
    volume = wp.zeros(1, dtype=wp.float64, device="cuda:0")
    domain = warpower.SurfaceDomain(*torus(24, 12))
    workspace = VTMWorkspace(domain, points, active_count=active, max_iterations=2)
    wp.load_module(module=__name__, device="cuda:0")
    wp.load_module(module="tests.test_vtm_core", device="cuda:0")
    storage = [points, active, workspace.mesh.points, workspace.full_tetrahedra]
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
            wp.launch(produce_prefix, len(points), inputs=[source, points, active, phase])
            workspace.run()
            volume.zero_()
            wp.launch(
                consume,
                len(workspace.full_tetrahedra),
                inputs=[
                    points,
                    workspace.full_tetrahedra,
                    workspace.full_tetrahedra_count,
                    workspace.error,
                    volume,
                ],
            )
            wp.launch(advance_phase, 1, inputs=[phase])
        wp.capture_launch(capture.graph)
    for step in range(4):
        if step:
            wp.capture_launch(capture.graph)
        assert [a.ptr for a in storage] == pointers
        if step == 2:
            assert workspace.error.numpy()[0] == 8
            assert workspace.full_tetrahedra_count.numpy()[0] == 0
            assert volume.numpy()[0] == 0
        else:
            validate_active(workspace, 24 + 16 * step)
            np.testing.assert_allclose(volume.numpy(), workspace.volume.numpy(), rtol=1e-12)
    path = tmp_path / "active-sites.dot"
    wp.capture_debug_dot_print(capture.graph, str(path), verbose=True)
    graph = path.read_text()
    assert all(node not in graph for node in ("MEM_ALLOC", "MEM_FREE", "HOST"))


def test_active_rollback_restores_mesh_scratch_and_preserves_unused_points():
    source = torus_sites(80, 82)
    source[40:] = np.nan
    domain = warpower.SurfaceDomain(*torus(24, 12))
    active = wp.array([40], dtype=int, device="cuda:0")
    workspace = VTMWorkspace(
        domain, source, active_count=active, max_iterations=2, max_backtracks=1
    )
    calls = wp.zeros(1, dtype=int, device="cuda:0")
    real_run = workspace.mesh.run

    def failing_run(stream=None):
        real_run(stream)
        wp.launch(
            fail_after_initial,
            1,
            inputs=[workspace.mesh.error, workspace.full_tetrahedra_count, calls],
        )

    workspace.mesh.run = failing_run
    wp.load_module(module="tests.test_vtm_core", device="cuda:0")
    with wp.ScopedCapture() as capture:
        calls.zero_()
        workspace.run()
    wp.capture_launch(capture.graph)
    validate_active(workspace, 40)
    np.testing.assert_array_equal(workspace.points.numpy().view(np.uint32), source.view(np.uint32))
    assert workspace.iterations.numpy()[0] == 0
    assert workspace.rejected_steps.numpy()[0] == 2
    assert workspace.termination.numpy()[0] == 2


def test_future_active_positions_survive_smaller_prefix_optimization():
    source = torus_sites(64, 1201)
    domain = warpower.SurfaceDomain(*torus(24, 12))
    active = wp.array([32], dtype=int, device="cuda:0")
    workspace = VTMWorkspace(domain, source.copy(), active_count=active, max_iterations=2).run()
    validate_active(workspace, 32)
    np.testing.assert_array_equal(workspace.points.numpy()[32:], source[32:])
    active.fill_(64)
    workspace.run()
    validate_active(workspace, 64)


@wp.kernel
def set_sampling_field(values: wp.array[float], phase: wp.array[int]):
    values[0] = 0.7 + 0.1 * float(phase[0])
    if phase[0] == 1:
        values[0] = -1.0


@wp.kernel
def copy_sampled_sites(source: wp.array[wp.vec3], count: wp.array[int], target: wp.array[wp.vec3]):
    i = wp.tid()
    if i < count[0]:
        target[i] = source[i]


def test_sampler_device_count_drives_graded_core_and_propagates_failure():
    domain = warpower.SurfaceDomain(*torus(24, 12))
    sizes = wp.ones(1, device="cuda:0")
    field = warpower.SizingField(np.array([[2, 0, 0]], np.float32), sizes, gradation=0.3)
    sampler = warpower.InitialSamplingWorkspace(domain, 48, sizing=field, resolution=8)
    points = wp.full(96, wp.vec3(wp.nan), dtype=wp.vec3, device="cuda:0")
    workspace = VTMWorkspace(
        domain, points, active_count=sampler.count, sizing=field, max_iterations=2
    )
    phase = wp.zeros(1, dtype=int, device="cuda:0")
    wp.load_module(module=__name__, device="cuda:0")
    with wp.ScopedCapture() as capture:
        wp.launch(set_sampling_field, 1, inputs=[sizes, phase])
        field.refit()
        sampler.run()
        wp.launch(copy_sampled_sites, len(points), inputs=[sampler.points, sampler.count, points])
        workspace.run()
        wp.launch(advance_phase, 1, inputs=[phase])
    for step in range(3):
        wp.capture_launch(capture.graph)
        if step == 1:
            assert sampler.error.numpy()[0] == 2
            assert sampler.count.numpy()[0] == 0
            assert workspace.error.numpy()[0] == 8
            assert workspace.full_tetrahedra_count.numpy()[0] == 0
        else:
            assert sampler.error.numpy()[0] == 0
            validate_active(workspace, 48)
            assert workspace.iterations.numpy()[0] == 2
        assert np.all(np.isnan(points.numpy()[48:]))
