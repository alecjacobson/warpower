import numpy as np
import pytest
import warp as wp

import warpower
from tests.test_delaunay_recovery import exact_empty
from tests.test_domain import combine, cube, torus
from tests.test_extraction import independent_topology
from tests.test_tetrahedra import check_mesh, reference
from tests.test_vtm_core import consume


def validate(mesh):
    assert mesh.error.numpy()[0] == 0 and mesh.valid.numpy()[0] == 1
    n, full_count, count, boundary_count = mesh.counts.numpy()
    assert n == mesh.target_count.numpy()[0]
    assert full_count >= count > 0 and boundary_count > 0
    p = mesh.points.numpy()[:n]
    full = mesh.full_tetrahedra.numpy()[:full_count]
    assert full.max() < n
    check_mesh(p, full)
    expected = reference(p)
    differing = np.array([tet for tet in full if tuple(sorted(tet)) not in expected], np.int32)
    if len(differing):
        exact_empty(p, differing)
    kept = mesh.tetrahedra.numpy()[:count]
    _, components, chi = independent_topology(p, kept)
    actual = mesh.topology.numpy()
    assert components == actual[0] and chi == actual[3]
    np.testing.assert_array_equal(actual, mesh.input_topology.counts.numpy())
    assert mesh.extraction.valid.numpy()[0] == 1
    np.testing.assert_allclose(mesh.volume.numpy(), mesh.extraction.volume.numpy(), rtol=1e-12)
    np.testing.assert_allclose(
        mesh.minimum_radius_ratio.numpy()[0], mesh.radius_ratio.numpy()[:count].min(), rtol=1e-12
    )
    corners = mesh.core.boundary.corner_sites.numpy()[: mesh.domain.corner_count.numpy()[0]]
    np.testing.assert_array_equal(p[corners], mesh.domain.corner_points.numpy()[: len(corners)])
    return p, kept


@pytest.mark.parametrize("case", ["cube", "torus", "cavity", "disconnected"])
def test_complete_pipeline_samples_grows_fits_and_preserves_domain_topology(case):
    if case == "cube":
        surface, n, initial, topology = cube(), 128, 64, [1, 1, 0, 2]
    elif case == "torus":
        surface, n, initial, topology = torus(24, 12), 256, 64, [1, 1, 0, 0]
    elif case == "cavity":
        inner = cube(0.4)
        surface = combine(cube(), (inner[0], inner[1][:, ::-1].copy()))
        n, initial, topology = 512, 128, [1, 2, 1, 4]
    else:
        surface = combine(cube(0.7, (-2, 0, 0)), cube(0.7, (2, 0, 0)))
        n, initial, topology = 256, 64, [2, 2, 0, 4]
    domain = warpower.SurfaceDomain(*surface, surface_subdivisions=4)
    mesh = warpower.VariationalMeshingWorkspace(
        domain, n, initial_count=initial, quality_rounds=1
    ).run()
    validate(mesh)
    np.testing.assert_array_equal(mesh.topology.numpy(), topology)
    assert mesh.rounds.numpy()[0] >= 2
    assert mesh.steps.numpy()[0] >= 20
    assert mesh.sampler.count.numpy()[0] == initial
    assert mesh.point_count.numpy()[0] == n
    assert mesh.rounds.numpy()[2] == 1


def test_quality_stages_improve_complete_pipeline_worst_element():
    d = warpower.SurfaceDomain(*cube(), surface_subdivisions=4)
    base = warpower.VariationalMeshingWorkspace(d, 128, initial_count=64, quality_rounds=0).run()
    improved = warpower.VariationalMeshingWorkspace(
        d, 128, initial_count=64, quality_rounds=1
    ).run()
    validate(base)
    validate(improved)
    assert base.jitter is None and base.interior is None and base.core.selection is None
    assert improved.minimum_radius_ratio.numpy()[0] > 10 * base.minimum_radius_ratio.numpy()[0]
    assert improved.moves.numpy()[0] > 0


def test_fixed_budget_skips_growth_and_automatic_sizing_composes():
    d = warpower.SurfaceDomain(*cube(), surface_subdivisions=4)
    fixed = warpower.VariationalMeshingWorkspace(d, 128, initial_count=128, quality_rounds=0).run()
    validate(fixed)
    assert fixed.refinement is None and fixed.rounds.numpy()[0] == 0
    graded = warpower.VariationalMeshingWorkspace(
        d, 128, initial_count=64, quality_rounds=1, automatic_sizing=True
    ).run()
    validate(graded)
    assert graded.automatic.error.numpy()[0] == 0
    assert graded.automatic.pole_count.numpy()[0] > 0
    assert graded.sizing is graded.automatic.field


def test_budget_failures_publish_zero_counts_and_metrics():
    d = warpower.SurfaceDomain(*cube(), surface_subdivisions=2)
    for options, expected in [
        (dict(initial_count=4), 1),
        (dict(initial_count=32, max_growth_rounds=0), 32),
        (dict(initial_count=32, max_growth_rounds=1), 32),
    ]:
        mesh = warpower.VariationalMeshingWorkspace(d, 64, quality_rounds=0, **options).run()
        assert mesh.error.numpy()[0] == expected and mesh.status.numpy()[0] == 3
        np.testing.assert_array_equal(mesh.counts.numpy(), 0)
        np.testing.assert_array_equal(mesh.metrics.numpy(), 0)
        assert mesh.valid.numpy()[0] == 0


def test_inadequate_topology_is_not_published_and_recovery_is_bounded():
    d = warpower.SurfaceDomain(*torus(24, 12))
    mesh = warpower.VariationalMeshingWorkspace(
        d, 4, initial_count=4, polish_iterations=0, max_recovery_rounds=2, quality_rounds=0
    ).run()
    assert mesh.core.error.numpy()[0] == 0
    assert mesh.error.numpy()[0] == 64 and mesh.extraction_reason.numpy()[0] != 0
    assert mesh.rounds.numpy()[1] == 2
    np.testing.assert_array_equal(mesh.counts.numpy(), 0)
    np.testing.assert_array_equal(mesh.topology.numpy(), 0)


@pytest.mark.parametrize("recovery", [0, 3])
def test_coarse_graded_nonmanifold_boundary_recovers_with_additional_fitting(recovery):
    d = warpower.SurfaceDomain(*cube(), surface_subdivisions=6)
    field = warpower.SizingField(
        np.array([[-0.8, 0, 0], [0.8, 0, 0]], np.float32),
        wp.array([0.2, 0.5], dtype=float, device=d.device),
        gradation=0.3,
    )
    source = wp.array(
        np.random.default_rng(2026).uniform(-0.9, 0.9, (128, 3)).astype(np.float32),
        dtype=wp.vec3,
        device=d.device,
    )
    mesh = warpower.VariationalMeshingWorkspace(
        d,
        136,
        target_count=128,
        initial_count=128,
        sizing=field,
        polish_iterations=5,
        quality_rounds=0,
        max_recovery_rounds=recovery,
    )

    def reproduce_sampling(stream=None):
        wp.copy(mesh.sampler.points, source)
        mesh.sampler.count.fill_(128)
        mesh.sampler.error.zero_()

    # Keep the original failing point set independent of sampler changes.
    mesh.sampler.run = reproduce_sampling
    with wp.ScopedCapture() as cap:
        mesh.run()
    wp.capture_launch(cap.graph)
    if recovery == 0:
        assert mesh.core.error.numpy()[0] == mesh.extraction.error.numpy()[0] == 0
        assert mesh.extraction.diagnostics.numpy()[0] == 2
        assert mesh.error.numpy()[0] == 64
        np.testing.assert_array_equal(mesh.counts.numpy(), 0)
    else:
        validate(mesh)
        assert mesh.rounds.numpy()[1] == 1
        assert mesh.steps.numpy()[0] == 10
        assert mesh.extraction.diagnostics.numpy()[0] == 0


def test_invalid_sizing_and_sampling_failures_gate_outputs_and_recover():
    d = warpower.SurfaceDomain(*cube(), surface_subdivisions=4)
    values = wp.array([0.3], dtype=float, device=d.device)
    field = warpower.SizingField(np.zeros((1, 3), np.float32), values, gradation=0)
    mesh = warpower.VariationalMeshingWorkspace(
        d, 128, initial_count=64, sizing=field, quality_rounds=0
    )
    with wp.ScopedCapture() as cap:
        field.refit()
        mesh.run()
    values.fill_(-1)
    wp.capture_launch(cap.graph)
    assert mesh.error.numpy()[0] == 2 and mesh.last_errors.numpy()[0] == 2
    np.testing.assert_array_equal(mesh.counts.numpy(), 0)
    values.fill_(0.3)
    wp.capture_launch(cap.graph)
    validate(mesh)

    def failed_sampling(stream=None):
        mesh.sampler.error.fill_(8)
        mesh.sampler.count.zero_()

    mesh.sampler.run = failed_sampling
    with wp.ScopedCapture() as failed:
        mesh.run()
    wp.capture_launch(failed.graph)
    assert mesh.error.numpy()[0] == 4 and mesh.last_errors.numpy()[1] == 8
    np.testing.assert_array_equal(mesh.counts.numpy(), 0)
    wp.capture_launch(cap.graph)
    validate(mesh)


def test_public_setup_contract_rejects_incompatible_arguments():
    d = warpower.SurfaceDomain(*cube())
    for options in [
        dict(initial_count=3),
        dict(initial_count=32, target_count=16),
        dict(target_count=wp.zeros(2, dtype=int, device=d.device)),
        dict(target_count=wp.zeros(1, dtype=float, device=d.device)),
        dict(seed=2**31),
        dict(quality_threshold=float("nan")),
        dict(jitter_attempts=0),
        dict(polish_iterations=-1),
        dict(max_recovery_rounds=2**31),
        dict(automatic_sizing=True, sizing=object()),
    ]:
        with pytest.raises(ValueError):
            warpower.VariationalMeshingWorkspace(d, 64, quality_rounds=0, **options)


@wp.kernel
def fail_batch(requested: wp.array[int], error: wp.array[int], count: wp.array[int]):
    if requested[0] > 1:
        error[0] = 2
        count[0] = 0


def test_failed_insertion_batch_retries_one_site_and_reaches_target():
    d = warpower.SurfaceDomain(*cube(), surface_subdivisions=4)
    mesh = warpower.VariationalMeshingWorkspace(d, 66, initial_count=64, quality_rounds=0)
    original_run = mesh.core.mesh.run
    wp.load_module(module=__name__, device=d.device)

    def failing_batch(stream=None):
        original_run(stream)
        wp.launch(
            fail_batch,
            1,
            inputs=[
                mesh.refinement.requested,
                mesh.core.mesh.error,
                mesh.core.full_tetrahedra_count,
            ],
        )

    mesh.core.mesh.run = failing_batch
    with wp.ScopedCapture() as cap:
        mesh.run()
    wp.capture_launch(cap.graph)
    validate(mesh)
    assert mesh.refinement_rejections.numpy()[0] == 1
    assert mesh.rounds.numpy()[0] == 2


@wp.kernel
def corrupt_quality(
    mode: wp.array[int],
    points: wp.array[wp.vec3],
    source_error: wp.array[int],
    core_error: wp.array[int],
    count: wp.array[int],
):
    if mode[0] == 1:
        points[0] = wp.vec3(12345.0)
        source_error[0] = 8
        core_error[0] = 8
        count[0] = 0


def test_failed_optional_stages_restore_last_valid_mesh():
    d = warpower.SurfaceDomain(*cube(), surface_subdivisions=4)
    mesh = warpower.VariationalMeshingWorkspace(d, 128, initial_count=64, quality_rounds=1)
    wp.load_module(module=__name__, device=d.device)
    for stage in (mesh.core, mesh.jitter, mesh.interior):
        original = stage.run

        def failing(stream=None, stage=stage, original=original):
            original(stream)
            wp.launch(
                corrupt_quality,
                1,
                inputs=[
                    mesh._mode,
                    mesh.points,
                    stage.error,
                    mesh.core.error,
                    mesh.core.full_tetrahedra_count,
                ],
            )

        stage.run = failing
    with wp.ScopedCapture() as cap:
        mesh.run()
    wp.capture_launch(cap.graph)
    validate(mesh)
    assert mesh.quality_rejections.numpy()[0] == 3
    np.testing.assert_array_equal(mesh.moves.numpy(), 0)
    np.testing.assert_array_equal(mesh.points.numpy(), mesh._old_points.numpy())
    assert mesh.status.numpy()[0] == 2
    np.testing.assert_allclose(mesh.core.energy_history.numpy()[0, 0], mesh.core.energy.numpy()[0])


@wp.kernel
def fail_boundary_in_quality(mode: wp.array[int], flag: wp.array[int], error: wp.array[int]):
    if mode[0] == 1 and flag[0] != 0:
        error[0] = 1


def test_failed_restore_publishes_nothing_and_later_replay_recovers():
    d = warpower.SurfaceDomain(*cube(), surface_subdivisions=4)
    mesh = warpower.VariationalMeshingWorkspace(d, 128, initial_count=64, quality_rounds=1)
    flag = wp.ones(1, dtype=int, device=d.device)
    original = mesh.core.boundary.run
    wp.load_module(module=__name__, device=d.device)

    def failing(stream=None):
        original(stream)
        wp.launch(fail_boundary_in_quality, 1, inputs=[mesh._mode, flag, mesh.core.boundary.error])

    mesh.core.boundary.run = failing
    with wp.ScopedCapture() as cap:
        mesh.run()
    wp.capture_launch(cap.graph)
    assert mesh.error.numpy()[0] & 128 and mesh.valid.numpy()[0] == 0
    np.testing.assert_array_equal(mesh.counts.numpy(), 0)
    flag.zero_()
    wp.capture_launch(cap.graph)
    validate(mesh)


@wp.kernel
def produce(
    source: wp.array[wp.vec3],
    points: wp.array[wp.vec3],
    phase: wp.array[int],
    target: wp.array[int],
    seed: wp.array[int],
):
    i = wp.tid()
    p = source[i] * (1.0 + 0.05 * float(phase[0]))
    if phase[0] == 3:
        p = wp.vec3(0.0)
    points[i] = p
    if i == 0:
        target[0] = 96
        if phase[0] == 1:
            target[0] = 128
        elif phase[0] == 2:
            target[0] = 63
        seed[0] = 2026 + phase[0]


def test_full_device_producer_mesher_consumer_with_changing_input_and_recovery(
    monkeypatch, tmp_path
):
    d = warpower.SurfaceDomain(*cube(), surface_subdivisions=4)
    source = wp.clone(d.points)
    target = wp.full(1, 96, dtype=int, device=d.device)
    seed = wp.full(1, 2026, dtype=int, device=d.device)
    phase = wp.zeros(1, dtype=int, device=d.device)
    volume = wp.zeros(1, dtype=wp.float64, device=d.device)
    mesh = warpower.VariationalMeshingWorkspace(
        d, 128, target_count=target, initial_count=64, seed=seed, quality_rounds=1
    )
    wp.load_module(module=__name__, device=d.device)
    wp.load_module(module="tests.test_vtm_core", device=d.device)

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
            wp.launch(produce, len(source), inputs=[source, d.points, phase, target, seed])
            d.refit()
            mesh.run()
            volume.zero_()
            wp.launch(
                consume,
                len(mesh.tetrahedra),
                inputs=[mesh.points, mesh.tetrahedra, mesh.tetrahedra_count, mesh.error, volume],
            )
        wp.capture_launch(cap.graph)
    pointers = [mesh.points.ptr, mesh.tetrahedra.ptr, mesh.tetrahedra_count.ptr]
    for value in [0, 1, 2, 3, 4, 0]:
        phase.fill_(value)
        wp.capture_launch(cap.graph)
        if value in (2, 3):
            assert mesh.error.numpy()[0] == 1
            np.testing.assert_array_equal(mesh.counts.numpy(), 0)
            assert volume.numpy()[0] == 0
        else:
            validate(mesh)
            np.testing.assert_allclose(volume.numpy(), mesh.volume.numpy(), rtol=1e-12)
        assert pointers == [mesh.points.ptr, mesh.tetrahedra.ptr, mesh.tetrahedra_count.ptr]
    path = tmp_path / "mesher.dot"
    wp.capture_debug_dot_print(cap.graph, str(path), verbose=True)
    assert all(node not in path.read_text() for node in ("MEM_ALLOC", "MEM_FREE", "HOST"))
