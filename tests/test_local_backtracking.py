"""Optional local quality retries preserve full transactions and captured recovery."""

import numpy as np
import pytest
import warp as wp

from tests.test_selective import initial, retained_oracle
from tests.test_vtm_core import valid_core
from warpower import _selective, _vtm

wp.set_module_options({"enable_backward": False, "fast_math": False})


@wp.kernel
def force_quality(enabled: wp.array[int], keep: wp.array[int], quality: wp.array[wp.float64]):
    i = wp.tid()
    if enabled[0] != 0 and keep[i] != 0:
        quality[i] = wp.float64(0)


@wp.kernel
def force_minimum(enabled: wp.array[int], minimum: wp.array[wp.float64]):
    if enabled[0] != 0:
        minimum[0] = wp.float64(0)


@pytest.fixture(scope="module")
def fitted():
    c = initial(64, 5)
    return c.domain, c.points.numpy().copy()


def test_forced_quality_failure_is_bounded_restores_and_recovers(fitted, monkeypatch):
    domain, points = fitted
    c = _vtm.VTMWorkspace(
        domain,
        points.copy(),
        max_iterations=1,
        max_backtracks=2,
        selective_threshold=1.0,
        local_quality_retries=2,
    )
    enabled = wp.ones(1, dtype=int)
    s = c.selection

    def check():
        s.evaluate()
        wp.launch(force_quality, len(s.keep), inputs=[enabled, s.keep, s.quality])
        wp.launch(force_minimum, 1, inputs=[enabled, s.minimum])
        wp.launch(
            _selective.guard_quality,
            1,
            inputs=[
                s.error,
                s.retained_count,
                s.minimum,
                s.previous_minimum,
                c._accepted,
                s.rejections,
                s.last_error,
            ],
        )
        wp.launch(
            _selective.accepted_evaluation, 1, inputs=[s.error, c._accepted, s._current_needed]
        )

    s.check = check
    wp.load_module(module=__name__, device=c.device)

    def forbidden(*args, **kwargs):
        raise AssertionError("host activity during capture/replay")

    with monkeypatch.context() as m:
        for name in [
            "empty",
            "zeros",
            "ones",
            "full",
            "empty_like",
            "clone",
            "synchronize",
            "synchronize_device",
            "synchronize_stream",
        ]:
            m.setattr(wp, name, forbidden)
        m.setattr(wp.array, "numpy", forbidden)
        with wp.ScopedCapture() as cap:
            c.local_backtracking.statistics.zero_()
            c.run()
        wp.capture_launch(cap.graph)
    result, _ = valid_core(c)
    np.testing.assert_array_equal(result.view(np.uint32), points.view(np.uint32))
    assert c.iterations.numpy()[0] == 0 and c.termination.numpy()[0] == 2
    assert 3 <= c.rejected_steps.numpy()[0] <= 5
    assert 0 < c.local_backtracking.statistics.numpy()[0] <= 2
    # Replay the same graph with the synthetic failure disabled and fresh input.
    c.points.assign(points)
    enabled.zero_()
    wp.capture_launch(cap.graph)
    result, _ = valid_core(c)
    fresh = _vtm.VTMWorkspace(
        domain,
        points.copy(),
        max_iterations=1,
        max_backtracks=2,
        selective_threshold=1.0,
        local_quality_retries=2,
    )
    fresh.run()
    expected, _ = valid_core(fresh)
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-8)
    for name in ["iterations", "termination", "rejected_steps"]:
        np.testing.assert_array_equal(getattr(c, name).numpy(), getattr(fresh, name).numpy())
    assert c.local_backtracking.used.numpy()[0] <= 2
    _, keep, q = retained_oracle(c)
    assert q[keep].min() >= s.previous_minimum.numpy()[0] * (1 - 1e-9)


def test_sticky_quality_reason_does_not_turn_objective_failure_into_local_retry(fitted):
    domain, points = fitted
    c = _vtm.VTMWorkspace(
        domain,
        points.copy(),
        max_iterations=1,
        max_backtracks=2,
        selective_threshold=1.0,
        local_quality_retries=2,
    )
    launch = c._launch

    def forced(kernel, dim, args, *, block_dim=256):
        launch(kernel, dim, args, block_dim=block_dim)
        if kernel is _vtm.candidate_ready:
            c._accepted.zero_()
            c.selection.last_error.fill_(4)

    c._launch = forced
    with wp.ScopedCapture() as cap:
        c.local_backtracking.statistics.zero_()
        c.run()
    wp.capture_launch(cap.graph)
    result, _ = valid_core(c)
    np.testing.assert_array_equal(result, points)
    assert c.rejected_steps.numpy()[0] == 3 and c.local_backtracking.statistics.numpy()[0] == 0
    assert c.selection.rejections.numpy()[0] == 0


def test_zero_budget_uses_original_transaction_schedule(fitted):
    domain, points = fitted
    baseline = _vtm.VTMWorkspace(
        domain, points.copy(), max_iterations=2, selective_threshold=1.0
    ).run()
    c = _vtm.VTMWorkspace(domain, points.copy(), max_iterations=2, selective_threshold=1.0)
    c.run()
    p, t = valid_core(c)
    bp, bt = valid_core(baseline)
    np.testing.assert_allclose(p, bp, rtol=0, atol=1e-8)
    assert {tuple(sorted(x)) for x in t} == {tuple(sorted(x)) for x in bt}
    for name in ["iterations", "termination", "rejected_steps"]:
        np.testing.assert_array_equal(getattr(c, name).numpy(), getattr(baseline, name).numpy())
    assert c.local_backtracking is None


@wp.kernel
def produce_active(phase: wp.array[int], active: wp.array[int]):
    active[0] = 64
    if phase[0] == 1:
        active[0] = 0


def test_captured_graded_prefix_producer_consumer_preserves_tail_and_recovers(
    monkeypatch, tmp_path
):
    import warpower
    from tests.test_domain import cube
    from tests.test_selective import advance
    from tests.test_tetrahedra import check_mesh
    from tests.test_vtm_core import consume

    domain = warpower.SurfaceDomain(*cube(), surface_subdivisions=6)
    source = np.full((96, 3), np.nan, np.float32)
    source[:64] = np.random.default_rng(174).uniform(-0.9, 0.9, (64, 3))
    initial_points = wp.array(source, dtype=wp.vec3)
    points = wp.clone(initial_points)
    active = wp.array([64], dtype=int)
    field = warpower.SizingField(
        np.array([[0, 0, 0]], np.float32), wp.array([0.3], dtype=float), gradation=0.4
    )
    c = _vtm.VTMWorkspace(
        domain,
        points,
        active_count=active,
        sizing=field,
        max_iterations=3,
        selective_threshold=0.3,
        local_quality_retries=4,
    )
    phase = wp.zeros(1, dtype=int)
    volume = wp.zeros(1, dtype=wp.float64)
    wp.load_module(module=__name__, device=c.device)
    wp.load_module(module="tests.test_vtm_core", device=c.device)
    wp.load_module(module="tests.test_selective", device=c.device)

    def forbidden(*args, **kwargs):
        raise AssertionError("host activity during capture/replay")

    with monkeypatch.context() as m:
        for name in [
            "empty",
            "zeros",
            "ones",
            "full",
            "empty_like",
            "clone",
            "synchronize",
            "synchronize_device",
            "synchronize_stream",
        ]:
            m.setattr(wp, name, forbidden)
        m.setattr(wp.array, "numpy", forbidden)
        with wp.ScopedCapture() as cap:
            wp.copy(points, initial_points)
            wp.launch(produce_active, 1, inputs=[phase, active])
            c.run()
            volume.zero_()
            wp.launch(
                consume,
                len(c.full_tetrahedra),
                inputs=[points, c.full_tetrahedra, c.full_tetrahedra_count, c.error, volume],
            )
            wp.launch(advance, 1, inputs=[phase])
        wp.capture_launch(cap.graph)
    for step in range(3):
        if step:
            wp.capture_launch(cap.graph)
        actual = points.numpy()
        np.testing.assert_array_equal(actual[64:].view(np.uint32), source[64:].view(np.uint32))
        if step == 1:
            assert (
                c.error.numpy()[0] == 8
                and c.full_tetrahedra_count.numpy()[0] == 0
                and volume.numpy()[0] == 0
            )
            assert (
                not c.local_backtracking.statistics.numpy().any()
                and not c.local_backtracking.blocked.numpy().any()
            )
        else:
            assert c.error.numpy()[0] == 0 and not c.selection.mask.numpy()[64:].any()
            tets = c.full_tetrahedra.numpy()[: c.full_tetrahedra_count.numpy()[0]]
            assert tets.max() < 64
            check_mesh(actual[:64], tets)
            corners = c.boundary.corner_sites.numpy()[:8]
            np.testing.assert_array_equal(actual[corners], domain.corner_points.numpy()[:8])
            np.testing.assert_allclose(volume.numpy(), c.volume.numpy(), rtol=1e-12)
    dot = tmp_path / "local-backtracking.dot"
    wp.capture_debug_dot_print(cap.graph, str(dot), verbose=True)
    assert all(name not in dot.read_text() for name in ["MEM_ALLOC", "MEM_FREE", "HOST"])


@pytest.mark.parametrize("seed", [2026, 7])
def test_optional_public_mesher_preserves_topology_and_corners(seed):
    import warpower
    from tests.test_domain import cube
    from tests.test_mesher import validate

    domain = warpower.SurfaceDomain(*cube(), surface_subdivisions=4)
    mesh = warpower.VariationalMeshingWorkspace(
        domain, 128, initial_count=64, quality_rounds=1, local_quality_retries=4, seed=seed
    ).run()
    validate(mesh)
    assert mesh.core.local_backtracking is not None and mesh.rounds.numpy()[2] == 1


def test_budget_validation_and_disabled_default(fitted):
    import warpower

    domain, points = fitted
    for budget in [-1, 2**31]:
        with pytest.raises(ValueError, match="local_quality_retries"):
            _vtm.VTMWorkspace(domain, points, selective_threshold=0.3, local_quality_retries=budget)
        with pytest.raises(ValueError, match="local_quality_retries"):
            warpower.VariationalMeshingWorkspace(domain, 64, local_quality_retries=budget)
    with pytest.raises(ValueError, match="requires selective_threshold"):
        _vtm.VTMWorkspace(domain, points, local_quality_retries=1)
    with pytest.raises(ValueError, match="requires quality_rounds"):
        warpower.VariationalMeshingWorkspace(domain, 64, quality_rounds=0, local_quality_retries=1)
    with pytest.raises(TypeError):
        _vtm.VTMWorkspace(domain, points, local_quality_retries=0.5)
    core = _vtm.VTMWorkspace(domain, points, selective_threshold=0.3, local_quality_retries=0)
    assert (
        core.local_backtracking is None
        and core._propose is _vtm.propose
        and core._propose_args == []
    )
