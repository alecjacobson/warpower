import numpy as np
import pytest
import warp as wp

import warpower
from tests.test_domain import cube
from tests.test_selective import initial
from tests.test_tetrahedra import check_mesh, tet_set
from tests.test_vtm_core import consume, valid_core
from warpower._vtm import VTMWorkspace


@pytest.mark.parametrize("budget", [0, 1, 3])
@pytest.mark.parametrize("selective", [0, 1])
def test_device_controls_match_static_budget_and_mode(budget, selective):
    source = initial(128, 8)
    mode = wp.full(1, selective, dtype=int, device=source.device)
    limit = wp.full(1, budget, dtype=int, device=source.device)
    expected = VTMWorkspace(
        source.domain,
        wp.clone(source.points),
        max_iterations=budget,
        selective_threshold=0.3 if selective else None,
    ).run()
    actual = VTMWorkspace(
        source.domain,
        wp.clone(source.points),
        max_iterations=5,
        iteration_limit=limit,
        selective_threshold=0.3,
        selective_enabled=mode,
    ).run()
    assert actual.error.numpy()[0] == expected.error.numpy()[0] == 0
    assert actual.iterations.numpy()[0] <= budget
    for name in ("iterations", "termination", "rejected_steps", "full_tetrahedra_count"):
        np.testing.assert_array_equal(
            getattr(actual, name).numpy(), getattr(expected, name).numpy()
        )
    np.testing.assert_array_equal(actual.points.numpy(), expected.points.numpy())
    count = actual.full_tetrahedra_count.numpy()[0]
    assert tet_set(actual.full_tetrahedra.numpy()[:count]) == tet_set(
        expected.full_tetrahedra.numpy()[:count]
    )
    np.testing.assert_allclose(
        actual.energy_history.numpy()[: budget + 1], expected.energy_history.numpy(), rtol=1e-12
    )
    assert np.isnan(actual.energy_history.numpy()[budget + 1 :]).all()
    if not selective:
        assert actual.selection.retained_count.numpy()[0] == 0
        assert actual.selection.rejections.numpy()[0] == 0


@wp.kernel
def produce_controls(
    source: wp.array[wp.vec3],
    points: wp.array[wp.vec3],
    phase: wp.array[int],
    budget: wp.array[int],
    mode: wp.array[int],
):
    i = wp.tid()
    points[i] = source[i]
    if i == 0:
        budget[0] = 2
        mode[0] = 0
        if phase[0] == 0:
            budget[0] = 0
        elif phase[0] == 2:
            budget[0] = 6
        elif phase[0] == 3:
            mode[0] = 1
        elif phase[0] == 4:
            mode[0] = -1
        elif phase[0] == 5:
            budget[0] = 1


def test_device_producer_changes_budget_and_mode_with_invalid_recovery(monkeypatch, tmp_path):
    fitted = initial(128, 8)
    source = wp.clone(fitted.points)
    budget = wp.zeros(1, dtype=int, device=fitted.device)
    mode = wp.zeros(1, dtype=int, device=fitted.device)
    phase = wp.zeros(1, dtype=int, device=fitted.device)
    volume = wp.zeros(1, dtype=wp.float64, device=fitted.device)
    core = VTMWorkspace(
        fitted.domain,
        wp.clone(source),
        max_iterations=5,
        iteration_limit=budget,
        selective_threshold=0.3,
        selective_enabled=mode,
    )
    wp.load_module(module=__name__, device=core.device)
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
            wp.launch(produce_controls, core.n, inputs=[source, core.points, phase, budget, mode])
            core.run()
            volume.zero_()
            wp.launch(
                consume,
                len(core.full_tetrahedra),
                inputs=[
                    core.points,
                    core.full_tetrahedra,
                    core.full_tetrahedra_count,
                    core.error,
                    volume,
                ],
            )
        wp.capture_launch(cap.graph)
    for value in range(6):
        phase.fill_(value)
        wp.capture_launch(cap.graph)
        if value in (2, 4):
            assert core.error.numpy()[0] == 32
            assert core.full_tetrahedra_count.numpy()[0] == volume.numpy()[0] == 0
            assert core.iterations.numpy()[0] == 0
            np.testing.assert_array_equal(core.points.numpy(), source.numpy())
        else:
            assert core.error.numpy()[0] == 0
            assert core.iterations.numpy()[0] <= budget.numpy()[0]
            if value in (1, 5):
                assert core.iterations.numpy()[0] == budget.numpy()[0]
            np.testing.assert_allclose(volume.numpy(), core.volume.numpy(), rtol=1e-12)
            valid_core(core)
    path = tmp_path / "controls.dot"
    wp.capture_debug_dot_print(cap.graph, str(path), verbose=True)
    assert all(node not in path.read_text() for node in ("MEM_ALLOC", "MEM_FREE", "HOST"))


def test_controls_compose_with_graded_active_prefix_and_corner_constraints():
    domain = warpower.SurfaceDomain(*cube(), surface_subdivisions=6)
    points = np.full((96, 3), np.nan, np.float32)
    points[:64] = np.random.default_rng(174).uniform(-0.9, 0.9, (64, 3))
    count = wp.full(1, 64, dtype=int, device="cuda:0")
    mode = wp.zeros(1, dtype=int, device="cuda:0")
    budget = wp.full(1, 2, dtype=int, device="cuda:0")
    field = warpower.SizingField(
        np.array([[0, 0, 0]], np.float32),
        wp.array([0.3], dtype=float, device="cuda:0"),
        gradation=0.4,
    )
    core = VTMWorkspace(
        domain,
        points,
        active_count=count,
        sizing=field,
        max_iterations=5,
        iteration_limit=budget,
        selective_threshold=0.3,
        selective_enabled=mode,
    )
    for choice in [0, 1, 0]:
        mode.fill_(choice)
        core.run()
        assert core.error.numpy()[0] == 0 and core.iterations.numpy()[0] <= 2
        p = core.points.numpy()
        np.testing.assert_array_equal(p[64:].view(np.uint32), points[64:].view(np.uint32))
        corners = core.boundary.corner_sites.numpy()[:8]
        np.testing.assert_array_equal(p[corners], domain.corner_points.numpy()[:8])
        tets = core.full_tetrahedra.numpy()[: core.full_tetrahedra_count.numpy()[0]]
        assert tets.max() < 64
        check_mesh(p[:64], tets)


def test_iteration_control_alone_and_mode_control_alone_are_supported():
    fitted = initial(64, 3)
    limit = wp.full(1, 1, dtype=int, device=fitted.device)
    core = VTMWorkspace(
        fitted.domain, wp.clone(fitted.points), max_iterations=5, iteration_limit=limit
    ).run()
    assert core.error.numpy()[0] == 0 and core.iterations.numpy()[0] == 1
    assert core.selection is None
    mode = wp.zeros(1, dtype=int, device=fitted.device)
    core = VTMWorkspace(
        fitted.domain,
        wp.clone(fitted.points),
        max_iterations=2,
        selective_threshold=0.3,
        selective_enabled=mode,
    ).run()
    assert core.error.numpy()[0] == 0 and core.iterations.numpy()[0] == 2


def test_control_array_contract_is_checked_at_setup():
    fitted = initial(16, 0)
    bad = wp.zeros(2, dtype=int, device=fitted.device)
    with pytest.raises(ValueError, match="iteration_limit"):
        VTMWorkspace(fitted.domain, fitted.points, iteration_limit=bad)
    with pytest.raises(ValueError, match="selective_enabled"):
        VTMWorkspace(fitted.domain, fitted.points, selective_threshold=0.3, selective_enabled=bad)
    with pytest.raises(ValueError, match="requires selective_threshold"):
        VTMWorkspace(
            fitted.domain,
            fitted.points,
            selective_enabled=wp.zeros(1, dtype=int, device=fitted.device),
        )
