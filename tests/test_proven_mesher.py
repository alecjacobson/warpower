"""Fresh solves and device-controlled recovery after ordinary fitting fails."""

import numpy as np
import pytest
import warp as wp

import warpower
from examples.vtm_inputs import surface
from tests.test_domain import cube
from tests.test_mesher import validate
from warpower._vtm import VTMWorkspace


def hand():
    return warpower.SurfaceDomain(
        *surface("thin-hand", None), surface_subdivisions=20, crease_subdivisions=20
    )


@pytest.mark.parametrize("seed,retries", [(2, 0), (7, 0), (7, 4)])
def test_complete_hand_recovery_preserves_corners_and_runs_quality(seed, retries):
    m = warpower.VariationalMeshingWorkspace(
        hand(), 1000, initial_count=256, seed=seed, local_quality_retries=retries
    ).run()
    validate(m)
    assert m._proven_exterior_enabled.numpy()[0] == 1
    assert m._boundary_rounding.statistics.numpy()[2] > 0
    assert m.extraction.proven_removed_count.numpy()[0] > 0
    assert m._material_recovery.numpy()[0] == 1
    assert m.rounds.numpy()[2] == 4
    assert m.core._boundary_rounding is None
    assert m.minimum_radius_ratio.numpy()[0] > 0.05
    for stage in [m.jitter, m.interior]:
        assert stage.extraction._proven_exterior.enabled is m._proven_exterior_enabled
    assert (
        m.core.selection.recovery_extraction._proven_exterior.enabled is m._proven_exterior_enabled
    )


def test_policy_opt_out_preserves_original_failure_and_default_core_storage():
    m = warpower.VariationalMeshingWorkspace(
        hand(), 1000, initial_count=256, seed=7, proven_exterior_recovery=False
    ).run()
    assert m.error.numpy()[0] == 64 and m.valid.numpy()[0] == 0
    np.testing.assert_array_equal(m.counts.numpy(), 0)
    assert m._boundary_rounding is None and m._proven_exterior_enabled is None
    assert m.extraction._proven_exterior is None
    d = warpower.SurfaceDomain(*cube())
    disabled = warpower.VariationalMeshingWorkspace(
        d,
        32,
        initial_count=32,
        quality_rounds=0,
        containment_recovery=False,
        exterior_recovery=False,
    )
    assert not disabled.proven_exterior_recovery and disabled._boundary_rounding is None
    core = VTMWorkspace(d, cube()[0], max_iterations=0)
    assert core._boundary_rounding is None and not core.proven_exterior_recovery


@wp.kernel
def producer(phase: wp.array[int], target: wp.array[int], seed: wp.array[int]):
    target[0] = 1000
    seed[0] = 7
    if phase[0] == 1:
        target[0] = 0
    if phase[0] == 2:
        seed[0] = 42


@wp.kernel
def consumer(
    phase: wp.array[int],
    condition: wp.array[int],
    error: wp.array[int],
    valid: wp.array[int],
    counts: wp.array[int],
    enabled: wp.array[int],
    statistics: wp.array[int],
    rows: wp.array2d[int],
):
    i = phase[0]
    rows[i, 0] = error[0]
    rows[i, 1] = valid[0]
    rows[i, 2] = enabled[0]
    rows[i, 3] = statistics[2]
    for k in range(4):
        rows[i, 4 + k] = counts[k]
    phase[0] += 1
    condition[0] = int(phase[0] < 4)


def test_one_captured_loop_failure_healthy_seed_and_recovery(monkeypatch, tmp_path):
    d = hand()
    seed = wp.full(1, 7, dtype=int, device=d.device)
    target = wp.full(1, 1000, dtype=int, device=d.device)
    m = warpower.VariationalMeshingWorkspace(
        d, 1000, initial_count=256, seed=seed, target_count=target
    )
    phase = wp.zeros(1, dtype=int, device=d.device)
    condition = wp.ones(1, dtype=int, device=d.device)
    rows = wp.empty((4, 8), dtype=int, device=d.device)
    wp.load_module(module=__name__, device=d.device)

    def forbidden(*args, **kwargs):
        raise AssertionError("host activity during capture/replay")

    with monkeypatch.context() as p:
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
            p.setattr(wp, name, forbidden)
        p.setattr(wp.array, "numpy", forbidden)

        def step():
            wp.launch(producer, 1, inputs=[phase, target, seed])
            m.run()
            wp.launch(
                consumer,
                1,
                inputs=[
                    phase,
                    condition,
                    m.error,
                    m.valid,
                    m.counts,
                    m._proven_exterior_enabled,
                    m._boundary_rounding.statistics,
                    rows,
                ],
            )

        with wp.ScopedCapture() as capture:
            phase.zero_()
            condition.fill_(1)
            wp.capture_while(condition, step)
        wp.capture_launch(capture.graph)
    result = rows.numpy()
    np.testing.assert_array_equal(result[:, 0], [0, 1, 0, 0])
    np.testing.assert_array_equal(result[:, 1], [1, 0, 1, 1])
    np.testing.assert_array_equal(result[:, 2], [1, 0, 0, 1])
    np.testing.assert_array_equal(result[[1, 2], 3], 0)
    assert np.all(result[[0, 3], 3] > 0)
    np.testing.assert_array_equal(result[1, 4:], 0)
    np.testing.assert_array_equal(result[0, 4:], result[3, 4:])
    validate(m)
    dot = tmp_path / "proven-mesher-loop.dot"
    wp.capture_debug_dot_print(capture.graph, str(dot), verbose=True)
    assert all(s not in dot.read_text() for s in ["MEM_ALLOC", "MEM_FREE", "HOST"])
