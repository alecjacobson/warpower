"""State-machine checks for bounded fallback and convergence verification."""

import os

import numpy as np
import pytest
import warp as wp

from warpower import _selective_step_history as candidate
from warpower import _vtm

DEVICE = os.environ.get("WARPOWER_HISTORY_TEST_DEVICE", "cuda:0")


def array(value, dtype=int):
    return wp.array(np.atleast_1d(value), dtype=dtype, device=DEVICE)


def state(accepted=0, level=0, maximum=0.1, iteration=2):
    values = dict(
        mesh_error=0,
        boundary_error=0,
        accepted=accepted,
        attempt=1,
        backtracks=level,
        rejected=5,
        last_geometry=0,
        last_boundary=0,
        iterations=iteration,
        running=1,
        termination=0,
        selection_rejections=0,
        selection_last_error=4,
    )
    arrays = {k: array(v) for k, v in values.items()}
    arrays.update(
        alpha=array(0.5 * 2.0**-level, wp.float64),
        maximum=array(maximum, wp.float64),
        energy=array(0.25, wp.float64),
        boundary_energy=array([0.125, 0.0625], wp.float64),
        energy_history=wp.full((16, 3), -7, dtype=wp.float64, device=DEVICE),
    )
    return arrays


def args(s):
    names = [
        "mesh_error",
        "boundary_error",
        "accepted",
        "attempt",
        "alpha",
        "backtracks",
        "rejected",
        "last_geometry",
        "last_boundary",
        "iterations",
        "running",
        "termination",
        "maximum",
    ]
    return [s[n] for n in names] + [
        1e-4,
        10,
        12,
        s["energy"],
        s["boundary_energy"],
        s["energy_history"],
    ]


def finish(s, mode, history, statistics):
    wp.launch(
        candidate.finish_attempt,
        1,
        inputs=args(s)
        + [mode, history, statistics, 0.5, s["selection_rejections"], s["selection_last_error"]],
        device=DEVICE,
    )


@pytest.mark.parametrize(
    "accepted,level,maximum,iteration",
    [(0, 0, 0.1, 2), (0, 12, 0.1, 2), (1, 3, 0.1, 2), (1, 3, 0.0, 2), (1, 3, 0.1, 9)],
)
def test_global_finish_matches_original_even_with_stale_history(
    accepted, level, maximum, iteration
):
    a = state(accepted, level, maximum, iteration)
    b = state(accepted, level, maximum, iteration)
    wp.launch(_vtm.finish_attempt, 1, inputs=args(a), device=DEVICE)
    finish(b, array(0), array([1, 8, 0, 1, 1, 0]), array([0, 0, 0, 0, 0]))
    for k in a:
        np.testing.assert_array_equal(a[k].numpy(), b[k].numpy())


@pytest.mark.parametrize(
    "mode,valid,previous,expected",
    [(0, 1, 8, 0), (1, 0, 8, 0), (1, 1, 0, 0), (1, 1, 8, 7), (1, 1, 12, 11)],
)
def test_start_uses_absolute_levels_and_resets_retry_state(mode, valid, previous, expected):
    alpha = array(0.0, wp.float64)
    attempt = array(0)
    accepted = array(1)
    backtracks = array(99)
    error = array(0)
    running = array(1)
    history = array([valid, previous, 1, 1, 1, 0])
    statistics = array([0, 0, 0, 0, 0])
    wp.launch(
        candidate.start_attempt,
        1,
        inputs=[
            alpha,
            0.5,
            attempt,
            accepted,
            backtracks,
            error,
            running,
            array(mode),
            history,
            statistics,
            12,
            array(0),
        ],
        device=DEVICE,
    )
    assert alpha.numpy()[0] == 0.5 * 2.0**-expected and backtracks.numpy()[0] == expected
    assert attempt.numpy()[0] == 1 and accepted.numpy()[0] == 0
    assert history.numpy()[2] == 0 and history.numpy()[3] == int(expected > 0)


def test_convergence_requires_default_search_before_accepting_an_iteration():
    s = state(accepted=1, level=8, maximum=0.0)
    mode = array(1)
    history = array([1, 9, 0, 1, 1, 0])
    statistics = array([0, 0, 0, 0, 0])
    before = s["energy_history"].numpy().copy()
    finish(s, mode, history, statistics)
    assert s["iterations"].numpy()[0] == 2 and s["rejected"].numpy()[0] == 5
    assert s["alpha"].numpy()[0] == 0.5 and s["backtracks"].numpy()[0] == 0
    assert s["attempt"].numpy()[0] == 1 and s["accepted"].numpy()[0] == 0
    assert s["termination"].numpy()[0] == 0 and history.numpy()[2] == 1
    np.testing.assert_array_equal(s["energy_history"].numpy(), before)
    # A real default-scale success at larger displacement must keep iterating.
    s["accepted"].fill_(1)
    s["maximum"].fill_(0.1)
    finish(s, mode, history, statistics)
    assert s["iterations"].numpy()[0] == 3 and s["running"].numpy()[0] == 1
    assert s["termination"].numpy()[0] == 0 and s["attempt"].numpy()[0] == 0
    np.testing.assert_array_equal(statistics.numpy(), [0, 1, 0, 0, 0])


def test_exhaustion_retries_full_search_once_then_preserves_failure_and_error_history():
    s = state(accepted=0, level=12)
    s["mesh_error"].fill_(7)
    s["boundary_error"].fill_(11)
    mode = array(1)
    history = array([1, 9, 0, 1, 1, 0])
    statistics = array([0, 0, 0, 0, 0])
    finish(s, mode, history, statistics)
    assert s["rejected"].numpy()[0] == 6 and s["alpha"].numpy()[0] == 0.5
    assert s["attempt"].numpy()[0] == 1 and s["running"].numpy()[0] == 1
    assert history.numpy()[2] == 1 and s["termination"].numpy()[0] == 0
    # Exhaust the full original bracket. No second retry or unbounded loop.
    for _ in range(13):
        finish(s, mode, history, statistics)
    assert s["rejected"].numpy()[0] == 19 and s["iterations"].numpy()[0] == 2
    assert s["attempt"].numpy()[0] == 0 and s["running"].numpy()[0] == 0
    assert s["termination"].numpy()[0] == 2 and history.numpy()[0] == 0
    assert s["last_geometry"].numpy()[0] == 7 and s["last_boundary"].numpy()[0] == 11
    np.testing.assert_array_equal(statistics.numpy(), [0, 0, 0, 0, 1])


def test_a_small_success_after_default_retry_can_converge_without_looping():
    s = state(accepted=1, level=12, maximum=0.0)
    history = array([1, 9, 1, 1, 1, 0])
    statistics = array([0, 0, 0, 0, 0])
    finish(s, array(1), history, statistics)
    assert s["iterations"].numpy()[0] == 3 and s["termination"].numpy()[0] == 1
    assert s["attempt"].numpy()[0] == 0 and s["running"].numpy()[0] == 0
    np.testing.assert_array_equal(statistics.numpy(), 0)


def restart(s, mode, history, statistics):
    wp.launch(
        candidate.start_attempt,
        1,
        inputs=[
            s["alpha"],
            0.5,
            s["attempt"],
            s["accepted"],
            s["backtracks"],
            array(0),
            s["running"],
            mode,
            history,
            statistics,
            12,
            s["selection_rejections"],
        ],
        device=DEVICE,
    )


def test_stale_quality_reason_does_not_seed_history_after_objective_failure():
    s = state(accepted=0, level=5)
    mode = array(1)
    history = array([0, 0, 0, 0, 1, 0])
    statistics = array([0, 0, 0, 0, 0])
    # A sticky reason4 without a new selective rejection is not quality evidence.
    finish(s, mode, history, statistics)
    assert history.numpy()[4] == 0
    s["accepted"].fill_(1)
    finish(s, mode, history, statistics)
    assert history.numpy()[0] == 0 and history.numpy()[1] == 6
    restart(s, mode, history, statistics)
    assert s["alpha"].numpy()[0] == 0.5 and s["backtracks"].numpy()[0] == 0


def test_nonquality_warm_failure_retries_default_immediately_and_only_once():
    s = state(accepted=0, level=5)
    s["selection_rejections"].fill_(1)
    s["selection_last_error"].fill_(3)
    mode = array(1)
    history = array([1, 6, 0, 1, 1, 0])
    statistics = array([0, 0, 0, 0, 0])
    finish(s, mode, history, statistics)
    assert s["alpha"].numpy()[0] == 0.5 and s["backtracks"].numpy()[0] == 0
    assert s["rejected"].numpy()[0] == 6 and history.numpy()[2] == 1
    np.testing.assert_array_equal(statistics.numpy(), [0, 0, 0, 0, 1])
    s["selection_rejections"].fill_(2)
    finish(s, mode, history, statistics)
    assert s["alpha"].numpy()[0] == 0.25 and s["backtracks"].numpy()[0] == 1
    assert s["rejected"].numpy()[0] == 7 and history.numpy()[4] == 0
    np.testing.assert_array_equal(statistics.numpy(), [0, 0, 0, 0, 1])


def test_fresh_quality_rejection_can_seed_the_next_iteration():
    s = state(accepted=0, level=4)
    s["selection_rejections"].fill_(1)
    mode = array(1)
    history = array([0, 0, 0, 0, 1, 0])
    statistics = array([0, 0, 0, 0, 0])
    finish(s, mode, history, statistics)
    assert history.numpy()[4] == 1 and s["backtracks"].numpy()[0] == 5
    s["accepted"].fill_(1)
    finish(s, mode, history, statistics)
    assert history.numpy()[0] == 1 and history.numpy()[1] == 5
    restart(s, mode, history, statistics)
    assert s["backtracks"].numpy()[0] == 4 and s["alpha"].numpy()[0] == 0.5 * 2.0**-4


def test_quality_only_exhaustion_retains_the_full_default_fallback():
    s = state(accepted=0, level=12)
    s["selection_rejections"].fill_(1)
    mode = array(1)
    history = array([1, 9, 0, 1, 1, 0])
    statistics = array([0, 0, 0, 0, 0])
    finish(s, mode, history, statistics)
    assert s["alpha"].numpy()[0] == 0.5 and s["backtracks"].numpy()[0] == 0
    assert s["rejected"].numpy()[0] == 6 and history.numpy()[2] == 1
    np.testing.assert_array_equal(statistics.numpy(), [0, 0, 1, 0, 0])


@pytest.mark.parametrize("quality_rounds,local_retries", [(0, 0), (1, 0), (1, 2)])
def test_complete_mesher_eligibility_and_fresh_replay_resets(quality_rounds, local_retries):
    from tests.test_domain import cube
    from tests.test_mesher import validate
    from warpower import SurfaceDomain, VariationalMeshingWorkspace

    domain = SurfaceDomain(*cube(), surface_subdivisions=2)
    mesh = VariationalMeshingWorkspace(
        domain,
        96,
        initial_count=64,
        quality_rounds=quality_rounds,
        local_quality_retries=local_retries,
    )
    eligible = quality_rounds > 0 and local_retries == 0
    assert (mesh.core.step_history is not None) == eligible
    mesh.run()
    validate(mesh)
    expected = mesh.points.numpy().copy()
    if eligible:
        history = mesh.core.step_history
        statistics = history.statistics.numpy().copy()
        # Neither a previous solve nor a failed fresh invocation can supply
        # step-size evidence or leak counters into the next complete replay.
        history.history.fill_(999999)
        history.statistics.fill_(999999)
        mesh.run()
        validate(mesh)
        np.testing.assert_array_equal(mesh.points.numpy(), expected)
        np.testing.assert_array_equal(history.statistics.numpy(), statistics)
