from pathlib import Path

import numpy as np
import pytest
import warp as wp

from tests.test_delaunay_recovery import valid
from tests.test_tetrahedra import tet_set
from warpower import DelaunayWorkspace
from warpower._delaunay_recovery import amplitude_level

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.kernel
def ordering(cases: wp.array2d[int], result: wp.array[int]):
    i = wp.tid()
    result[i] = amplitude_level(cases[i, 0], cases[i, 1], cases[i, 2], cases[i, 3] != 0)


def test_every_budget_and_preference_visits_each_level_exactly_once():
    cases, expected = [], []
    for maximum in range(1, 13):
        for preferred in range(-1, maximum + 1):
            for cache in [0, 1]:
                first = preferred if cache and 0 <= preferred < maximum else 0
                sequence = [first] + [x for x in range(maximum) if x != first]
                for attempt in range(maximum):
                    cases.append([attempt, preferred, maximum, cache])
                    expected.append(sequence[attempt])
    source = wp.array(np.array(cases, np.int32), dtype=int, device="cuda:0")
    result = wp.empty(len(cases), dtype=int, device=source.device)
    wp.launch(ordering, len(cases), inputs=[source, result])
    np.testing.assert_array_equal(result.numpy(), expected)


@wp.kernel
def controlled_candidate(
    level: wp.array[int],
    attempts: wp.array[int],
    wanted: wp.array[int],
    error: wp.array[int],
    history: wp.array[int],
):
    history[attempts[0]] = level[0]
    if level[0] != wanted[0]:
        error[0] = 2


@pytest.mark.parametrize("cache", [False, True])
def test_captured_order_changes_preserve_full_certificate_and_finite_failure(cache):
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
    workspace = DelaunayWorkspace(points, cache_strategy=cache, symbolic_recovery=False)
    wanted = wp.full(1, 4, dtype=int, device=workspace.device)
    history = wp.empty(6, dtype=int, device=workspace.device)
    base_run, auxiliary_run = workspace._base.run, workspace._auxiliary.run

    def force_recovery(stream=None):
        base_run(stream)
        workspace._base.error.fill_(2)

    def candidate(stream=None):
        auxiliary_run(stream)
        wp.launch(
            controlled_candidate,
            1,
            inputs=[
                workspace.auxiliary_level,
                workspace.attempts,
                wanted,
                workspace._auxiliary.error,
                history,
            ],
        )

    workspace._base.run = force_recovery
    workspace._auxiliary.run = candidate
    wp.load_module(module=__name__, device=workspace.device)
    with wp.ScopedCapture() as capture:
        history.fill_(-1)
        workspace.run()

    previous = 0
    for chosen in [4, 4, 2, -1, 4]:
        wanted.fill_(chosen)
        wp.capture_launch(capture.graph)
        order = [previous] + [x for x in range(6) if x != previous] if cache else list(range(6))
        attempts = order.index(chosen) + 1 if chosen >= 0 else 6
        assert workspace.attempts.numpy()[0] == attempts
        np.testing.assert_array_equal(history.numpy()[:attempts], order[:attempts])
        np.testing.assert_array_equal(history.numpy()[attempts:], -1)
        if chosen < 0:
            assert workspace.error.numpy()[0] == 2
            assert workspace.tetrahedra_count.numpy()[0] == 0
            assert workspace._preferred_level.numpy()[0] == 0
            previous = 0
        else:
            assert tet_set(valid(workspace, points)) == {(0, 1, 2, 3)}
            assert workspace.certificate_error.numpy()[0] == 0
            assert workspace.auxiliary_level.numpy()[0] == chosen
            assert workspace._preferred_level.numpy()[0] == (chosen if cache else 0)
            previous = chosen

    bad = points.copy()
    bad[0, 0] = np.nan
    workspace.points.assign(bad)
    wp.capture_launch(capture.graph)
    # The injected base failure overrides the normal invalid-input code, but
    # every candidate still fails and no stale successful count can survive.
    assert workspace.error.numpy()[0] != 0
    assert workspace.tetrahedra_count.numpy()[0] == 0
    assert workspace._preferred_level.numpy()[0] == 0


def test_real_hand_reuses_fifth_amplitude_and_recovers_after_invalid_input():
    points = np.load(Path(__file__).parent / "data/hand-fit256.npz")["points"]
    workspace = DelaunayWorkspace(points).run()
    expected = tet_set(valid(workspace, points))
    assert workspace.attempts.numpy()[0] == 5
    assert workspace.auxiliary_level.numpy()[0] == 4
    workspace.run()
    assert tet_set(valid(workspace, points)) == expected
    assert workspace.attempts.numpy()[0] == 1
    assert workspace.auxiliary_level.numpy()[0] == 4
    bad = points.copy()
    bad[10, 1] = np.nan
    workspace.points.assign(bad)
    workspace.run()
    assert workspace.error.numpy()[0] == 1
    assert workspace.tetrahedra_count.numpy()[0] == 0
    assert workspace._preferred_level.numpy()[0] == 0
    workspace.points.assign(points)
    workspace.run()
    assert tet_set(valid(workspace, points)) == expected
    assert workspace.attempts.numpy()[0] == 5
