"""Bounded selective step history with full-search convergence and failure fallbacks.

The complete mesher opts in for selective cores without local vertex rollback.
History resets at every core invocation; statistics cover a complete mesher run.
Every actual proposal retains the ordinary geometry, objective, and quality guards.
"""

import warp as wp

from ._vtm import _finish_attempt, _start_attempt

wp.set_module_options({"enable_backward": False, "fast_math": False})


@wp.kernel
def start_attempt(
    alpha: wp.array[wp.float64],
    damping: wp.float64,
    attempt: wp.array[int],
    accepted: wp.array[int],
    backtracks: wp.array[int],
    error: wp.array[int],
    running: wp.array[int],
    mode: wp.array[int],
    history: wp.array[int],
    statistics: wp.array[int],
    max_backtracks: int,
    selection_rejections: wp.array[int],
):
    _start_attempt(alpha, damping, attempt, accepted, backtracks, error, running)
    history[2] = 0  # full default search has not been retried in this iteration
    history[3] = 0  # this iteration starts below the default scale
    history[4] = 1  # every observed rejection in this search was quality-only
    history[5] = selection_rejections[0]
    if mode[0] == 1 and error[0] == 0 and history[0] != 0:
        level = wp.max(0, wp.min(history[1] - 1, max_backtracks))
        for k in range(level):
            alpha[0] *= wp.float64(0.5)
        # Absolute halving level preserves the original minimum trial scale.
        # rejected_steps continues to count only proposals actually rejected.
        backtracks[0] = level
        history[3] = int(level > 0)
        statistics[0] += int(level > 0)


@wp.kernel
def finish_attempt(
    mesh_error: wp.array[int],
    boundary_error: wp.array[int],
    accepted: wp.array[int],
    attempt: wp.array[int],
    alpha: wp.array[wp.float64],
    backtracks: wp.array[int],
    rejected: wp.array[int],
    last_geometry: wp.array[int],
    last_boundary: wp.array[int],
    iterations: wp.array[int],
    running: wp.array[int],
    termination: wp.array[int],
    maximum: wp.array[wp.float64],
    tolerance: wp.float64,
    max_iterations: int,
    max_backtracks: int,
    energy: wp.array[wp.float64],
    boundary_energy: wp.array[wp.float64],
    energy_history: wp.array2d[wp.float64],
    mode: wp.array[int],
    history: wp.array[int],
    statistics: wp.array[int],
    damping: wp.float64,
    selection_rejections: wp.array[int],
    selection_last_error: wp.array[int],
):
    fresh_quality_failure = (
        selection_rejections[0] > history[5]
        and selection_last_error[0] == 4
        and mesh_error[0] == 0
        and boundary_error[0] == 0
    )
    history[5] = selection_rejections[0]
    if accepted[0] == 0 and not fresh_quality_failure:
        history[4] = 0
    warm = mode[0] == 1 and history[3] != 0 and history[2] == 0
    retry_convergence = warm and accepted[0] != 0 and maximum[0] <= tolerance
    retry_exhaustion = (
        warm and accepted[0] == 0 and backtracks[0] >= max_backtracks and fresh_quality_failure
    )
    retry_nonquality = warm and accepted[0] == 0 and not fresh_quality_failure
    if retry_convergence:
        # This valid trial is discarded, not counted as an accepted iteration
        # or a rejected proposal. Recompute from the unchanged original sites.
        history[2] = 1
        history[4] = 1
        statistics[1] += 1
        alpha[0] = damping
        backtracks[0] = 0
        accepted[0] = 0
        attempt[0] = 1
        running[0] = 1
        termination[0] = 0
        return
    _finish_attempt(
        mesh_error,
        boundary_error,
        accepted,
        attempt,
        alpha,
        backtracks,
        rejected,
        last_geometry,
        last_boundary,
        iterations,
        running,
        termination,
        maximum,
        tolerance,
        max_iterations,
        max_backtracks,
        energy,
        boundary_energy,
        energy_history,
    )
    if retry_exhaustion or retry_nonquality:
        # The failed warm trial has been counted by the unchanged finish body.
        # Retry all default scales once; no rejection monotonicity is assumed.
        history[2] = 1
        history[4] = 1
        statistics[2] += int(retry_exhaustion)
        statistics[4] += int(retry_nonquality)
        alpha[0] = damping
        backtracks[0] = 0
        accepted[0] = 0
        attempt[0] = 1
        running[0] = 1
        termination[0] = 0
        return
    if mode[0] == 1:
        if accepted[0] != 0:
            history[0] = history[4]
            history[1] = backtracks[0]
            statistics[3] += int(history[3] != 0 and history[2] == 0)
        elif attempt[0] == 0:
            history[0] = 0


class SelectiveStepHistory:
    def __init__(self, core):
        # valid, previous absolute halving level, default retried, warm start,
        # quality-only rejection history, previous selective rejection count.
        self.history = wp.zeros(6, dtype=int, device=core.device)
        # Warm starts, convergence retries, exhaustion retries, warm accepts,
        # and immediate default retries after non-quality failures.
        self.statistics = wp.zeros(5, dtype=int, device=core.device)
        core._start_attempt_kernel = start_attempt
        core._start_attempt_args = [
            core.controls.mode,
            self.history,
            self.statistics,
            core.max_backtracks,
            core.selection.rejections,
        ]
        core._finish_attempt_kernel = finish_attempt
        core._finish_attempt_args = [
            core.controls.mode,
            self.history,
            self.statistics,
            core.damping,
            core.selection.rejections,
            core.selection.last_error,
        ]
        wp.load_module(module=__name__, device=core.device)

    def reset(self):
        self.history.zero_()
