"""Independent precision requirements and storage floors for cell retries.

State is indexed by original site, even when outputs use compact retry rows.
Its first component holds precision flags; the second is the minimum scratch
capacity. Public status codes describe the last failure, never this history.
"""

import warp as wp

wp.set_module_options({"enable_backward": False})

DOUBLE = wp.constant(1)
COOPERATIVE_TRIED = wp.constant(2)
NUMERICAL_EXHAUSTED = wp.constant(4)


@wp.func
def numerical(status: int):
    return status == 5 or status == 6


@wp.func
def capacity_failure(status: int):
    return status == 1 or status == 2 or status == 4


@wp.func
def promote(state: wp.vec2i, status: int):
    if numerical(status):
        state[0] = state[0] | DOUBLE
    return state


@wp.func
def after_attempt(
    state: wp.vec2i,
    status: int,
    capacity: int,
    robust: bool,
    cooperative: bool,
    scratch_overflow: bool,
):
    state = promote(state, status)
    state[1] = wp.max(state[1], capacity)
    if robust:
        state[0] = state[0] | DOUBLE
        if cooperative:
            state[0] = state[0] | COOPERATIVE_TRIED
        elif numerical(status):
            # More storage cannot resolve a failed FP64 numerical check.
            state[0] = state[0] | NUMERICAL_EXHAUSTED
    if capacity_failure(status) and scratch_overflow:
        state[1] = wp.max(state[1], 2 * capacity)
    return state


@wp.func
def eligible(state: wp.vec2i, status: int, capacity: int, robust: bool):
    state = promote(state, status)
    return (
        status != 0
        and status != 3
        and (state[0] & NUMERICAL_EXHAUSTED) == 0
        and ((state[0] & DOUBLE) != 0) == robust
        and capacity >= state[1]
    )


@wp.func
def cooperative_eligible(state: wp.vec2i, status: int):
    return (
        numerical(status)
        and eligible(state, status, 128, True)
        and (state[0] & COOPERATIVE_TRIED) == 0
    )


@wp.kernel
def collect_retries(
    status: wp.array[int],
    states: wp.array[wp.vec2i],
    jobs: wp.array[int],
    counts: wp.array[int],
):
    i = wp.tid()
    s = status[i]
    if s != 0 and s != 3:
        at = wp.atomic_add(counts, 0, 1)
        jobs[at] = i
        if eligible(states[i], s, 512, False):
            wp.atomic_add(counts, 2, 1)
        if eligible(states[i], s, 512, True):
            wp.atomic_add(counts, 3, 1)


@wp.kernel
def prepare_repairs(jobs: wp.array[int], states: wp.array[wp.vec2i]):
    i = jobs[wp.tid()]
    # A new consistency-repair request may use changed neighbor hints, but
    # retains the established storage floor and always requires FP64.
    states[i] = wp.vec2i(DOUBLE, states[i][1])
