"""Independent precision requirements and storage floors for cell retries.

State is indexed by original site, even when outputs use compact retry rows.
Its components hold precision flags, the minimum clipping capacity, and the
minimum traversal-queue capacity. A queue failure cannot increase the clipping
requirement. Public status codes describe the last failure, never this history.
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
def promote(state: wp.vec3i, status: int):
    if numerical(status):
        state[0] = state[0] | DOUBLE
    return state


@wp.func
def after_attempt(
    state: wp.vec3i,
    status: int,
    capacity: int,
    robust: bool,
    cooperative: bool,
    scratch_overflow: bool,
    queue_capacity: int = 128,
):
    state = promote(state, status)
    state[1] = wp.max(state[1], capacity)
    state[2] = wp.max(state[2], queue_capacity)
    if robust:
        state[0] = state[0] | DOUBLE
        if cooperative:
            state[0] = state[0] | COOPERATIVE_TRIED
        elif numerical(status):
            # More storage cannot resolve a failed FP64 numerical check.
            state[0] = state[0] | NUMERICAL_EXHAUSTED
    if (status == 1 or status == 2) and scratch_overflow:
        state[1] = wp.max(state[1], 2 * capacity)
    if status == 4:
        state[2] = wp.max(state[2], 2 * queue_capacity)
    return state


@wp.func
def eligible(state: wp.vec3i, status: int, capacity: int, robust: bool, queue_capacity: int = 512):
    state = promote(state, status)
    return (
        status != 0
        and status != 3
        and (state[0] & NUMERICAL_EXHAUSTED) == 0
        and ((state[0] & DOUBLE) != 0) == robust
        and capacity >= state[1]
        and queue_capacity >= state[2]
    )


@wp.func
def cooperative_eligible(state: wp.vec3i, status: int):
    return (
        numerical(status)
        and eligible(state, status, 128, True, 128)
        and (state[0] & COOPERATIVE_TRIED) == 0
    )


@wp.kernel
def collect_retries(
    status: wp.array[int],
    states: wp.array[wp.vec3i],
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
            if counts.shape[0] > 4 and states[i][2] > 128:
                wp.atomic_max(counts, 4, states[i][2])
        if eligible(states[i], s, 512, True):
            wp.atomic_add(counts, 3, 1)
            if counts.shape[0] > 5 and states[i][2] > 128:
                wp.atomic_max(counts, 5, states[i][2])


@wp.kernel
def prepare_repairs(jobs: wp.array[int], states: wp.array[wp.vec3i]):
    i = jobs[wp.tid()]
    # A new consistency-repair request may use changed neighbor hints, but
    # retains the established storage floor and always requires FP64.
    states[i] = wp.vec3i(DOUBLE, states[i][1], states[i][2])


@wp.kernel
def pending_large_queue(
    status: wp.array[int],
    states: wp.array[wp.vec3i],
    robust: bool,
    pending: wp.array[int],
    largest: wp.array[int],
):
    i = wp.tid()
    if states[i][2] > 128 and eligible(states[i], status[i], 512, robust):
        wp.atomic_max(pending, 0, 1)
        if states[i][2] > 256:
            wp.atomic_max(largest, 0, 1)
