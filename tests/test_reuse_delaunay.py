from itertools import product

import numpy as np
import pytest
import warp as wp

from tests.test_delaunay_recovery import exact_empty, valid
from tests.test_tetrahedra import reference, tet_set
from warpower._reuse_delaunay import ReuseDelaunayWorkspace


@wp.kernel
def produce(frames: wp.array2d[wp.vec3], phase: wp.array[int], points: wp.array[wp.vec3]):
    i = wp.tid()
    points[i] = frames[phase[0], i]


@wp.kernel
def consume(
    phase: wp.array[int],
    tets: wp.array2d[int],
    count: wp.array[int],
    error: wp.array[int],
    attempted: wp.array[int],
    accepted: wp.array[int],
    base_attempted: wp.array[int],
    history: wp.array3d[int],
    meta: wp.array2d[int],
):
    i, frame = wp.tid(), phase[0]
    if i < count[0]:
        for k in range(4):
            history[frame, i, k] = tets[i, k]
    if i == 0:
        meta[frame, 0] = count[0]
        meta[frame, 1] = error[0]
        meta[frame, 2] = attempted[0]
        meta[frame, 3] = accepted[0]
        meta[frame, 4] = base_attempted[0]


@wp.kernel
def advance(phase: wp.array[int]):
    phase[0] += 1


@pytest.mark.parametrize("rounds", [0, 4])
def test_changing_device_producer_invalid_phase_and_consumer_in_one_graph(rounds):
    rng = np.random.default_rng(930)
    initial = rng.normal(size=(32, 3)).astype(np.float32)
    nearby = initial.copy()
    nearby[5] += np.float32(1e-5)
    unrelated = rng.normal(size=(32, 3)).astype(np.float32)
    duplicate = unrelated.copy()
    duplicate[16:] = duplicate[:16]
    invalid = initial.copy()
    invalid[4, 0] = np.nan
    inputs = np.stack([initial, nearby, unrelated, duplicate, invalid, initial])
    w = ReuseDelaunayWorkspace(initial, reuse_max_rounds=rounds)
    frames = wp.array(inputs, dtype=wp.vec3, device=w.device)
    phase = wp.zeros(1, dtype=int, device=w.device)
    history = wp.empty((len(inputs), len(w.tetrahedra), 4), dtype=int, device=w.device)
    meta = wp.empty((len(inputs), 5), dtype=int, device=w.device)
    wp.load_module(module=__name__, device=w.device)
    with wp.ScopedCapture() as capture:
        wp.launch(produce, w.n, inputs=[frames, phase, w.points])
        w.run()
        wp.launch(
            consume,
            len(w.tetrahedra),
            inputs=[
                phase,
                w.tetrahedra,
                w.tetrahedra_count,
                w.error,
                w.reuse_attempted,
                w.reuse_accepted,
                w.base_attempted,
                history,
                meta,
            ],
        )
        wp.launch(advance, 1, inputs=[phase])
    # No readback or host decisions between producer/consumer phases.
    for _ in inputs:
        wp.capture_launch(capture.graph)
    rows, cells = meta.numpy(), history.numpy()
    for i, points in enumerate(inputs):
        if i == 4:
            assert rows[i, 0] == 0 and rows[i, 1] != 0
        else:
            assert rows[i, 0] > 0 and rows[i, 1] == 0
            # Qhull may choose any duplicate index; the device uses the first.
            owners = np.unique(points, axis=0, return_index=True)[1]
            expected = {tuple(sorted(owners[list(tet)])) for tet in reference(points[owners])}
            assert tet_set(cells[i, : rows[i, 0]]) == expected, i
    assert rows[0, 2:4].tolist() == [0, 0]
    assert rows[1, 2:5].tolist() == [1, 1, 0]
    assert rows[5, 2:4].tolist() == [0, 0]  # Invalid phase clears the retained prefix.
    if rounds == 0:
        assert rows[2, 2:5].tolist() == [1, 0, 1]  # No completion budget: rebuild.


@pytest.mark.parametrize("corruption", ["count", "indices", "empty"])
def test_bad_retained_fragment_cannot_bypass_fresh_construction(corruption):
    points = np.random.default_rng(931).normal(size=(24, 3)).astype(np.float32)
    w = ReuseDelaunayWorkspace(points).run()
    expected = tet_set(valid(w, points))
    if corruption == "count":
        w._previous_count.fill_(len(w._previous) + 1)
    elif corruption == "indices":
        w._previous.fill_(-1)
    else:
        w.clear_reuse()
    w.run()
    assert w.reuse_accepted.numpy()[0] == 0
    assert w.base_attempted.numpy()[0] == 1
    assert tet_set(valid(w, points)) == expected


def test_exhausted_output_capacity_stays_empty_across_replays():
    points = np.random.default_rng(932).normal(size=(24, 3)).astype(np.float32)
    w = ReuseDelaunayWorkspace(points, tetrahedra_capacity=1)
    with wp.ScopedCapture() as capture:
        w.run()
    for _ in range(2):
        wp.capture_launch(capture.graph)
        assert w.error.numpy()[0] != 0
        assert w.tetrahedra_count.numpy()[0] == 0
        assert w._previous_count.numpy()[0] == 0


def test_exact_cospherical_duplicates_and_one_ulp_motion():
    points = np.array(list(product([-1, 1], repeat=3)), np.float32)
    points = np.vstack([points, points[:2]])
    w = ReuseDelaunayWorkspace(points).run()
    exact_empty(points, valid(w, points))
    w.run()
    assert w.reuse_accepted.numpy()[0] == 1
    exact_empty(points, valid(w, points))
    points[0, 0] = np.nextafter(points[0, 0], np.float32(-np.inf))
    wp.copy(w.points, wp.array(points, dtype=wp.vec3, device=w.device))
    w.run()
    exact_empty(points, valid(w, points))
