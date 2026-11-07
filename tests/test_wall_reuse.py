"""Private candidate acceptance, capacity and changing-input graph checks."""

import numpy as np
import pytest
import warp as wp
from scipy.spatial import Delaunay

from warpower._wall_reuse import WallReuseDelaunayWorkspace as DelaunayWorkspace

wp.set_module_options({"enable_backward": False})


def cloud(n=1000):
    return np.random.default_rng(2026).uniform(-1, 1, (n, 3)).astype(np.float32)


def canonical(t):
    return {tuple(sorted(row)) for row in t}


def check(w, p):
    assert w.error.numpy()[0] == 0
    count = int(w.tetrahedra_count.numpy()[0])
    actual = w.tetrahedra.numpy()[:count]
    assert count == len(canonical(actual))
    unique, indices = np.unique(p, axis=0, return_index=True)
    ref = indices[Delaunay(unique).simplices]
    assert canonical(actual) == canonical(ref)
    return actual


@wp.kernel
def producer(frames: wp.array2d[wp.vec3], phase: wp.array[int], points: wp.array[wp.vec3]):
    i = wp.tid()
    points[i] = frames[phase[0], i]


@wp.kernel
def consume(
    tets: wp.array2d[int], count: wp.array[int], error: wp.array[int], checksum: wp.array[wp.int64]
):
    i = wp.tid()
    if error[0] == 0 and i < count[0]:
        v = wp.int64(0)
        for k in range(4):
            v += wp.int64(tets[i, k])
        wp.atomic_add(checksum, 0, v)


@wp.kernel
def advance(phase: wp.array[int], count: int):
    phase[0] = (phase[0] + 1) % count


def test_device_producer_consumer_owner_changes_invalid_recovery(monkeypatch):
    p = cloud()
    frames = np.tile(p, (7, 1, 1))
    frames[1, 0] += np.array([0.05, -0.02, 0.01], np.float32)
    frames[2, 128:] = p[0]
    frames[3, 0] = p[1]
    frames[4, 0, 1] = np.nan
    frames[5] = p * np.float32(2.0**20) + np.float32(2.0**30)
    w = DelaunayWorkspace(p, fast_candidates=False, cache_strategy=False)
    source = wp.array(frames, dtype=wp.vec3, device=w.device)
    phase = wp.zeros(1, dtype=int, device=w.device)
    checksum = wp.zeros(1, dtype=wp.int64, device=w.device)
    wp.load_module(module=__name__, device=w.device)
    pointers = (w.points.ptr, w.tetrahedra.ptr, w.tetrahedra_count.ptr)

    def forbidden(*args, **kwargs):
        raise AssertionError("allocation/readback/synchronization inside capture")

    with monkeypatch.context() as patch:
        for name in [
            "empty",
            "empty_like",
            "zeros",
            "ones",
            "full",
            "clone",
            "synchronize",
            "synchronize_device",
            "synchronize_stream",
        ]:
            patch.setattr(wp, name, forbidden)
        patch.setattr(wp.array, "numpy", forbidden)
        with wp.ScopedCapture(device=w.device) as capture:
            wp.launch(producer, len(p), inputs=[source, phase, w.points])
            w.run()
            checksum.zero_()
            wp.launch(
                consume,
                len(w.tetrahedra),
                inputs=[w.tetrahedra, w.tetrahedra_count, w.error, checksum],
            )
            wp.launch(advance, 1, inputs=[phase, len(frames)])
    for i, frame in enumerate(frames):
        wp.capture_launch(capture.graph)
        if i == 4:
            assert w.error.numpy()[0] != 0
            assert w.tetrahedra_count.numpy()[0] == checksum.numpy()[0] == 0
        else:
            tets = check(w, frame)
            assert checksum.numpy()[0] == int(tets.astype(np.int64).sum())
        assert pointers == (w.points.ptr, w.tetrahedra.ptr, w.tetrahedra_count.ptr)
    assert phase.numpy()[0] == 0


@pytest.mark.parametrize("failure", ["overflow", "empty", "filter", "completion"])
def test_forced_failure_runs_ordinary_constructor(monkeypatch, failure):
    p = cloud()
    w = DelaunayWorkspace(p, cache_strategy=False, fast_candidates=False)
    base = w
    if failure in ["overflow", "empty"]:
        original = base._wall.run

        def corrupt():
            original()
            base._wall.raw_count.fill_(len(base._wall.raw_tets) + 1 if failure == "overflow" else 0)

        monkeypatch.setattr(base._wall, "run", corrupt)
    else:
        original = base._reuse_completion.run

        def corrupt():
            original()
            if failure == "filter":
                base._reuse_filter.error.fill_(2)
            else:
                base._reuse_completion.error.fill_(16)

        monkeypatch.setattr(base._reuse_completion, "run", corrupt)
    with wp.ScopedCapture(device=w.device) as capture:
        w.run()
    wp.capture_launch(capture.graph)
    check(w, p)
    assert base.wall_accepted.numpy()[0] == 0
    assert w.base_attempted.numpy()[0] == 1


def test_output_capacity_fails_without_partial_output():
    w = DelaunayWorkspace(
        cloud(128), tetrahedra_capacity=1, cache_strategy=False, fast_candidates=False
    ).run()
    assert w.error.numpy()[0] != 0
    assert w.tetrahedra_count.numpy()[0] == 0


@pytest.mark.parametrize("scale", [2.0**-60, 2.0**60])
def test_extreme_finite_scales_use_original_coordinate_certificate(scale):
    p = cloud(256) * np.float32(scale)
    w = DelaunayWorkspace(p, fast_candidates=False).run()
    check(w, p)


def check_face_index(w, tets):
    from collections import Counter

    expected = Counter(
        tuple(sorted(np.delete(tet, opposite))) for tet in tets for opposite in range(4)
    )
    c = w._certificate
    table, occurrences = c._table.numpy(), c._occurrences.numpy()
    actual = {}
    for slot in np.flatnonzero(occurrences):
        face = int(table[slot])
        assert 0 <= face < 4 * len(tets)
        key = tuple(sorted(np.delete(tets[face // 4], face % 4)))
        assert key not in actual
        actual[key] = int(occurrences[slot])
    assert actual == expected
    count = int(c._boundary_count.numpy()[0])
    faces = c._boundary.numpy()[:count]
    boundary = {tuple(sorted(np.delete(tets[int(face) // 4], int(face) % 4))) for face in faces}
    assert count == len(boundary)
    assert boundary == {key for key, value in expected.items() if value == 1}
    np.testing.assert_array_equal(np.flatnonzero(c._used.numpy()), np.unique(tets.ravel()))
    assert w._candidate_count.numpy()[0] == len(tets)
    np.testing.assert_array_equal(w._candidate.numpy()[: len(tets)], tets)


def test_current_wall_proof_publishes_matching_metadata_without_second_certificate(monkeypatch):
    p = cloud()
    w = DelaunayWorkspace(p, cache_strategy=False, fast_candidates=False)
    assert w._wall.raw_tets.ptr == w._previous.ptr
    assert w._wall.raw_count.ptr == w._previous_count.ptr
    assert not any(hasattr(w._wall, name) for name in ["keys", "table", "live", "tets"])

    def forbidden_outer_certificate(*args, **kwargs):
        # This conditional body may be recorded, but must not execute after
        # publication of the already certified current wall completion.
        w._certificate.error.fill_(256)

    monkeypatch.setattr(w._certificate, "run", forbidden_outer_certificate)
    with wp.ScopedCapture(device=w.device) as capture:
        w.clear_reuse()
        w.run()
    for shift in [0.0, 0.02]:
        current = p.copy()
        current[0, 0] += np.float32(shift)
        w.points.assign(current)
        wp.capture_launch(capture.graph)
        tets = check(w, current)
        assert w.wall_accepted.numpy()[0] == 1
        assert w.base_attempted.numpy()[0] == 0
        assert w.reuse_attempted.numpy()[0] == 0
        assert w.reuse_seeds.numpy()[0] == w.reuse_rounds.numpy()[0] == 0
        assert w.wall_diagnostics.numpy()[0] > 0
        assert w._reuse_completion.max_rounds == 4
        check_face_index(w, tets)


@pytest.mark.parametrize("failure", ["negative-count", "malformed-all", "final-count"])
def test_borrowed_prefix_rejects_corruption_and_remembers_only_complete_output(
    monkeypatch, failure
):
    p = cloud()
    w = DelaunayWorkspace(p, cache_strategy=False, fast_candidates=False)
    if failure == "final-count":
        original = w._reuse_completion.run

        def corrupt():
            original()
            w._reuse_completion.tetrahedra_count.fill_(len(w.tetrahedra) + 1)

        monkeypatch.setattr(w._reuse_completion, "run", corrupt)
    else:
        original = w._wall.run

        def corrupt():
            original()
            if failure == "negative-count":
                w._wall.raw_count.fill_(-1)
            else:
                w._wall.raw_tets.fill_(-1)

        monkeypatch.setattr(w._wall, "run", corrupt)
    with wp.ScopedCapture(device=w.device) as capture:
        w.clear_reuse()
        w.run()
    wp.capture_launch(capture.graph)
    tets = check(w, p)
    assert w.wall_attempted.numpy()[0] == 1 and w.wall_accepted.numpy()[0] == 0
    assert w.base_attempted.numpy()[0] == 1
    assert w._previous_count.numpy()[0] == len(tets)
    np.testing.assert_array_equal(w._previous.numpy()[: len(tets)], tets)
    check_face_index(w, tets)


def test_small_private_reuse_keeps_original_storage_and_construction():
    p = cloud(128)
    w = DelaunayWorkspace(p).run()
    assert w._wall is None
    assert w.base_attempted.numpy()[0] == 1
    check(w, p)
