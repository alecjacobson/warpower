"""Separate post-proof transaction guards from exact proof and integrated replay tests."""

from pathlib import Path

import numpy as np
import pytest
import warp as wp

from tests.test_domain import cube
from tests.test_exterior_band import patch
from warpower import ExtractionWorkspace, SurfaceDomain
from warpower._proven_exterior import remove_proved

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.kernel
def recorded_proofs(
    count: wp.array[int], selected: wp.array[int], proved: wp.array[int], removed: wp.array[int]
):
    i = wp.tid()
    if i < count[0] and selected[i] != 0 and proved[i] != 0:
        selected[i] = 0
        wp.atomic_add(removed, 0, 1)


def guard_fixture(points, tets, distances, proved, keep=None):
    # This fixture deliberately supplies proof decisions to isolate transaction
    # policy. Exact geometry and full queries are exercised separately below.
    domain = SurfaceDomain(*cube(5))
    e = ExtractionWorkspace(
        domain, points, tets, exterior_recovery=True, proven_exterior_recovery=True
    )
    proof = e._proven_exterior
    keep = np.ones(len(tets), np.int32) if keep is None else keep
    p = points[tets].astype(float)
    volumes = np.linalg.det(p[:, 1:] - p[:, :1]) / 6
    assert (volumes > 0).all()
    sources = [
        wp.array(keep, dtype=int),
        wp.array(distances, dtype=wp.float64),
        wp.array(volumes, dtype=wp.float64),
        wp.array(proved, dtype=int),
    ]
    for dst, src in zip([e.selected, e.signed_distance, e._input_volume], sources):
        wp.copy(dst, src)
    e._input_quality.fill_(1)
    launch = proof.launch

    def fixture_launch(kernel, dim, args):
        if kernel is remove_proved:
            wp.launch(
                recorded_proofs, dim, inputs=[e.input_count, e.selected, sources[3], proof.removed]
            )
        else:
            launch(kernel, dim, args)

    proof.launch = fixture_launch
    wp.load_module(module=__name__, device=e.device)
    with wp.ScopedCapture() as capture:
        e._recompact()
        proof.run()
    wp.capture_launch(capture.graph)
    wp.synchronize()
    return e, proof


def test_post_proof_acceptance_preserves_input_and_all_references():
    p, t, distances, removed = patch()
    proved = np.zeros(len(t), np.int32)
    proved[removed] = 1
    e, r = guard_fixture(p, t, distances, proved)
    assert (
        e.diagnostics.numpy()[0] == 0 and r.removed.numpy()[0] == 1 and r.rejected.numpy()[0] == 0
    )
    np.testing.assert_array_equal(np.flatnonzero(e.selected.numpy() == 0), [removed])
    np.testing.assert_array_equal(e.points.numpy(), p)
    np.testing.assert_array_equal(e.input_tetrahedra.numpy(), t)
    np.testing.assert_array_equal(np.unique(t[e.selected.numpy() != 0]), np.unique(t))


@pytest.mark.parametrize("failure", ["volume", "orphan", "unresolved"])
def test_post_proof_guards_restore_entire_selection(failure):
    p, t, distances, removed = patch()
    keep = np.ones(len(t), np.int32)
    proved = np.zeros(len(t), np.int32)
    proved[removed] = 1
    if failure == "volume":
        keep[-1] = 0
    elif failure == "orphan":
        other = next(
            i
            for i, cell in enumerate(t)
            if distances[i] <= 0 and len(set(t[removed]) & set(cell)) == 2
        )
        keep[:] = 0
        keep[[removed, other, len(t) - 1]] = 1
    else:
        q = p.copy()
        q[:, 2] += 3
        p = np.vstack([p, q])
        t = np.vstack([t, t + len(q)])
        distances = np.r_[distances, distances]
        keep = np.r_[keep, keep]
        proved = np.r_[proved, np.zeros(len(proved), np.int32)]
    e, r = guard_fixture(p, t, distances, proved, keep)
    np.testing.assert_array_equal(e.selected.numpy(), keep)
    assert (
        r.rejected.numpy()[0] == 1 and r.removed.numpy()[0] == 0 and e.diagnostics.numpy()[0] == 2
    )
    if failure == "volume":
        assert not np.any(r.references.numpy() == 1)
    if failure == "orphan":
        assert np.any(r.references.numpy() == 1)


def positive(tets, points):
    tets = tets.copy()
    for t in tets:
        p = points[t].astype(float)
        if np.linalg.det(p[1:] - p[0]) < 0:
            t[1], t[2] = t[2], t[1]
    return tets


@pytest.mark.parametrize("exterior_recovery", [False, True])
def test_real_point_contact_proof_cannot_orphan_exterior_sites(exterior_recovery):
    p = np.array(
        [
            [1, 0, 0],
            [0.5, 0, 0],
            [0.5, 0.5, 0],
            [0.5, 0, 0.5],
            [1, 0.04, 0],
            [1, 0, 0.04],
            [1.04, 0.02, 0.02],
        ],
        np.float32,
    )
    t = positive(np.array([[0, 1, 2, 3], [0, 4, 5, 6]], np.int32), p)
    e = ExtractionWorkspace(
        SurfaceDomain(*cube()),
        p,
        t,
        exterior_recovery=exterior_recovery,
        proven_exterior_recovery=True,
    )
    r = e._proven_exterior
    with wp.ScopedCapture() as capture:
        e.run()
    wp.capture_launch(capture.graph)
    assert e.error.numpy()[0] == 0 and e.diagnostics.numpy()[0] == 4 and e.valid.numpy()[0] == 0
    assert r.rejected.numpy()[0] == 1 and r.removed.numpy()[0] == 0
    assert np.any(r.references.numpy() == 1)
    np.testing.assert_array_equal(e.selected.numpy(), [1, 1])
    # This case isolates vertex loss: volume loss alone is below one percent.
    c = p[t].astype(float)
    v = np.linalg.det(c[:, 1:] - c[:, :1]) / 6
    assert v[1] / v.sum() < 0.01


def test_original_unrounded_hand_rolls_back_partial_repair():
    d = np.load(Path(__file__).parent / "data/proven-exterior-original.npz")
    domain = SurfaceDomain(d["input_points"], d["input_faces"])
    e = ExtractionWorkspace(
        domain,
        d["points"],
        d["full_tetrahedra"],
        containment_recovery=True,
        exterior_recovery=True,
        proven_exterior_recovery=True,
        feature_roles=d["roles"],
    )
    r = e._proven_exterior
    with wp.ScopedCapture() as capture:
        e.run()
    wp.capture_launch(capture.graph)
    assert e.error.numpy()[0] == 0 and e.diagnostics.numpy()[0] == 2 and e.valid.numpy()[0] == 0
    assert r.rejected.numpy()[0] == 1 and r.removed.numpy()[0] == 0
    np.testing.assert_array_equal(e.selected.numpy(), d["selected"])


@wp.kernel
def produce(
    source: wp.array[wp.vec3],
    points: wp.array[wp.vec3],
    phase: wp.array[int],
    count: wp.array[int],
    upstream: wp.array[int],
    enabled: wp.array[int],
    n: int,
):
    i = wp.tid()
    points[i] = source[i]
    if i == 0:
        count[0] = n
        upstream[0] = 0
        enabled[0] = 1
        if phase[0] == 1:
            upstream[0] = 1
        if phase[0] == 2:
            count[0] = 0
        if phase[0] == 3:
            points[0] = wp.vec3(wp.nan)
        if phase[0] == 5:
            enabled[0] = 0


@wp.kernel
def consume(
    phase: wp.array[int],
    condition: wp.array[int],
    error: wp.array[int],
    valid: wp.array[int],
    count: wp.array[int],
    diagnostics: wp.array[int],
    removed: wp.array[int],
    rejected: wp.array[int],
    volume: wp.array[wp.float64],
    rows: wp.array2d[int],
    values: wp.array[wp.float64],
):
    i = phase[0]
    rows[i, 0] = error[0]
    rows[i, 1] = valid[0]
    rows[i, 2] = count[0]
    rows[i, 3] = diagnostics[0]
    rows[i, 4] = removed[0]
    rows[i, 5] = rejected[0]
    values[i] = wp.float64(0)
    if error[0] == 0 and valid[0] != 0:
        values[i] = volume[0]
    phase[0] += 1
    condition[0] = int(phase[0] < 7)


def test_captured_producer_consumer_failure_empty_disable_and_recovery(monkeypatch, tmp_path):
    d = np.load(Path(__file__).parent / "data/proven-exterior-continuation.npz")
    domain = SurfaceDomain(d["input_points"], d["input_faces"])
    p = np.vstack([d["points"], np.full((16, 3), np.nan, np.float32)]).astype(np.float32)
    source = wp.array(p, dtype=wp.vec3)
    points = wp.clone(source)
    t = np.vstack([d["full_tetrahedra"], np.full((16, 4), -1, np.int32)])
    n = len(t) - 16
    count = wp.array([n], dtype=int)
    upstream = wp.zeros(1, dtype=int)
    roles = wp.array(np.r_[d["roles"], np.zeros(16, np.int32)], dtype=int)
    e = ExtractionWorkspace(
        domain,
        points,
        t,
        count,
        source_error=upstream,
        containment_recovery=True,
        exterior_recovery=True,
        proven_exterior_recovery=True,
        feature_roles=roles,
    )
    r = e._proven_exterior
    r.enabled = wp.ones(1, dtype=int)
    phase = wp.zeros(1, dtype=int)
    condition = wp.ones(1, dtype=int)
    rows = wp.empty((7, 6), dtype=int)
    values = wp.empty(7, dtype=wp.float64)
    wp.load_module(module=__name__, device=e.device)

    def forbidden(*args, **kwargs):
        raise AssertionError("host activity inside capture/replay")

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

        def step():
            wp.launch(
                produce, len(points), inputs=[source, points, phase, count, upstream, r.enabled, n]
            )
            e.run()
            wp.launch(
                consume,
                1,
                inputs=[
                    phase,
                    condition,
                    e.error,
                    e.valid,
                    e.tetrahedra_count,
                    e.diagnostics,
                    r.removed,
                    r.rejected,
                    e.volume,
                    rows,
                    values,
                ],
            )

        with wp.ScopedCapture() as capture:
            phase.zero_()
            condition.fill_(1)
            wp.capture_while(condition, step)
        wp.capture_launch(capture.graph)
    actual = rows.numpy()
    np.testing.assert_array_equal(actual[:, 0], [0, 2, 0, 4, 0, 0, 0])
    np.testing.assert_array_equal(actual[:, 1], [1, 0, 1, 0, 1, 0, 1])
    np.testing.assert_array_equal(actual[[0, 4, 6], 4], [9, 9, 9])
    assert np.all(actual[[1, 2, 3, 5], 4:] == 0)
    assert np.all(actual[[1, 2, 3], 2] == 0) and actual[5, 3] == 2
    np.testing.assert_allclose(values.numpy()[[0, 4, 6]], values.numpy()[0], rtol=1e-12)
    assert np.all(values.numpy()[[1, 2, 3, 5]] == 0)
    np.testing.assert_array_equal(points.numpy()[1000:].view(np.uint32), p[1000:].view(np.uint32))
    dot = tmp_path / "proof-transactions.dot"
    wp.capture_debug_dot_print(capture.graph, str(dot), verbose=True)
    assert all(s not in dot.read_text() for s in ["MEM_ALLOC", "MEM_FREE", "HOST"])
