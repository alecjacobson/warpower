"""Transaction safeguards on a reduced independently inspected bad edge star."""

import json
from pathlib import Path

import numpy as np
import pytest
import warp as wp

import warpower
from tests.test_domain import cube
from tests.test_extraction import independent_topology


def patch():
    data = json.loads((Path(__file__).parent / "data/exterior-band-patch.json").read_text())
    points = np.array(data["points"], np.float32)
    tets = np.array(data["tetrahedra"], np.int32)
    # A separate interior cell represents the rest of the selected material.
    # The local repair must also obey a global one-percent volume budget.
    points = np.vstack([points, [[2, 0, 0], [3, 0, 0], [2, 1, 0], [2, 0, 1]]]).astype(np.float32)
    tets = np.vstack([tets, np.arange(len(points) - 4, len(points))]).astype(np.int32)
    distances = np.array(data["signed_distance"] + [-1], np.float64)
    return points, tets, distances, data["removed"]


def transaction(points, tets, distances, keep=None, roles=None):
    # This exercises the post-classification transaction, using recorded signed
    # distances. It is deliberately independent of the surface-query algorithm.
    d = warpower.SurfaceDomain(*cube(5))
    roles = np.zeros(len(points), np.int32) if roles is None else roles
    e = warpower.ExtractionWorkspace(d, points, tets, exterior_recovery=True, feature_roles=roles)
    keep = np.ones(len(tets), np.int32) if keep is None else keep
    wp.copy(e.selected, wp.array(keep, device=e.device))
    wp.copy(e.signed_distance, wp.array(distances, device=e.device))
    p = points[tets].astype(np.float64)
    volumes = np.linalg.det(p[:, 1:] - p[:, :1]) / 6
    assert np.all(volumes > 0)
    wp.copy(e._input_volume, wp.array(volumes, device=e.device))
    e._input_quality.fill_(1)
    with wp.ScopedCapture() as cap:
        e._recompact()
        e._exterior.run()
    wp.capture_launch(cap.graph)
    return e, cap.graph


def test_reduced_exterior_star_preserves_vertices_interior_cells_and_input():
    p, t, distances, removed = patch()
    e, _ = transaction(p, t, distances)
    keep = e.selected.numpy()
    np.testing.assert_array_equal(np.flatnonzero(keep == 0), [removed])
    assert e.diagnostics.numpy()[0] == e.exterior_rejected.numpy()[0] == 0
    assert e.removed_count.numpy()[0] == e.exterior_rounds.numpy()[0] == 1
    np.testing.assert_array_equal(e.input_tetrahedra.numpy(), t)
    np.testing.assert_array_equal(e.points.numpy(), p)
    np.testing.assert_array_equal(np.unique(t[keep != 0]), np.unique(t))
    assert np.all(keep[distances <= 0] == 1)
    _, components, euler = independent_topology(p, t[keep != 0])
    assert components == 2 and euler == 4


@pytest.mark.parametrize("protection", ["interior", "on_surface", "crease", "corner"])
def test_interior_surface_and_feature_cells_cannot_be_removed(protection):
    p, t, distances, removed = patch()
    roles = np.zeros(len(p), np.int32)
    if protection in ["interior", "on_surface"]:
        distances[removed] = -1 if protection == "interior" else 0
    else:
        roles[t[removed, 0]] = 2 if protection == "crease" else 3
    e, _ = transaction(p, t, distances, roles=roles)
    np.testing.assert_array_equal(e.selected.numpy(), 1)
    assert e.diagnostics.numpy()[0] == 2 and e.exterior_rejected.numpy()[0] == 1
    assert e.removed_count.numpy()[0] == 0


def test_orphaned_vertex_rejects_transaction_even_if_remaining_cells_are_manifold():
    p, t, distances, removed = patch()
    # These two tets meet only along the original bad edge; deleting the
    # exterior one would lose its two other vertices. The large cell remains.
    # Derive the shared edge directly from the fixture's only nonmanifold edge.
    candidate = t[removed]
    other = next(
        i for i, tet in enumerate(t) if distances[i] <= 0 and len(set(candidate) & set(tet)) == 2
    )
    keep = np.zeros(len(t), np.int32)
    keep[[removed, other, len(t) - 1]] = 1
    e, _ = transaction(p, t, distances, keep=keep)
    np.testing.assert_array_equal(e.selected.numpy(), keep)
    assert np.any(e._exterior.references.numpy() == 1)
    assert e.exterior_rejected.numpy()[0] == 1 and e.removed_count.numpy()[0] == 0
    assert e.diagnostics.numpy()[0] == 2


def test_large_volume_change_rolls_back_an_otherwise_manifold_repair():
    p, t, distances, _ = patch()
    keep = np.ones(len(t), np.int32)
    keep[-1] = 0
    e, _ = transaction(p, t, distances, keep=keep)
    np.testing.assert_array_equal(e.selected.numpy(), keep)
    assert e.exterior_rejected.numpy()[0] == 1 and e.removed_count.numpy()[0] == 0
    assert not np.any(e._exterior.references.numpy() == 1)
    assert e.diagnostics.numpy()[0] == 2


def test_residual_bad_edge_rolls_back_all_earlier_changes():
    p, t, distances, removed = patch()
    q = p.copy()
    q[:, 2] += 3
    p = np.vstack([p, q])
    t = np.vstack([t, t + len(q)])
    second_distances = distances.copy()
    second_distances[removed] = -1
    e, _ = transaction(p, t, np.r_[distances, second_distances])
    np.testing.assert_array_equal(e.selected.numpy(), 1)
    assert e.exterior_rounds.numpy()[0] == 2
    assert e.exterior_rejected.numpy()[0] == 1 and e.removed_count.numpy()[0] == 0
    assert e.diagnostics.numpy()[0] == 2


def test_independent_extractor_has_no_recovery_allocations_by_default():
    p, t, _, _ = patch()
    e = warpower.ExtractionWorkspace(warpower.SurfaceDomain(*cube(5)), p, t).run()
    assert e._exterior is None and not hasattr(e, "removed_count")
    np.testing.assert_array_equal(e.selected.numpy(), 1)


@wp.kernel
def producer(
    phase: wp.array[int],
    keep: wp.array[int],
    roles: wp.array[int],
    count: wp.array[int],
    error: wp.array[int],
    n: int,
    feature: int,
):
    i = wp.tid()
    if i < keep.shape[0]:
        keep[i] = int(i < n and phase[0] != 2)
    if i < roles.shape[0]:
        roles[i] = 0
        if i == feature and phase[0] == 1:
            roles[i] = 2
    if i == 0:
        count[0] = wp.where(phase[0] == 2, 0, n)
        error[0] = wp.where(phase[0] == 3, 4, 0)


@wp.kernel
def consumer(
    phase: wp.array[int],
    error: wp.array[int],
    diagnostics: wp.array[int],
    count: wp.array[int],
    removed: wp.array[int],
    rejected: wp.array[int],
    output: wp.array2d[int],
):
    k = phase[0]
    output[k, 0] = error[0]
    output[k, 1] = diagnostics[0]
    output[k, 2] = count[0]
    output[k, 3] = removed[0]
    output[k, 4] = rejected[0]
    phase[0] += 1


def test_device_producer_consumer_replay_resets_failure_empty_and_restored_frames(
    monkeypatch, tmp_path
):
    p, t, distances, removed = patch()
    e, _ = transaction(p, t, distances)
    n = len(t)
    # Add inaccessible invalid padding to exercise active-prefix reads.
    padded = wp.array(np.vstack([t, [-1, -1, -1, -1]]).astype(np.int32), device=e.device)
    e.input_tetrahedra = padded
    phase = wp.zeros(1, dtype=int, device=e.device)
    output = wp.empty((5, 5), dtype=int, device=e.device)
    wp.load_module(module=__name__, device=e.device)

    def forbidden(*args, **kwargs):
        raise AssertionError("Host allocation, sync or readback during captured recovery")

    with monkeypatch.context() as m:
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
            m.setattr(wp, name, forbidden)
        m.setattr(wp.array, "numpy", forbidden)
        with wp.ScopedCapture() as cap:
            wp.launch(
                producer,
                max(len(p), n),
                inputs=[
                    phase,
                    e.selected,
                    e.feature_roles,
                    e.input_count,
                    e.error,
                    n,
                    int(t[removed, 0]),
                ],
            )
            e._recompact()
            e._exterior.run()
            wp.launch(
                consumer,
                1,
                inputs=[
                    phase,
                    e.error,
                    e.diagnostics,
                    e.tetrahedra_count,
                    e.removed_count,
                    e.exterior_rejected,
                    output,
                ],
            )
        for _ in range(5):
            wp.capture_launch(cap.graph)
    np.testing.assert_array_equal(
        output.numpy(),
        [
            [0, 0, n - 1, 1, 0],
            [0, 2, n, 0, 1],
            [0, 0, 0, 0, 0],
            [4, 0, 0, 0, 0],
            [0, 0, n - 1, 1, 0],
        ],
    )
    dot = tmp_path / "exterior.dot"
    wp.capture_debug_dot_print(cap.graph, str(dot), verbose=True)
    text = dot.read_text()
    assert all(x not in text for x in ["MEM_ALLOC", "MEM_FREE", "HOST"])
