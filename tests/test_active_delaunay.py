from itertools import product

import numpy as np
import pytest
import warp as wp

from tests.test_delaunay_recovery import exact_empty
from tests.test_tetrahedra import check_mesh, reference, tet_set
from warpower._active_delaunay import ActiveDelaunayWorkspace
from warpower._active_sites import prepare_mesh


@wp.kernel
def produce(
    frames: wp.array2d[wp.vec3],
    counts: wp.array[int],
    phase: wp.array[int],
    points: wp.array[wp.vec3],
    active: wp.array[int],
):
    i = wp.tid()
    points[i] = frames[phase[0], i]
    if i == 0:
        active[0] = counts[phase[0]]


@wp.kernel
def consume(
    phase: wp.array[int],
    tets: wp.array2d[int],
    count: wp.array[int],
    error: wp.array[int],
    selected: wp.array[int],
    history: wp.array3d[int],
    meta: wp.array2d[int],
):
    i, row = wp.tid(), phase[0]
    if i < count[0]:
        for k in range(4):
            history[row, i, k] = tets[i, k]
    if i == 0:
        meta[row, 0] = count[0]
        meta[row, 1] = error[0]
        meta[row, 2] = selected[0]


@wp.kernel
def advance(phase: wp.array[int]):
    phase[0] += 1


def test_changing_prefixes_producers_consumers_and_failures_in_one_graph(monkeypatch, tmp_path):
    counts = [16, 17, 48, 49, 96, 32, 0, 97, -1, 3, 24, 16, 16, 16]
    rng = np.random.default_rng(1210)
    frames = rng.normal(size=(len(counts), 96, 3)).astype(np.float32)
    frames[10, 12:24] = frames[10, :12]
    frames[11, 4, 0] = np.nan
    frames[12, :16, 2] = 0  # No 3D hull: finite explicit failure.
    for frame, count in zip(frames, counts):
        if 4 <= count <= 96:
            frame[count:] = np.nan
    device = "cuda:0"
    source = wp.array(frames, dtype=wp.vec3, device=device)
    sizes = wp.array(counts, dtype=int, device=device)
    phase = wp.zeros(1, dtype=int, device=device)
    points = wp.zeros(96, dtype=wp.vec3, device=device)
    prepared = wp.zeros_like(points)
    active = wp.zeros(1, dtype=int, device=device)
    w = ActiveDelaunayWorkspace(prepared, active_count=active, prefix_capacities=(16, 48))
    history = wp.empty((len(counts), len(w.tetrahedra), 4), dtype=int, device=device)
    meta = wp.empty((len(counts), 3), dtype=int, device=device)
    wp.load_module(module=__name__, device=device)
    wp.load_module(module="warpower._active_sites", device=device)
    pointers = (w.tetrahedra.ptr, w.tetrahedra_count.ptr, w.points.ptr)
    assert not hasattr(w, "_certificate")
    stream = wp.Stream(device)

    def forbidden(*args, **kwargs):
        raise AssertionError("Host allocation/readback/synchronization during captured work")

    with monkeypatch.context() as patch:
        for name in (
            "empty",
            "zeros",
            "ones",
            "full",
            "empty_like",
            "clone",
            "synchronize",
            "synchronize_device",
            "synchronize_stream",
        ):
            patch.setattr(wp, name, forbidden)
        patch.setattr(wp.array, "numpy", forbidden)
        with wp.ScopedStream(stream), wp.ScopedCapture(stream=stream) as capture:
            wp.launch(produce, w.n, inputs=[source, sizes, phase, points, active])
            wp.launch(prepare_mesh, w.n, inputs=[points, active, prepared])
            w.run(stream)
            wp.launch(
                consume,
                len(w.tetrahedra),
                inputs=[
                    phase,
                    w.tetrahedra,
                    w.tetrahedra_count,
                    w.error,
                    w.selected_capacity,
                    history,
                    meta,
                ],
            )
            wp.launch(advance, 1, inputs=[phase])
        for _ in counts:
            wp.capture_launch(capture.graph, stream=stream)
    wp.synchronize_stream(stream)
    rows, cells = meta.numpy(), history.numpy()
    for i, count in enumerate(counts):
        if i in (6, 7, 8, 9, 11, 12):
            assert rows[i, 0] == 0 and rows[i, 1] != 0, (i, rows[i])
            if i in (6, 7, 8, 9):
                assert rows[i, 1:].tolist() == [8, 0]
        else:
            assert rows[i, 0] > 0 and rows[i, 1] == 0, (i, rows[i])
            cloud = frames[i, :count]
            owners = np.unique(cloud, axis=0, return_index=True)[1]
            expected = {tuple(sorted(owners[list(tet)])) for tet in reference(cloud[owners])}
            actual = cells[i, : rows[i, 0]]
            assert tet_set(actual) == expected
            check_mesh(cloud, actual)
            assert rows[i, 2] == next(n for n in w.capacities if count <= n)
    assert pointers == (w.tetrahedra.ptr, w.tetrahedra_count.ptr, w.points.ptr)
    np.testing.assert_array_equal(points.numpy().view(np.uint32), frames[-1].view(np.uint32))
    dot = tmp_path / "active-delaunay.dot"
    wp.capture_debug_dot_print(capture.graph, str(dot), verbose=True)
    assert all(node not in dot.read_text() for node in ("MEM_ALLOC", "MEM_FREE", "HOST"))


def test_cospherical_first_duplicate_owners_and_one_ulp_update():
    cloud = np.array(list(product([-1, 1], repeat=3)), np.float32)
    cloud = np.vstack([cloud, cloud[:2]])
    source = np.tile(cloud[0], (64, 1))
    source[: len(cloud)] = cloud
    points = wp.array(source, dtype=wp.vec3, device="cuda:0")
    active = wp.array([len(cloud)], dtype=int, device="cuda:0")
    w = ActiveDelaunayWorkspace(points, active_count=active, prefix_capacities=(16, 32))
    for step in range(3):
        if step == 2:
            source[0, 0] = np.nextafter(source[0, 0], np.float32(0))
            source[len(cloud) :] = source[0]
            points.assign(source)
        w.run()
        assert w.error.numpy()[0] == 0
        tets = w.tetrahedra.numpy()[: w.tetrahedra_count.numpy()[0]]
        check_mesh(source[: len(cloud)], tets)
        exact_empty(source[: len(cloud)], tets)
        assert tets.max() < len(cloud)


def test_output_capacity_failure_and_fresh_reset_are_capturable():
    source = np.random.default_rng(1211).normal(size=(64, 3)).astype(np.float32)
    source[16:] = source[0]
    points = wp.array(source, dtype=wp.vec3, device="cuda:0")
    active = wp.array([16], dtype=int, device="cuda:0")
    w = ActiveDelaunayWorkspace(
        points, active_count=active, prefix_capacities=(16, 32), tetrahedra_capacity=1
    )
    with wp.ScopedCapture() as capture:
        w.clear_reuse()
        w.run()
    for _ in range(2):
        wp.capture_launch(capture.graph)
        assert w.error.numpy()[0] != 0 and w.tetrahedra_count.numpy()[0] == 0
        assert w.route_calls.numpy().tolist() == [1, 0, 1]
        assert w.selected_capacity.numpy()[0] == 64
        assert all(child._previous_count.numpy()[0] == 0 for child in w._children)


def test_failed_small_constructor_falls_back_to_full_certified_hull():
    source = np.random.default_rng(1212).normal(size=(64, 3)).astype(np.float32)
    source[16:] = source[0]
    points = wp.array(source, dtype=wp.vec3, device="cuda:0")
    active = wp.array([16], dtype=int, device="cuda:0")
    w = ActiveDelaunayWorkspace(points, active_count=active, prefix_capacities=(16, 32))

    def fail(stream=None):
        w._children[0].error.fill_(16)
        w._children[0].tetrahedra_count.zero_()

    w._children[0].run = fail
    with wp.ScopedCapture() as capture:
        w.run()
    wp.capture_launch(capture.graph)
    assert w.error.numpy()[0] == w.certificate_error.numpy()[0] == 0
    assert w.route_calls.numpy().tolist() == [1, 0, 1]
    assert w.selected_capacity.numpy()[0] == 64
    cells = w.tetrahedra.numpy()[: w.tetrahedra_count.numpy()[0]]
    check_mesh(source[:16], cells)
    assert tet_set(cells) == reference(source[:16])


@pytest.mark.parametrize("levels", [(), (3,), (32, 16), (16, 16), (64,), (2049,)])
def test_rejects_invalid_prefix_capacities(levels):
    points = wp.zeros(64, dtype=wp.vec3, device="cuda:0")
    active = wp.array([16], dtype=int, device="cuda:0")
    with pytest.raises(ValueError, match="Prefix capacities"):
        ActiveDelaunayWorkspace(points, active_count=active, prefix_capacities=levels)


def test_vtm_rollback_restores_active_mesh_across_routed_construction(monkeypatch):
    import warpower._active_delaunay as routed
    from tests.test_active_sites import (
        test_active_rollback_restores_mesh_scratch_and_preserves_unused_points,
    )
    from warpower import _vtm

    # Exercise the unchanged VTM rollback with a small routed constructor.
    def create(points, **kwargs):
        return routed.ActiveDelaunayWorkspace(
            points, active_count=count, prefix_capacities=(48, 64), **kwargs
        )

    original = _vtm.VTMWorkspace.__init__
    count = None

    def initialize(self, *args, **kwargs):
        nonlocal count
        count = kwargs["active_count"]
        original(self, *args, **kwargs)

    monkeypatch.setattr(_vtm.VTMWorkspace, "__init__", initialize)
    monkeypatch.setattr(_vtm, "DelaunayWorkspace", create)
    test_active_rollback_restores_mesh_scratch_and_preserves_unused_points()
