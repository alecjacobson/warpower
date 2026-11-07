from itertools import combinations, product

import numpy as np
import pytest
import warp as wp

from tests.test_active_delaunay import advance, consume, produce
from tests.test_delaunay_recovery import exact_empty
from tests.test_local_certificate import positive
from tests.test_predicates import exact_reference
from tests.test_proved_completion import corrupt_new
from tests.test_tetrahedra import check_mesh, reference, tet_set
from warpower._active_delaunay import ActiveDelaunayWorkspace
from warpower._active_sites import prepare_mesh
from warpower._reuse_delaunay import ReuseDelaunayWorkspace


@wp.kernel
def consume_reuse(
    phase: wp.array[int],
    attempted: wp.array[int],
    accepted: wp.array[int],
    proofs: wp.array[int],
    history: wp.array2d[int],
):
    row = phase[0]
    history[row, 0] = attempted[0]
    history[row, 1] = accepted[0]
    history[row, 2] = proofs[0]
    history[row, 3] = proofs[1]


@wp.kernel
def consume_whole(phase:wp.array[int],status:wp.array[int],history:wp.array2d[int]):
    i=wp.tid()
    history[phase[0],i]=status[i]


@wp.kernel
def remove_last_cell(count: wp.array[int]):
    count[0] -= 1


@pytest.mark.parametrize("capacity", [2050, 4096])
def test_large_route_changes_inputs_and_prefixes_without_host_decisions(capacity, monkeypatch):
    rng = np.random.default_rng(161)
    initial = rng.normal(size=(capacity, 3)).astype(np.float32)
    nearby = initial.copy()
    nearby[5] += np.float32(1e-5)
    unrelated = rng.normal(size=(capacity, 3)).astype(np.float32)
    invalid = initial.copy()
    invalid[4, 0] = np.nan
    counts = [capacity, capacity, 512, capacity, capacity + 1, capacity, capacity, capacity]
    frames = np.stack([initial, nearby, initial, unrelated, initial, initial, invalid, initial])
    frames[2, 512:] = np.nan
    device = "cuda:0"
    source = wp.array(frames, dtype=wp.vec3, device=device)
    sizes = wp.array(counts, dtype=int, device=device)
    phase = wp.zeros(1, dtype=int, device=device)
    points = wp.zeros(capacity, dtype=wp.vec3, device=device)
    prepared = wp.zeros_like(points)
    active = wp.zeros(1, dtype=int, device=device)
    w = ActiveDelaunayWorkspace(prepared, active_count=active)
    full = w._full
    history = wp.empty((len(counts), len(w.tetrahedra), 4), dtype=int, device=device)
    meta = wp.empty((len(counts), 3), dtype=int, device=device)
    reuse = wp.empty((len(counts), 4), dtype=int, device=device)
    for module in [__name__, "tests.test_active_delaunay", "warpower._active_sites"]:
        wp.load_module(module=module, device=device)
    whole = wp.empty((len(counts),3),dtype=int,device=device)
    pointers = w.tetrahedra.ptr, w.tetrahedra_count.ptr, w.points.ptr
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
            wp.launch(
                consume_reuse,
                1,
                inputs=[
                    phase,
                    full.reuse_attempted,
                    full.reuse_accepted,
                    full._reuse_completion._certificate.proof_statistics,
                    reuse,
                ],
            )
            wp.launch(consume_whole,3,inputs=[phase,full._whole_status,whole])
            wp.launch(advance, 1, inputs=[phase])
        for _ in counts:
            wp.capture_launch(capture.graph, stream=stream)
    wp.synchronize_stream(stream)
    rows, cells, reused = meta.numpy(), history.numpy(), reuse.numpy()
    for i, count in enumerate(counts):
        if i in (4, 6):
            assert rows[i, 0] == 0 and rows[i, 1] != 0
        else:
            assert rows[i, 0] > 0 and rows[i, 1] == 0, (i, rows[i])
            cloud = frames[i, :count]
            actual = cells[i, : rows[i, 0]]
            assert tet_set(actual) == reference(cloud)
            check_mesh(cloud, actual)
            assert rows[i, 2] == (1024 if i == 2 else capacity)
    assert reused[0, 0] == 0
    assert reused[1, :2].tolist() == [1, 1]
    assert whole.numpy()[1].tolist() == [1,1,0]  # Fresh whole-mesh certificate.
    assert reused[1,2] == 0  # No fragment proof-consumption count on that branch.
    assert reused[7, 0] == 0  # Nonfinite input clears the full retained fragment.
    assert pointers == (w.tetrahedra.ptr, w.tetrahedra_count.ptr, w.points.ptr)
    np.testing.assert_array_equal(points.numpy().view(np.uint32), initial.view(np.uint32))

    # A fresh solve must not inherit the preceding solution's fragment.
    assert full._previous_count.numpy()[0] > 0
    with wp.ScopedCapture() as fresh:
        w.clear_reuse()
        w.run()
    wp.capture_launch(fresh.graph)
    assert w.error.numpy()[0] == 0 and full.reuse_attempted.numpy()[0] == 0
    assert w.route_calls.numpy().tolist() == [0, 0, 1]
    assert all(child._previous_count.numpy()[0] == 0 for child in w._children)
    actual = w.tetrahedra.numpy()[: w.tetrahedra_count.numpy()[0]]
    assert tet_set(actual) == reference(initial)


def test_large_unproved_cell_corruption_requires_ordinary_fallback():
    points = np.random.default_rng(162).normal(size=(2050, 3)).astype(np.float32)
    w = ReuseDelaunayWorkspace(points)
    w.run()
    assert w.error.numpy()[0] == 0
    expected = reference(points)
    bad_key = next(k for k in combinations(range(len(points)), 4) if k not in expected)
    bad = positive(points, np.array([bad_key], np.int32))[0]
    p = points[bad].astype(np.float64)
    edges = p[1:] - p[0]
    offset = np.linalg.solve(2 * edges, np.sum(edges * edges, axis=1))
    distance2 = np.sum((points.astype(np.float64) - p[0] - offset) ** 2, axis=1)
    queries = np.flatnonzero(distance2 < np.dot(offset, offset) * 0.99)
    query = next(int(i) for i in queries if i not in bad)
    assert exact_reference(points[[*bad, query]], np.zeros(5))[1] > 0
    completion = w._reuse_completion
    original = completion._deduplicate

    def inject():
        original()
        wp.launch(
            corrupt_new,
            completion.capacity,
            inputs=[
                completion._work,
                completion._count,
                completion._certificate._input_proved,
                wp.vec4i(*map(int, bad)),
            ],
            device=w.device,
        )

    completion._deduplicate = inject
    wp.load_module(module=__name__, device=w.device)
    wp.load_module(module="tests.test_proved_completion", device=w.device)
    with wp.ScopedCapture() as capture:
        wp.launch(remove_last_cell, 1, inputs=[w._previous_count], device=w.device)
        w.run()
    wp.capture_launch(capture.graph)
    assert w.reuse_attempted.numpy()[0] == 1
    assert w.reuse_accepted.numpy()[0] == 0 and w.reuse_error.numpy()[0] != 0
    assert completion.certificate_error.numpy()[0] & 8
    assert np.all(completion._certificate.proof_statistics.numpy() > 0)
    assert w.error.numpy()[0] == 0
    actual = w.tetrahedra.numpy()[: w.tetrahedra_count.numpy()[0]]
    assert tet_set(actual) == expected
    check_mesh(points, actual)


def test_large_output_exhaustion_does_not_retain_or_publish_partial_cells():
    points = wp.array(
        np.random.default_rng(163).normal(size=(2050, 3)).astype(np.float32),
        dtype=wp.vec3,
        device="cuda:0",
    )
    active = wp.full(1, len(points), dtype=int, device=points.device)
    w = ActiveDelaunayWorkspace(points, active_count=active, tetrahedra_capacity=1)
    with wp.ScopedCapture() as capture:
        w.clear_reuse()
        w.run()
    for _ in range(2):
        wp.capture_launch(capture.graph)
        assert w.error.numpy()[0] != 0 and w.tetrahedra_count.numpy()[0] == 0
        assert w._full._previous_count.numpy()[0] == 0


@pytest.mark.parametrize("scale", [2.0**-60, 1.0, 2.0**60])
def test_large_full_route_preserves_cospherical_first_duplicate_owners(scale):
    cloud = np.array(list(product([-1, 1], repeat=3)), np.float32) * np.float32(scale)
    source = np.tile(cloud[0], (4096, 1))
    source[: len(cloud)] = cloud
    points = wp.array(source, dtype=wp.vec3, device="cuda:0")
    active = wp.full(1, len(points), dtype=int, device=points.device)
    w = ActiveDelaunayWorkspace(points, active_count=active)
    with wp.ScopedCapture() as capture:
        w.run()
    for _ in range(2):
        wp.capture_launch(capture.graph)
        assert w.error.numpy()[0] == 0
        actual = w.tetrahedra.numpy()[: w.tetrahedra_count.numpy()[0]]
        assert actual.max() < len(cloud)
        check_mesh(cloud, actual)
        exact_empty(cloud, actual)
    assert w._full.reuse_attempted.numpy()[0] == 1
