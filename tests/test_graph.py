import numpy as np
import pytest
import warp as wp

import warpower
from tests.test_tetrahedra import check_mesh, reference, tet_set


@wp.kernel
def produce(
    base: wp.array[wp.vec3],
    points: wp.array[wp.vec3],
    weights: wp.array[float],
    phase: wp.array[int],
):
    i = wp.tid()
    p = base[i]
    t = float(phase[0])
    points[i] = p + wp.vec3(0.12 * wp.sin(t + float(i)), 0.07 * wp.cos(2.0 * t + float(i)), 0.0)
    weights[i] = 0.05 * wp.sin(0.7 * t + float(i))


@wp.kernel
def advance(phase: wp.array[int]):
    phase[0] += 1


@wp.kernel
def consume(
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    count: wp.array[int],
    error: wp.array[int],
    volume: wp.array[wp.float64],
):
    i = wp.tid()
    if error[0] == 0 and i < count[0]:
        a = wp.vec3d(points[tets[i, 0]])
        b = wp.vec3d(points[tets[i, 1]]) - a
        c = wp.vec3d(points[tets[i, 2]]) - a
        d = wp.vec3d(points[tets[i, 3]]) - a
        wp.atomic_add(volume, 0, wp.dot(b, wp.cross(c, d)) / wp.float64(6.0))


@wp.kernel
def consume_adjacency(
    offsets: wp.array[int], adjacency: wp.array[int], error: wp.array[int], checksum: wp.array[int]
):
    i = wp.tid()
    if error[0] == 0:
        for k in range(offsets[i], offsets[i + 1]):
            wp.atomic_add(checksum, 0, adjacency[k])


def rows(result):
    def host(array):
        return array.numpy() if isinstance(array, wp.array) else array.cpu().numpy()

    offsets = host(result.offsets)
    adjacency = host(result.adjacency)
    return [set(adjacency[offsets[i] : offsets[i + 1]]) for i in range(len(offsets) - 1)]


@pytest.mark.parametrize("weighted", [False, True])
def test_captured_producer_geometry_consumer_changes_topology(weighted, monkeypatch):
    p = np.random.default_rng(93).normal(size=(80, 3)).astype(np.float32)
    base = wp.array(p, dtype=wp.vec3, device="cuda:0")
    points = wp.clone(base)
    weights = wp.zeros(len(p), dtype=float, device="cuda:0")
    phase = wp.zeros(1, dtype=int, device="cuda:0")
    volume = wp.zeros(1, dtype=wp.float64, device="cuda:0")
    checksum = wp.zeros(1, dtype=int, device="cuda:0")
    workspace = warpower.DiagramWorkspace(
        points, weights if weighted else None, return_tetrahedra=True, chunk_size=17
    )
    # Compile our producer/consumer before instrumenting library capture.
    wp.load_module(module=__name__, device="cuda:0")

    def forbidden(*args, **kwargs):
        raise AssertionError("Host readback, synchronization, or allocation during capture/run")

    with monkeypatch.context() as guard:
        guard.setattr(wp.array, "numpy", forbidden)
        for name in (
            "synchronize",
            "synchronize_stream",
            "synchronize_device",
            "empty",
            "zeros",
            "full",
        ):
            guard.setattr(wp, name, forbidden)
        with wp.ScopedCapture() as capture:
            wp.launch(produce, len(p), inputs=[base, points, weights, phase])
            workspace.run()
            volume.zero_()
            checksum.zero_()
            wp.launch(
                consume,
                len(workspace.tetrahedra),
                inputs=[
                    points,
                    workspace.tetrahedra,
                    workspace.tetrahedra_count,
                    workspace.error,
                    volume,
                ],
            )
            wp.launch(
                consume_adjacency,
                len(p),
                inputs=[workspace.offsets, workspace.adjacency, workspace.error, checksum],
            )
            wp.launch(advance, 1, inputs=[phase])
    topology = []
    for _ in range(3):
        with monkeypatch.context() as guard:
            guard.setattr(wp.array, "numpy", forbidden)
            for name in (
                "synchronize",
                "synchronize_stream",
                "synchronize_device",
                "empty",
                "zeros",
                "full",
            ):
                guard.setattr(wp, name, forbidden)
            wp.capture_launch(capture.graph)
        wp.synchronize()
        assert workspace.error.numpy()[0] == 0
        current = points.numpy()
        w = weights.numpy() if weighted else None
        count = int(workspace.tetrahedra_count.numpy()[0])
        tets = workspace.tetrahedra.numpy()[:count]
        check_mesh(current, tets)
        assert tet_set(tets) == reference(current, w)
        topology.append(tet_set(tets))
        eager = (
            warpower.power_diagram(current, w) if weighted else warpower.voronoi_diagram(current)
        )
        assert rows(workspace) == rows(eager)
        assert checksum.numpy()[0] == eager.adjacency.numpy().sum()
        from scipy.spatial import ConvexHull

        assert np.isclose(volume.numpy()[0], ConvexHull(current).volume, rtol=1e-9)
    assert topology[0] != topology[1]
    assert phase.numpy()[0] == 3


@pytest.mark.parametrize("mode,bit", [("adjacency", 4), ("tetrahedra", 8)])
def test_capacity_failure_zeroes_visible_output(mode, bit):
    points = wp.array(
        np.random.default_rng(5).normal(size=(32, 3)).astype(np.float32),
        dtype=wp.vec3,
        device="cuda:0",
    )
    kwargs = {f"{mode}_capacity": 1}
    workspace = warpower.DiagramWorkspace(points, return_tetrahedra=True, **kwargs)
    workspace.run()
    assert workspace.error.numpy()[0] & bit
    assert workspace.adjacency_count.numpy()[0] == 0
    assert workspace.tetrahedra_count.numpy()[0] == 0
    assert not workspace.offsets.numpy().any()


def test_invalid_and_degenerate_then_valid_replay():
    cube = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], np.float32)
    points = wp.array(cube, dtype=wp.vec3, device="cuda:0")
    workspace = warpower.DiagramWorkspace(points, return_tetrahedra=True)
    workspace.run()
    assert workspace.error.numpy()[0] & (2 | 16)
    assert workspace.tetrahedra_count.numpy()[0] == 0
    invalid = cube.copy()
    invalid[0, 0] = np.nan
    points.assign(invalid)
    workspace.run()
    assert workspace.error.numpy()[0] == 1
    assert not workspace.offsets.numpy().any()
    valid = np.random.default_rng(9).normal(size=(8, 3)).astype(np.float32)
    points.assign(valid)
    workspace.run()
    assert workspace.error.numpy()[0] == 0
    assert tet_set(
        workspace.tetrahedra.numpy()[: int(workspace.tetrahedra_count.numpy()[0])]
    ) == reference(valid)


def test_full_hull_expansion_and_device_domain_reuse():
    p = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.2, 0.2, 1e-4]], np.float32)
    workspace = warpower.DiagramWorkspace(p, adjacency=False, return_tetrahedra=True)
    workspace.run()
    assert workspace.error.numpy()[0] == 0
    assert workspace.tetrahedra_count.numpy()[0] == 1
    assert workspace.tetrahedra_expansions.numpy()[0] >= 4
    workspace.run()
    assert workspace.error.numpy()[0] == 0
    assert workspace.tetrahedra_expansions.numpy()[0] == 0
    limited = warpower.DiagramWorkspace(p, return_tetrahedra=True, max_expansions=1)
    limited.run()
    assert limited.error.numpy()[0] & 64
    assert limited.tetrahedra_count.numpy()[0] == 0


@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_small_graph(n):
    p = np.random.default_rng(45).normal(size=(n, 3)).astype(np.float32)
    workspace = warpower.DiagramWorkspace(p, return_tetrahedra=True)
    workspace.run()
    assert workspace.error.numpy()[0] == 0
    assert workspace.tetrahedra_count.numpy()[0] == 0
    assert rows(workspace) == rows(warpower.voronoi_diagram(p))


def test_duplicate_owners_and_weighted_hidden_sites():
    p = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.2, 0.2, 0.2]], np.float32)
    points = wp.array(np.repeat(p, 2, axis=0), dtype=wp.vec3, device="cuda:0")
    weights = wp.array(np.tile([0, 0.01], 5).astype(np.float32), device="cuda:0")
    workspace = warpower.DiagramWorkspace(points, weights, return_tetrahedra=True)
    for center_weight in (1000, -1000):
        w = np.tile([0, 0.01], 5).astype(np.float32)
        w[-2:] += center_weight
        weights.assign(w)
        workspace.run()
        assert workspace.error.numpy()[0] == 0
        t = workspace.tetrahedra.numpy()[: int(workspace.tetrahedra_count.numpy()[0])]
        assert np.all(t % 2 == 1)
        assert tet_set((t - 1) // 2) == reference(p, w[1::2])


def test_scan_multi_level_and_adjacency_recovery():
    from benchmarks.datasets import generate

    p, w = generate(10000, "cluster5", 42, False)
    workspace = warpower.DiagramWorkspace(p)
    workspace.run()
    assert workspace.error.numpy()[0] == 0
    assert rows(workspace) == rows(warpower.voronoi_diagram(p))


def test_torch_producer_and_consumer_on_shared_stream():
    torch = pytest.importorskip("torch")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        points = torch.tensor(
            np.random.default_rng(4).normal(size=(40, 3)), dtype=torch.float32, device="cuda:0"
        )
        workspace = warpower.DiagramWorkspace(points)
        offsets = wp.to_torch(workspace.offsets)
        degrees = torch.empty(40, dtype=torch.int32, device="cuda:0")
        total = torch.empty((), dtype=torch.int64, device="cuda:0")
        # Warm Torch operators before capture, with a shared Warp/Torch stream.
        bridge = wp.stream_from_torch(stream)
        workspace.run(stream=bridge)
        torch.sub(offsets[1:], offsets[:-1], out=degrees)
        torch.sum(degrees, dim=0, out=total)
        with wp.ScopedStream(bridge):
            with wp.ScopedCapture(stream=bridge) as capture:
                points.add_(0.01)
                workspace.run()
                torch.sub(offsets[1:], offsets[:-1], out=degrees)
                torch.sum(degrees, dim=0, out=total)
            wp.capture_launch(capture.graph)
            wp.capture_launch(capture.graph)
    stream.synchronize()
    assert workspace.error.numpy()[0] == 0
    assert total.item() == workspace.adjacency_count.numpy()[0]
    expected = warpower.voronoi_diagram(points)
    assert rows(workspace) == rows(expected)
