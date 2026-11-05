import numpy as np
import pytest
import warp as wp

from tests.oracle import reference_cells
from warpower._cell import allocate_scratch, clip_candidates


def compute(points, weights=None, device="cuda:0", capacity=128):
    points = np.asarray(points, dtype=np.float32)
    n = len(points)
    weights = (
        np.zeros(n, dtype=np.float32) if weights is None else np.asarray(weights, dtype=np.float32)
    )
    order = np.argsort(np.linalg.norm(points[:, None] - points[None, :], axis=2), axis=1).astype(
        np.int32
    )
    scratch = allocate_scratch(n, capacity, device)
    neighbors = wp.empty((n, capacity), dtype=int, device=device)
    counts = wp.empty(n, dtype=int, device=device)
    status = wp.empty(n, dtype=int, device=device)
    p_gpu = wp.array(points, dtype=wp.vec3, device=device)
    w_gpu = wp.array(weights, device=device)
    order_gpu = wp.array(order, device=device)
    wp.launch(
        clip_candidates,
        n,
        inputs=[
            p_gpu,
            w_gpu,
            order_gpu,
            wp.vec3(points.min(axis=0) - 1.00001),
            wp.vec3(points.max(axis=0) + 1.00001),
            scratch,
            neighbors,
            counts,
            status,
        ],
        device=device,
        block_dim=32,
    )
    adj, sizes = neighbors.numpy(), counts.numpy()
    return [set(map(int, row[:size])) for row, size in zip(adj, sizes)], status.numpy()


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("seed", [1, 42, 123])
def test_random_against_all_halfspaces(weighted, seed):
    rng = np.random.default_rng(seed)
    points = rng.uniform(-1, 1, (48, 3)).astype(np.float32)
    weights = rng.normal(0, 0.05, 48).astype(np.float32) if weighted else None
    actual, status = compute(points, weights)
    expected, vertices = reference_cells(points, weights)
    assert actual == expected
    assert np.all(np.isin(status, [0, 3]))
    assert np.array_equal(status == 3, [len(v) == 0 for v in vertices])


@pytest.mark.parametrize("weights", [None, [0, 100], [100, 0]])
def test_two_sites(weights):
    points = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    actual, status = compute(points, weights)
    expected, _ = reference_cells(points, weights)
    assert actual == expected
    assert np.all(np.isin(status, [0, 3]))


def test_duplicate_sites_have_one_owner():
    rows, status = compute([[0, 0, 0], [0, 0, 0], [1, 0, 0]])
    assert status[1] == 3
    # The third cell may initially retain either equal duplicate plane. Global
    # duplicate preprocessing will remove dominated representatives later.
    assert rows[0] == {2}


def test_cpu_execution():
    rows, status = compute([[0, 0, 0], [1, 0, 0]], device="cpu")
    assert rows == [{1}, {0}]
    assert list(status) == [0, 0]


def test_overflow_is_reported_without_partial_adjacency():
    p = np.random.default_rng(42).uniform(-1, 1, (24, 3)).astype(np.float32)
    rows, status = compute(p, capacity=8)
    assert np.any(np.isin(status, [1, 2]))
    assert all(not row for row, code in zip(rows, status) if code != 0)


def test_translated_sites():
    p = np.random.default_rng(5).uniform(-1, 1, (24, 3)).astype(np.float32)
    p += np.array([1000, -2000, 3000], dtype=np.float32)
    rows, status = compute(p)
    assert rows == reference_cells(p)[0]
    assert np.all(status == 0)
