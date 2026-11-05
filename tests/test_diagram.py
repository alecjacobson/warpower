import numpy as np
import pytest

import warpower
from tests.oracle import csr_sets, reference_cells


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("leaf_size", [1, 7, 17])
def test_complete_diagram(weighted, leaf_size):
    rng = np.random.default_rng(3)
    p = rng.uniform(-1, 1, (48, 3)).astype(np.float32)
    w = rng.normal(0, 0.1, len(p)).astype(np.float32) if weighted else None
    result = (
        warpower.power_diagram(p, w, leaf_size=leaf_size)
        if weighted
        else warpower.voronoi_diagram(p, leaf_size=leaf_size)
    )
    rows = csr_sets(result.adjacency.numpy(), result.offsets.numpy())
    assert rows == reference_cells(p, w)[0]
    assert np.all(np.isin(result.status.numpy(), [0, 3]))


def test_forced_chunking_and_queue_retry():
    p = np.random.default_rng(6).uniform(-1, 1, (128, 3)).astype(np.float32)
    a = warpower.voronoi_diagram(p)
    b = warpower.voronoi_diagram(p, chunk_size=19, leaf_size=2, _queue_capacity=1)
    assert b.diagnostics["retried_cells"] > 0
    assert np.all(b.status.numpy() == 0)
    assert csr_sets(a.adjacency.numpy(), a.offsets.numpy()) == csr_sets(
        b.adjacency.numpy(), b.offsets.numpy()
    )


def test_empty_and_single_site():
    result = warpower.voronoi_diagram(np.empty((0, 3), dtype=np.float32))
    assert result.offsets.numpy().tolist() == [0]
    result = warpower.voronoi_diagram(np.zeros((1, 3), dtype=np.float32))
    assert result.offsets.numpy().tolist() == [0, 0]
    assert result.status.numpy().tolist() == [0]


def test_strong_weight_eliminates_cell():
    p = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    result = warpower.power_diagram(p, np.array([0, 100], dtype=np.float32))
    assert result.offsets.numpy().tolist() == [0, 0, 0]
    assert result.status.numpy().tolist() == [3, 0]


@pytest.mark.parametrize("weighted", [False, True])
def test_separated_clusters_with_tight_pruning(weighted):
    rng = np.random.default_rng(17)
    centers = np.array([[-5, 0, 0], [4, 2, 0], [0, -4, 3]])
    p = (centers[np.arange(120) % 3] + rng.normal(0, 0.1, (120, 3))).astype(np.float32)
    w = rng.normal(0, 0.001, len(p)).astype(np.float32) if weighted else None
    result = (
        warpower.power_diagram(p, w, leaf_size=4)
        if weighted
        else warpower.voronoi_diagram(p, leaf_size=4)
    )
    assert np.all(np.isin(result.status.numpy(), [0, 3]))
    assert csr_sets(result.adjacency.numpy(), result.offsets.numpy()) == reference_cells(p, w)[0]


@pytest.mark.parametrize("weighted", [False, True])
def test_double_precision_retry_matches_oracle(weighted):
    rng = np.random.default_rng(22)
    p = rng.normal(size=(48, 3)).astype(np.float32)
    w = rng.normal(0, 0.05, len(p)).astype(np.float32) if weighted else None
    result = (
        warpower.power_diagram(p, w, _capacity=8)
        if weighted
        else warpower.voronoi_diagram(p, _capacity=8)
    )
    assert result.diagnostics["retried_cells"] > 0
    assert np.all(np.isin(result.status.numpy(), [0, 3]))
    assert csr_sets(result.adjacency.numpy(), result.offsets.numpy()) == reference_cells(p, w)[0]


@pytest.mark.parametrize("weighted", [False, True])
def test_ten_cluster_boundary_recovery(weighted):
    from benchmarks.datasets import generate

    p, w = generate(100000, "cluster10", weighted=weighted)
    result = warpower.power_diagram(p, w) if weighted else warpower.voronoi_diagram(p)
    assert result.diagnostics["retried_cells"] > 0
    assert np.all(np.isin(result.status.numpy(), [0, 3] if weighted else [0]))


@pytest.mark.parametrize("weighted", [False, True])
def test_clustered_one_sided_facet_recovery(weighted):
    from benchmarks.datasets import generate

    points, weights = generate(100000, "cluster5", 42, weighted)
    result = (
        warpower.power_diagram(points, weights) if weighted else warpower.voronoi_diagram(points)
    )
    # These hull cells previously disagreed on one facet despite successful
    # statuses. Check their complete neighborhoods against all input halfspaces.
    ids = [26333, 78334]
    expected, _ = reference_cells(points, weights, indices=ids)
    adjacency, offsets = result.adjacency.numpy(), result.offsets.numpy()
    actual = [set(adjacency[offsets[i] : offsets[i + 1]]) for i in ids]
    assert actual == expected
    assert result.diagnostics["asymmetric_cells_remaining"] == 0
