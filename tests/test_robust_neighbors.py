"""Original-coordinate regressions and forced cooperative recovery checks."""

import json
from pathlib import Path

import numpy as np
import pytest
import warp as wp

import warpower
from tests.oracle import csr_sets, reference_cells
from tests.test_filtered_predicates import check
from warpower import _bvh
from warpower._cooperative_cell import allocate_scratch
from warpower._precision_construct import make_precision_kernel
from warpower._query_predicates import collect_precision_retry

FIXTURES = json.loads((Path(__file__).parent / "data/near_cospherical_neighbors.json").read_text())


@pytest.mark.parametrize("route", ["default", "chunked", "capacity", "hints", "graph"])
@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda f: f"{f['scene']}-{f['original_ids'][0]}")
def test_reduced_near_cospherical_neighbor_graph(fixture, route):
    points = wp.array(np.asarray(fixture["points"], np.float32), dtype=wp.vec3, device="cuda:0")
    if route == "capacity":
        from warpower._api import _build

        result = _build(points, _capacity=8)
        assert result.diagnostics["retried_cells"] > 0
    elif route == "graph":
        result = warpower.DiagramWorkspace(points).run()
        assert result.error.numpy()[0] == 0
    else:
        options = {}
        if route == "chunked":
            options["chunk_size"] = 1
        if route == "hints":
            n = len(points)
            options["initial_guesses"] = np.tile(np.arange(n, dtype=np.int32), n)
            options["initial_guesses_offsets"] = np.arange(n + 1, dtype=np.int32) * n
        result = warpower.voronoi_diagram(points, **options)
    offsets = result.offsets.numpy()
    # A captured workspace exposes capacity storage; only its prefix is valid.
    assert csr_sets(result.adjacency[: int(offsets[-1])].numpy(), offsets) == [
        set(row) for row in fixture["exact_neighbors"]
    ]
    assert np.all(result.status.numpy() == 0)


def test_original_coordinate_predicates_on_reduced_failures():
    check([f["points"] for f in FIXTURES])


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("remapped", [False, True])
def test_cooperative_recovery_rows_match_independent_oracle(weighted, remapped):
    rng = np.random.default_rng(97001)
    points = rng.normal(size=(48, 3)).astype(np.float32)
    weights = (
        rng.normal(0, 0.08, len(points)).astype(np.float32)
        if weighted
        else np.zeros(len(points), np.float32)
    )
    expected, _ = reference_cells(points, weights if weighted else None)
    owners = np.arange(len(points), dtype=np.int32)
    if remapped:
        rng.shuffle(owners)
    row_map = np.empty(len(points), np.int32)
    row_map[owners] = np.arange(len(points), dtype=np.int32)
    p = wp.array(points, dtype=wp.vec3, device="cuda:0")
    w = wp.array(weights, device="cuda:0")
    tree = _bvh.build(p, w, 4)
    jobs = wp.array(owners, device="cuda:0")
    count = wp.array([len(points)], dtype=int, device="cuda:0")
    mapping = wp.array(row_map, device="cuda:0")
    neighbors = wp.full((len(points), 128), -1, dtype=int, device="cuda:0")
    counts = wp.zeros(len(points), dtype=int, device="cuda:0")
    status = wp.full(len(points), 6, dtype=int, device="cuda:0")
    empty = wp.empty(0, dtype=int, device="cuda:0")
    scratch = allocate_scratch(len(points), 128, "cuda:0", robust=True)
    kernel = make_precision_kernel(weighted, 128, True, 32)
    wp.launch(
        kernel,
        len(points) * 32,
        inputs=[
            p,
            w,
            tree.tree,
            jobs,
            0,
            count,
            mapping,
            remapped,
            True,
            scratch,
            neighbors,
            counts,
            status,
            neighbors,
            False,
            empty,
            empty,
            False,
            wp.full(len(p), wp.vec2i(1, 64), dtype=wp.vec2i, device="cuda:0"),
        ],
        block_dim=32,
        device="cuda:0",
    )
    ns, sizes, codes = neighbors.numpy(), counts.numpy(), status.numpy()
    assert np.all(np.isin(codes, [0, 3]))
    for row, owner in enumerate(owners):
        output = row if remapped else int(owner)
        assert set(ns[output, : sizes[output]]) == expected[owner]


def test_bounded_precision_job_queue_retains_overflow_for_fallback():
    n, capacity = 600, 512
    status = wp.full(n, 6, dtype=int, device="cuda:0")
    original = np.arange(n, dtype=np.int32)[::-1].copy()
    owners = wp.array(original, device="cuda:0")
    jobs = wp.full(capacity, -1, dtype=int, device="cuda:0")
    count = wp.zeros(1, dtype=int, device="cuda:0")
    wp.launch(
        collect_precision_retry,
        n,
        inputs=[
            status,
            owners,
            jobs,
            count,
            wp.full(n, wp.vec2i(1, 64), dtype=wp.vec2i, device="cuda:0"),
        ],
        device="cuda:0",
    )
    queued = jobs.numpy()
    assert count.numpy()[0] == n
    assert len(np.unique(queued)) == capacity
    assert np.all((queued >= 0) & (queued < n))
    assert np.all(status.numpy() == 6)


def test_cooperative_recovery_nonowning_duplicate():
    p = wp.array(np.zeros((2, 3), np.float32), dtype=wp.vec3, device="cuda:0")
    w = wp.zeros(2, dtype=float, device="cuda:0")
    tree = _bvh.build(p, w, 4)
    jobs = wp.array([1], dtype=int, device="cuda:0")
    job_count = wp.array([1], dtype=int, device="cuda:0")
    mapping = wp.array([0, 1], dtype=int, device="cuda:0")
    neighbors = wp.full((2, 128), -1, dtype=int, device="cuda:0")
    counts = wp.full(2, -99, dtype=int, device="cuda:0")
    status = wp.full(2, 6, dtype=int, device="cuda:0")
    empty = wp.empty(0, dtype=int, device="cuda:0")
    scratch = allocate_scratch(1, 128, "cuda:0", robust=True)
    wp.launch(
        make_precision_kernel(False, 128, True, 32),
        32,
        inputs=[
            p,
            w,
            tree.tree,
            jobs,
            0,
            job_count,
            mapping,
            False,
            True,
            scratch,
            neighbors,
            counts,
            status,
            neighbors,
            False,
            empty,
            empty,
            False,
            wp.full(len(p), wp.vec2i(1, 64), dtype=wp.vec2i, device="cuda:0"),
        ],
        block_dim=32,
        device="cuda:0",
    )
    assert status.numpy().tolist() == [6, 3]
    assert counts.numpy().tolist() == [-99, 0]
