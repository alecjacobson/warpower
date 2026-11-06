import json
from pathlib import Path

import numpy as np
import pytest
import warp as wp

import warpower
from tests.test_tetrahedra import check_mesh, reference, tet_set
from warpower import _bvh
from warpower._cell import Cell64, Scratch64, allocate_scratch, clip, init_cell


@wp.kernel
def ordered_clips(
    points: wp.array[wp.vec3],
    bounds: wp.array[wp.vec3],
    scratch: Scratch64,
    result: wp.array[int],
):
    cell = init_cell(scratch, 0, points[0], bounds[0], bounds[1], Cell64())
    for j in range(1, points.shape[0]):
        cell = clip(scratch, cell, 0, j, wp.vec3d(points[j]) - wp.vec3d(points[0]), 0.0, 0.0, False)
        if cell.status != 0:
            break
    result[0] = cell.status
    result[1] = cell.nv
    result[2] = cell.np


def test_intermediate_cell_complexity_recovers_and_matches_cgal_star():
    fixture = json.loads(
        (Path(__file__).parent / "data/intermediate-cell-capacity.json").read_text()
    )
    points = wp.array(np.array(fixture["points"], np.float32), dtype=wp.vec3, device="cuda:0")
    bounds = wp.array(np.array(fixture["bounds"], np.float32), dtype=wp.vec3, device="cuda:0")
    result = wp.zeros(3, dtype=int, device="cuda:0")
    for capacity in (512, 1024):
        scratch = allocate_scratch(32, capacity, points.device, robust=True)
        wp.launch(ordered_clips, 1, inputs=[points, bounds, scratch, result])
        status, nv, _ = result.numpy()
        vertices = scratch.vertices.numpy()[: 32 * nv : 32]
        # The overflow comes from real intermediate complexity, not a repeated
        # broken edge cycle. Larger capacity finishes with a much smaller cell.
        assert len(np.unique(vertices)) == nv
        if capacity == 512:
            assert status == 2 and nv == 512
        else:
            assert status == 0 and nv == fixture["final_vertices"]
            planes = np.column_stack([(vertices >> shift) & 1023 for shift in (0, 10, 20)])
            neighbors = scratch.neighbors.numpy()[::32][planes]
            finite = neighbors[np.all(neighbors >= 0, axis=1)]
            actual = {tuple(sorted((0, *map(int, row)))) for row in finite}
            assert actual == {tuple(t) for t in fixture["expected_seed_tetrahedra"]}


@wp.kernel
def defer_small_cells(
    points: wp.array[wp.vec3],
    weights: wp.array[float],
    tree: _bvh.Tree,
    jobs: wp.array[int],
    first: wp.array[int],
    job_count: wp.array[int],
    bounds: wp.array[wp.vec3],
    scratch: Scratch64,
    status: wp.array[int],
    candidates: wp.array2d[int],
    candidate_count: wp.array[int],
):
    job = first[0] + wp.tid()
    if job < job_count[0]:
        status[jobs[job]] = 2


@pytest.mark.parametrize("weighted", [False, True])
def test_captured_larger_recovery_stage_and_odd_chunks(weighted, monkeypatch):
    rng = np.random.default_rng(91)
    points = rng.normal(size=(43, 3)).astype(np.float32)
    weights = rng.normal(0, 0.05, len(points)).astype(np.float32) if weighted else None
    workspace = warpower.DiagramWorkspace(
        points,
        weights,
        adjacency=False,
        return_tetrahedra=True,
        tetrahedra_cell_capacity=1024,
        chunk_size=17,
    )
    original = workspace._gt.make_cells
    monkeypatch.setattr(
        workspace._gt, "make_cells", lambda cap: defer_small_cells if cap < 1024 else original(cap)
    )
    wp.load_module(module=__name__, device="cuda:0")
    with wp.ScopedCapture() as capture:
        workspace.run()
    for _ in range(2):
        wp.capture_launch(capture.graph)
        assert workspace.error.numpy()[0] == 0
        tets = workspace.tetrahedra.numpy()[: workspace.tetrahedra_count.numpy()[0]]
        check_mesh(points, tets)
        assert tet_set(tets) == reference(points, weights)


@pytest.mark.parametrize("capacity", [0, 256, 513, 2048])
def test_cell_capacity_is_bounded_by_packed_plane_indices(capacity):
    points = np.random.default_rng(3).normal(size=(10, 3)).astype(np.float32)
    with pytest.raises(ValueError, match="tetrahedra_cell_capacity"):
        warpower.DiagramWorkspace(points, return_tetrahedra=True, tetrahedra_cell_capacity=capacity)
