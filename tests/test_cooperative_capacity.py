"""Exercise independent queue limits and parallel large-cell clipping."""

import hashlib
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import warp as wp

import warpower
from benchmarks.replay.retry_diagnostics import observe
from tests.oracle import reference_cells
from tests.test_retry_escalation import RetryCell, shell


def test_weighted_capacity_retry_avoids_a_certified_false_facet():
    """Exclude a certified false facet without triggering consistency repair."""
    from benchmarks.datasets import generate

    fixture = json.loads((Path(__file__).parent / "data/weighted-capacity-facet.json").read_text())
    points, weights = generate(**fixture["generator"])
    for data, name in [(points, "points"), (weights, "weights")]:
        np.testing.assert_equal(
            hashlib.sha256(data.tobytes()).hexdigest(), fixture[name + "_sha256"]
        )
    owner, candidate = fixture["owner"], fixture["candidate"]
    origin = [Fraction(float(value)) for value in points[owner]]
    origin_weight = Fraction(float(weights[owner]))
    normal = [Fraction(0)] * 3
    bound = Fraction(0)
    for term in fixture["certificate"]:
        multiplier = Fraction(term["coefficient"])
        site = term["site"]
        if term["kind"] == "site":
            np.testing.assert_equal(multiplier >= 0, True)
        else:
            np.testing.assert_equal(site, candidate)
        delta = [Fraction(float(points[site, axis])) - origin[axis] for axis in range(3)]
        for axis in range(3):
            normal[axis] += multiplier * 2 * delta[axis]
        bound += multiplier * (
            sum(value * value for value in delta) + origin_weight - Fraction(float(weights[site]))
        )
    np.testing.assert_equal(normal, [Fraction(0)] * 3)
    np.testing.assert_equal(bound < 0, True)
    result, stages = observe(points, weights)
    np.testing.assert_equal(result.diagnostics["consistency_retried_cells"], 0)
    np.testing.assert_equal(
        any(
            stage["precision"] == "fp32"
            and stage["threads_per_cell"] == 32
            and candidate in stage.get("active_site_ids", [])
            for stage in stages
        ),
        True,
    )
    adjacency, offsets, status = (
        result.adjacency.numpy(),
        result.offsets.numpy(),
        result.status.numpy(),
    )
    for left, right in [(owner, candidate), (candidate, owner)]:
        np.testing.assert_equal(status[left], 0)
        np.testing.assert_equal(right in adjacency[offsets[left] : offsets[left + 1]], False)


@wp.kernel
def queue_requirement_kernel(status: wp.array[int], states: wp.array[wp.vec3i]):
    status[0] = 4
    states[0] = wp.vec3i(0, 64, 192)


def test_mixed_queue_batch_finishes_remaining_clipping_retries(monkeypatch):
    """Keep a mixed recovery batch together until its large clipping cell succeeds."""
    from warpower import _api

    points, owner = shell()
    original_launch = wp.launch

    def launch(kernel, dim, *args, **kwargs):
        result = original_launch(kernel, dim, *args, **kwargs)
        if kernel == _api.collect_precision:
            inputs = kwargs["inputs"]
            original_launch(
                queue_requirement_kernel, 1, inputs=[inputs[0], inputs[3]], device="cuda:0"
            )
        return result

    monkeypatch.setattr(wp, "launch", launch)
    result, stages = observe(points)
    first = next(
        stage
        for stage in stages
        if stage["precision"] == "fp32" and stage["threads_per_cell"] == 32
    )
    np.testing.assert_equal(first["allocated_queue_capacity"], 256)
    np.testing.assert_equal({0, owner}.issubset(first["active_site_ids"]), True)
    np.testing.assert_equal(
        any(
            stage["precision"] == "fp32"
            and stage["allocated_cell_capacity"] == 512
            and owner in stage.get("active_site_ids", [])
            for stage in stages
        ),
        True,
    )
    np.testing.assert_array_equal(result.status.numpy(), np.zeros(len(points), np.int32))
    offsets = result.offsets.numpy()
    np.testing.assert_array_equal(
        np.sort(result.adjacency.numpy()[offsets[owner] : offsets[owner + 1]]),
        np.delete(np.arange(len(points)), owner),
    )


@pytest.mark.parametrize("route", ["eager", "graph"])
@pytest.mark.parametrize("robust", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_large_cooperative_cell_has_every_independently_witnessed_facet(route, robust, weighted):
    """Recover a high-degree cell with all 32 lanes at either precision."""
    points, owner = shell()
    weights = None
    if weighted:
        weights = np.random.default_rng(61).uniform(-1e-5, 1e-5, len(points)).astype(np.float32)
        others = np.delete(np.arange(len(points)), owner)
        relative = points[others].astype(float) - points[owner].astype(float)
        squared = np.sum(relative * relative, axis=1)
        heights = squared + float(weights[owner]) - weights[others].astype(float)
        witnesses = relative * (heights / (2 * squared))[:, None]
        residual = heights[None, :] - 2 * witnesses @ relative.T
        np.fill_diagonal(residual, np.inf)
        np.testing.assert_array_less(0.001, residual)
    cell = RetryCell(points, owner, route, 2, (int(robust), 512, 128), weights=weights)
    code, state = cell.run(512, robust, cooperative=True, queue=128)
    np.testing.assert_equal(code, 0)
    np.testing.assert_equal(state[0] & 1, int(robust))
    np.testing.assert_equal(state[1:], [512, 128])
    count = int(cell.counts.numpy()[cell.row])
    np.testing.assert_array_equal(
        np.sort(cell.neighbors.numpy()[cell.row, :count]),
        np.delete(np.arange(len(points)), owner),
    )


@pytest.mark.parametrize("route", ["eager", "graph"])
@pytest.mark.parametrize("robust", [False, True])
def test_queue_exhaustion_preserves_clipping_capacity_and_precision(route, robust):
    """Grow an exhausted queue while retaining clipping capacity and precision."""
    points = np.random.default_rng(51).normal(size=(24, 3)).astype(np.float32)
    cell = RetryCell(points, 7, route, 4, (int(robust), 64, 1))
    code, state = cell.run(128, robust, cooperative=True, queue=1)
    np.testing.assert_equal(code, 4)
    np.testing.assert_equal(state, [3 if robust else 0, 64, 2])
    before = cell.snapshot()
    cell.run(512, robust, cooperative=True, queue=1)
    for expected, actual in zip(before, cell.snapshot()):
        np.testing.assert_array_equal(actual, expected)
    for _ in range(8):
        previous_queue = state[2]
        code, state = cell.run(128, robust, cooperative=True, queue=128)
        np.testing.assert_equal(state[:2], [3 if robust else 0, 64])
        if code == 0:
            break
        np.testing.assert_equal(code, 4)
        np.testing.assert_equal(state[2], previous_queue * 2)
    np.testing.assert_equal(code, 0)
    expected, _ = reference_cells(points)
    count = int(cell.counts.numpy()[cell.row])
    np.testing.assert_array_equal(
        np.sort(cell.neighbors.numpy()[cell.row, :count]), sorted(expected[7])
    )


def test_default_capacity_path_reports_active_cooperative_retries():
    """Prove that ordinary construction uses the large FP32 cooperative path."""
    points, owner = shell()
    result, stages = observe(points)
    matching = [
        stage
        for stage in stages
        if stage["precision"] == "fp32"
        and stage["threads_per_cell"] == 32
        and stage["allocated_cell_capacity"] == 512
        and owner in stage.get("active_site_ids", [])
    ]
    np.testing.assert_equal(len(matching), 1)
    np.testing.assert_equal(matching[0]["outcomes_after"].get("success", 0) > 0, True)
    offsets = result.offsets.numpy()
    np.testing.assert_array_equal(
        np.sort(result.adjacency.numpy()[offsets[owner] : offsets[owner + 1]]),
        np.delete(np.arange(len(points)), owner),
    )


def test_captured_default_uses_large_fp32_recovery():
    """Resolve the witnessed high-degree cell in a captured default constructor."""
    points, owner = shell()
    device_points = wp.array(points, dtype=wp.vec3, device="cuda:0")
    workspace = warpower.DiagramWorkspace(device_points).run()
    np.testing.assert_equal(workspace.error.numpy()[0], 0)
    state = workspace._retry_states.numpy()[owner]
    np.testing.assert_equal(state[0] & 1, 0)
    np.testing.assert_equal(state[1], 512)
    offsets = workspace.offsets.numpy()
    np.testing.assert_array_equal(
        np.sort(workspace.adjacency.numpy()[offsets[owner] : offsets[owner + 1]]),
        np.delete(np.arange(len(points)), owner),
    )


@pytest.mark.parametrize("queue_requirement", [192, 384, 1024])
def test_captured_queue_dispatch_retains_cell_limit(queue_requirement):
    """Exercise conditional larger-queue recovery and its explicit terminal limit."""
    points = np.random.default_rng(51).normal(size=(24, 3)).astype(np.float32)
    owner = 7
    device_points = wp.array(points, dtype=wp.vec3, device="cuda:0")
    workspace = warpower.DiagramWorkspace(device_points).run()
    states = np.tile(np.array([0, 64, 96], np.int32), (len(points), 1))
    states[owner, 2] = queue_requirement
    codes = np.zeros(len(points), np.int32)
    codes[owner] = 4
    wp.copy(workspace._retry_states, wp.array(states, dtype=wp.vec3i, device="cuda:0"))
    wp.copy(workspace.status, wp.array(codes, device="cuda:0"))
    wp.copy(workspace._jobs, wp.array([owner], dtype=int, device="cuda:0"), count=1)
    workspace._job_count.fill_(1)
    workspace._source.fill_(-1)
    workspace._base.fill_(-1)
    workspace._counts.zero_()
    with wp.ScopedCapture(device="cuda:0") as capture:
        workspace._retries()
    wp.capture_launch(capture.graph)
    state = workspace._retry_states.numpy()[owner]
    np.testing.assert_equal(state[:2], [0, 64])
    if queue_requirement > 512:
        np.testing.assert_equal(workspace.status.numpy()[owner], 4)
        np.testing.assert_equal(state[2], queue_requirement)
        np.testing.assert_equal(workspace._counts.numpy()[owner], 0)
    else:
        np.testing.assert_equal(workspace.status.numpy()[owner], 0)
        np.testing.assert_equal(state[2], 256 if queue_requirement <= 256 else 512)
        expected, _ = reference_cells(points)
        count = int(workspace._counts.numpy()[owner])
        np.testing.assert_array_equal(
            np.sort(workspace._retry.numpy()[owner, :count]), sorted(expected[owner])
        )


def test_fp64_cooperative_retry_rejects_undersized_queue():
    """Skip a cooperative FP64 attempt that cannot honor the queue requirement."""
    points = np.random.default_rng(51).normal(size=(24, 3)).astype(np.float32)
    cell = RetryCell(points, 7, "eager", 6, (1, 64, 2))
    before = cell.snapshot()
    cell.run(128, True, cooperative=True, queue=1)
    for expected, actual in zip(before, cell.snapshot()):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("route,robust", [("eager", False), ("graph", True)])
def test_queue_retry_grows_clipping_only_when_needed(route, robust):
    """Reuse allocated slots when queue recovery encounters actual clipping exhaustion."""
    points, owner = shell(80)
    cell = RetryCell(points, owner, route, 4, (int(robust), 64, 192))
    code, state = cell.run(256, robust, cooperative=True, queue=256)
    np.testing.assert_equal(code, 0)
    np.testing.assert_equal(state[0] & 1, int(robust))
    np.testing.assert_equal(state[1:], [256, 256])
    count = int(cell.counts.numpy()[cell.row])
    np.testing.assert_array_equal(
        np.sort(cell.neighbors.numpy()[cell.row, :count]),
        np.delete(np.arange(len(points)), owner),
    )


@pytest.mark.parametrize("route", ["eager", "graph"])
def test_serial_queue_retry_keeps_physical_scratch_rows_disjoint(route):
    """Preserve independent scratch rows when queue retries retain different limits."""
    from warpower import _bvh
    from warpower._cell import allocate_scratch
    from warpower._construct import make_construct_kernel
    from warpower._graph_construct import make_retry_kernel

    shell_points, owner = shell(60)
    points = np.vstack((shell_points, np.eye(3, dtype=np.float32) * 3))
    owners = np.arange(len(points), dtype=np.int32)
    owners[[0, owner]] = owners[[owner, 0]]
    device_points = wp.array(points, dtype=wp.vec3, device="cuda:0")
    weights = wp.zeros(len(points), dtype=float, device="cuda:0")
    tree = _bvh.build(device_points, weights, 4)
    jobs = wp.array(owners, device="cuda:0")
    empty = wp.empty(0, dtype=int, device="cuda:0")
    scratch = allocate_scratch(len(points), 128, "cuda:0", robust=True)
    scratch.planes.fill_(wp.vec4d(float("nan")))
    neighbors = wp.full((len(points), 512), -1, dtype=int, device="cuda:0")
    counts = wp.zeros(len(points), dtype=int, device="cuda:0")
    codes = np.full(len(points), 4, np.int32)
    codes[0 if route == "eager" else owner] = 2
    status = wp.array(codes, device="cuda:0")
    states = np.tile(np.array([1, 64, 128], np.int32), (len(points), 1))
    states[owner, 1] = 128
    device_states = wp.array(states, dtype=wp.vec3i, device="cuda:0")
    if route == "eager":
        retry_kernel = make_construct_kernel(False, 256, True, 32)
        arguments = [
            device_points,
            weights,
            tree.tree,
            jobs,
            0,
            True,
            scratch,
            neighbors,
            counts,
            status,
            neighbors,
            True,
            empty,
            empty,
            False,
            device_states,
        ]
        wp.launch(retry_kernel, len(points) * 32, inputs=arguments, block_dim=32, device="cuda:0")
    else:
        retry_kernel = make_retry_kernel(False, 256, True, 32)
        cursor = wp.zeros(1, dtype=int, device="cuda:0")
        job_count = wp.array([len(points)], dtype=int, device="cuda:0")
        sources = wp.full(len(points), -1, dtype=int, device="cuda:0")
        arguments = [
            device_points,
            weights,
            tree.tree,
            jobs,
            cursor,
            job_count,
            sources,
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
            device_states,
        ]
        with wp.ScopedCapture(device="cuda:0") as capture:
            wp.launch(
                retry_kernel, len(points) * 32, inputs=arguments, block_dim=32, device="cuda:0"
            )
        wp.capture_launch(capture.graph)
    # Each physical row must be initialized even when its logical limit is
    # smaller. Changing the physical pitch per job aliases adjacent groups.
    slots = np.arange(len(points))
    physical_starts = (slots // 32) * (32 * 128) + slots % 32
    np.testing.assert_array_equal(
        np.isfinite(scratch.planes.numpy()[physical_starts]),
        np.ones((len(points), 4), bool),
    )
    np.testing.assert_array_equal(status.numpy(), np.zeros(len(points), np.int32))
    expected, _ = reference_cells(points)
    adjacency, sizes = neighbors.numpy(), counts.numpy()
    for row, point_owner in enumerate(owners if route == "eager" else np.arange(len(points))):
        np.testing.assert_array_equal(
            np.sort(adjacency[row, : sizes[row]]), sorted(expected[point_owner])
        )


def test_tiny_initial_queue_still_reaches_largest_clipping_capacity():
    """Finish a high-degree cell after queue recovery exposes clipping overflow."""
    from warpower._api import _build

    points, owner = shell()
    result = _build(points, _queue_capacity=1)
    np.testing.assert_array_equal(result.status.numpy(), np.zeros(len(points), np.int32))
    offsets = result.offsets.numpy()
    np.testing.assert_array_equal(
        np.sort(result.adjacency.numpy()[offsets[owner] : offsets[owner + 1]]),
        np.delete(np.arange(len(points)), owner),
    )
