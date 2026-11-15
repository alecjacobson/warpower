"""Actual retry kernels must retain precision and storage across failures."""

import numpy as np
import pytest
import warp as wp

from warpower import _bvh
from warpower._cell import allocate_scratch
from warpower._construct import make_construct_kernel
from warpower._cooperative_cell import allocate_scratch as cooperative_scratch
from warpower._graph_construct import make_retry_kernel
from warpower._precision_construct import make_precision_kernel


def shell(n=160):
    k = np.arange(n)
    z = 1 - 2 * (k + 0.5) / n
    phi = k * (np.pi * (3 - np.sqrt(5)))
    xy = np.sqrt(1 - z * z)
    q = np.column_stack((xy * np.cos(phi), xy * np.sin(phi), z))
    q *= (1 + 0.001 * np.sin(k * 1.731))[:, None]
    q = q.astype(np.float32)
    # Each bisector has a strict interior witness at q/2; this independently
    # establishes that every shell site is a neighbor of the central site.
    residual = np.sum(q.astype(float) ** 2, axis=1)[None, :] - q.astype(float) @ q.T
    np.fill_diagonal(residual, np.inf)
    assert residual.min() > 0.01
    points = np.vstack((np.zeros((1, 3), np.float32), q))
    points[[0, 7]] = points[[7, 0]]
    return points, 7


class RetryCell:
    def __init__(self, points, owner, route, code, state, width=512, weights=None):
        self.route, self.owner = route, owner
        self.weighted = weights is not None
        self.p = wp.array(points, dtype=wp.vec3, device="cuda:0")
        self.w = (
            wp.zeros(len(points), dtype=float, device="cuda:0")
            if weights is None
            else wp.array(weights, dtype=float, device="cuda:0")
        )
        self.tree = _bvh.build(self.p, self.w, 4)
        self.jobs = wp.array([owner], dtype=int, device="cuda:0")
        self.job_count = wp.array([1], dtype=int, device="cuda:0")
        self.cursor = wp.zeros(1, dtype=int, device="cuda:0")
        self.sources = wp.full(len(points), -1, dtype=int, device="cuda:0")
        mapping = np.full(len(points), -1, np.int32)
        mapping[owner] = 0
        self.mapping = wp.array(mapping, device="cuda:0")
        self.empty = wp.empty(0, dtype=int, device="cuda:0")
        self.row = 0 if route == "eager" else owner
        rows = 1 if route == "eager" else len(points)
        status = np.full(rows, 3, np.int32)
        status[self.row] = code
        self.status = wp.array(status, device="cuda:0")
        self.counts = wp.zeros(rows, dtype=int, device="cuda:0")
        self.neighbors = wp.full((rows, width), -1, dtype=int, device="cuda:0")
        if len(state) == 2:
            state = (*state, 128)
        self.states = wp.full(len(points), wp.vec3i(*state), dtype=wp.vec3i, device="cuda:0")

    def run(self, capacity, robust, cooperative=False, queue=None):
        queue = capacity if queue is None else queue
        if cooperative:
            scratch = cooperative_scratch(1, capacity, "cuda:0", robust=robust)
            kernel = make_precision_kernel(self.weighted, queue, robust, 32, capacity)
            args = [
                self.p,
                self.w,
                self.tree.tree,
                self.jobs,
                0,
                self.job_count,
                self.mapping,
                self.route == "eager",
                True,
                scratch,
                self.neighbors,
                self.counts,
                self.status,
                self.neighbors,
                False,
                self.empty,
                self.empty,
                False,
                self.states,
            ]
        else:
            scratch = allocate_scratch(1, capacity, "cuda:0", robust=robust)
            if self.route == "eager":
                kernel = make_construct_kernel(self.weighted, queue, robust=robust, lane_stride=32)
                args = [
                    self.p,
                    self.w,
                    self.tree.tree,
                    self.jobs,
                    0,
                    True,
                    scratch,
                    self.neighbors,
                    self.counts,
                    self.status,
                    self.neighbors,
                    True,
                    self.empty,
                    self.empty,
                    False,
                    self.states,
                ]
            else:
                kernel = make_retry_kernel(self.weighted, queue, robust, 32)
                args = [
                    self.p,
                    self.w,
                    self.tree.tree,
                    self.jobs,
                    self.cursor,
                    self.job_count,
                    self.sources,
                    True,
                    scratch,
                    self.neighbors,
                    self.counts,
                    self.status,
                    self.neighbors,
                    False,
                    self.empty,
                    self.empty,
                    False,
                    self.states,
                ]
        if self.route == "graph":
            with wp.ScopedCapture(device="cuda:0") as capture:
                wp.launch(kernel, 32, inputs=args, block_dim=32, device="cuda:0")
            wp.capture_launch(capture.graph)
        else:
            wp.launch(kernel, 32, inputs=args, block_dim=32, device="cuda:0")
        wp.synchronize()
        return int(self.status.numpy()[self.row]), self.states.numpy()[self.owner].tolist()

    def snapshot(self):
        return [x.numpy() for x in [self.status, self.counts, self.neighbors, self.states]]

    def assert_unchanged(self, before):
        for a, b in zip(before, self.snapshot()):
            assert np.array_equal(a, b)


@pytest.mark.parametrize("route", ["eager", "graph"])
@pytest.mark.parametrize("robust", [False, True])
def test_real_capacity_overflow_preserves_precision_and_storage(route, robust):
    points, owner = shell()
    cell = RetryCell(points, owner, route, 6 if robust else 1, (int(robust), 128))
    code, state = cell.run(128, robust, cooperative=robust)
    assert code in (1, 2, 4)
    assert state[0] & 1 == int(robust)
    assert state[1] == 256
    before = cell.snapshot()
    # Neither changing precision to accommodate size, nor restarting with a
    # smaller scratch buffer, is allowed to rebuild this pending cell.
    cell.run(512, not robust)
    cell.assert_unchanged(before)
    cell.run(128, robust)
    cell.assert_unchanged(before)
    code, state = cell.run(256, robust)
    assert code in (1, 2, 4)
    assert state[0] & 1 == int(robust)
    assert state[1] == 512
    code, state = cell.run(512, robust)
    assert code == 0
    assert state[0] & 1 == int(robust)
    assert state[1] == 512
    count = int(cell.counts.numpy()[cell.row])
    assert set(cell.neighbors.numpy()[cell.row, :count]) == set(range(len(points))) - {owner}


@pytest.mark.parametrize("route", ["eager", "graph"])
@pytest.mark.parametrize("code", [5, 6])
@pytest.mark.parametrize("capacity", [256, 512])
def test_numerical_failure_skips_fp32_and_keeps_existing_capacity(route, code, capacity):
    points = np.random.default_rng(51).normal(size=(24, 3)).astype(np.float32)
    cell = RetryCell(points, 7, route, code, (1, capacity))
    before = cell.snapshot()
    cell.run(512, False)
    cell.assert_unchanged(before)
    cell.run(128, True, cooperative=True)
    cell.assert_unchanged(before)
    cell.run(128, True)
    cell.assert_unchanged(before)
    result, state = cell.run(capacity, True)
    assert result == 0
    assert state[0] & 1 and state[1] == capacity


@pytest.mark.parametrize("route", ["eager", "graph"])
def test_output_row_overflow_does_not_double_scratch_capacity(route):
    points = np.random.default_rng(51).normal(size=(24, 3)).astype(np.float32)
    cell = RetryCell(points, 7, route, 6, (1, 64), width=1)
    code, state = cell.run(128, True, cooperative=True)
    assert code == 1
    assert state[:2] == [3, 128]
    # The cell fit in scratch; only the emitted row was too small. Widening
    # that row must preserve FP64 and permit the same 128-slot scratch.
    cell.neighbors = wp.full((cell.neighbors.shape[0], 512), -1, dtype=int, device="cuda:0")
    before = cell.snapshot()
    cell.run(512, False)
    cell.assert_unchanged(before)
    code, state = cell.run(128, True)
    assert code == 0 and state[:2] == [3, 128]


@pytest.mark.parametrize("route", ["eager", "graph"])
def test_reaching_storage_limit_does_not_promote_precision(route):
    points, owner = shell(260)
    cell = RetryCell(points, owner, route, 1, (0, 128))
    for capacity in (128, 256, 512):
        code, state = cell.run(capacity, False)
        assert code in (1, 2, 4)
        assert state[:2] == [0, 2 * capacity]
    before = cell.snapshot()
    cell.run(512, True)
    cell.assert_unchanged(before)


def test_captured_replay_resets_retry_history_for_new_coordinates():
    import warpower
    from tests.oracle import csr_sets, reference_cells
    from tests.test_robust_neighbors import FIXTURES

    points = wp.array(np.asarray(FIXTURES[0]["points"], np.float32), dtype=wp.vec3, device="cuda:0")
    workspace = warpower.DiagramWorkspace(points).run()
    assert workspace.error.numpy()[0] == 0
    assert np.any(workspace._retry_states.numpy()[:, 0] & 1)
    fresh = np.random.default_rng(391).normal(size=(len(points), 3)).astype(np.float32)
    wp.copy(points, wp.array(fresh, dtype=wp.vec3, device="cuda:0"))
    workspace.run()
    assert workspace.error.numpy()[0] == 0
    assert np.all(workspace._retry_states.numpy()[:, 0] == 0)
    offsets = workspace.offsets.numpy()
    expected, _ = reference_cells(fresh)
    assert csr_sets(workspace.adjacency[: int(offsets[-1])].numpy(), offsets) == expected
