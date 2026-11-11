"""Direct lane/error regression against the original exact/symbolic face search."""

import subprocess
from pathlib import Path

import numpy as np
import pytest
import warp as wp

from benchmarks.replay import split_search_probe as split
from benchmarks.replay.backends import WarpBackend
from benchmarks.replay.fixture import Fixture, Writer
from tests import test_analytic_face_search as oracle


class SplitLauncher:
    """Use production kernels and retain queue diagnostics for every replay."""

    guard = -190297

    def __init__(self):
        self.observations = []

    def prepare(self, candidates, device):
        self.device = device
        self.capacity = len(candidates)
        # Guard words detect a one-past-end queue write at full occupancy.
        self.storage = wp.full(self.capacity + 2, self.guard, dtype=int, device=device)
        self.pending = self.storage[1 : self.capacity + 1]
        self.count = wp.zeros(1, dtype=int, device=device)
        self.statistics = wp.zeros(1, dtype=int, device=device)
        for block in [32, 256]:
            wp.load_module(module=split.__name__, device=device, block_dim=block)

    def launch(self, inputs, block):
        points, owners, tets, faces, count, candidates, workers, error, supported, statistics = (
            inputs
        )
        self.workers = workers
        self.count.zero_()
        self.statistics.zero_()
        wp.launch(
            split.strict_search,
            len(candidates),
            inputs=[
                points,
                owners,
                tets,
                faces,
                count,
                candidates,
                workers,
                supported,
                statistics,
                self.pending,
                self.count,
            ],
            device=self.device,
            block_dim=block,
        )
        wp.capture_if(
            self.count,
            lambda: wp.launch(
                split.exact_search,
                4096,
                inputs=[
                    points,
                    owners,
                    tets,
                    faces,
                    candidates,
                    workers,
                    error,
                    self.pending,
                    self.count,
                    4096,
                    self.statistics,
                ],
                device=self.device,
                block_dim=32,
            ),
        )

    def observe(self, phase, active):
        queued = int(self.count.numpy()[0])
        assert 0 <= queued <= min(self.capacity, active * self.workers)
        storage = self.storage.numpy()
        assert storage[0] == storage[-1] == self.guard
        pending = storage[1 : queued + 1]
        assert len(np.unique(pending)) == queued
        assert np.all((pending >= 0) & (pending < active * self.workers))
        # The exact kernel's strided loop must see the complete queue, not only
        # the first 4,096 queued lanes. Winner equality checks its actual work.
        assert int(self.statistics.numpy()[0]) == queued
        self.observations.append(dict(phase=phase, active=active, queued=queued))
        return queued


def _historical_fixture_cwd(monkeypatch):
    """Sparse worktrees can read the tracked torus fixture in the primary checkout."""
    relative = Path(
        "results/validation/vtm/large-fragment-reuse/large-reuse-final/torus4096-after.npz"
    )
    if relative.is_file():
        return
    common = Path(
        subprocess.check_output(["git", "rev-parse", "--git-common-dir"], text=True).strip()
    ).resolve()
    primary = common.parent
    if not (primary / relative).is_file():
        pytest.fail(f"Required historical torus fixture is missing: {relative}")
    monkeypatch.chdir(primary)


@pytest.mark.parametrize(
    "case", ["cube", "octahedron", "thin", "random33", "random257", "torus4096"]
)
@pytest.mark.parametrize("exponent", [-60, 0, 60])
@pytest.mark.parametrize("block", [32, 256])
def test_split_matches_every_original_lane_and_error_across_changed_inputs(
    case, exponent, block, monkeypatch
):
    if case == "torus4096":
        _historical_fixture_cwd(monkeypatch)
    launcher = SplitLauncher()
    # Reuse the independent rational winner oracle and the entire existing
    # captured mutation/ownership/count-change corpus. It compares every lane,
    # errors, live cells, final winners, and inactive-tail initialization.
    oracle.test_changed_inputs_match_exhaustive_and_rational_winners(
        case, exponent, block, search_override=launcher
    )
    assert len(launcher.observations) == 6
    assert launcher.observations[4]["active"] == 0
    assert launcher.observations[4]["queued"] == 0  # Empty active set only.


def _fully_uncertain_points():
    points = np.zeros((96, 3), np.float32)
    points[:4] = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    # Every lane first encounters a nondefining coplanar point, forcing a queue
    # entry under any strict filter. It later has one positive exterior winner.
    points[4:64, 0] = np.arange(4, 64, dtype=np.float32) / 8
    points[4:64, 1] = np.float32(1 / 8)
    points[64:, 0] = np.arange(1, 33, dtype=np.float32) / 8
    points[64:, 1] = np.float32(1 / 4)
    points[64:, 2] = -1
    assert len(np.unique(points, axis=0)) == len(points)
    return points


@pytest.mark.parametrize("block", [32, 256])
def test_full_queue_above_exact_worker_count_replays_every_lane_and_invalid_phase(block):
    device, workers, face_count = "cuda:0", 32, 256
    points = _fully_uncertain_points()
    invalid = points.copy()
    invalid[64, 0] = np.nan
    device_points = wp.array(points, dtype=wp.vec3, device=device)
    owners = wp.array(np.arange(len(points), dtype=np.int32), dtype=int, device=device)
    tets = wp.array(np.array([[0, 1, 2, 3]], np.int32), dtype=int, device=device)
    faces = wp.full(face_count, 3, dtype=int, device=device)
    count = wp.full(1, face_count, dtype=int, device=device)
    supported = wp.zeros(face_count, dtype=int, device=device)
    errors = wp.zeros(2, dtype=int, device=device)
    statistics = wp.zeros(2, dtype=int, device=device)
    candidates = [wp.empty(workers * face_count, dtype=int, device=device) for _ in range(2)]
    launcher = SplitLauncher()
    launcher.prepare(candidates[1], device)
    wp.load_module(module=oracle.__name__, device=device, block_dim=block)
    with wp.ScopedCapture(device=device) as captured:
        errors.zero_()
        statistics.zero_()
        for j in range(2):
            inputs = [
                device_points,
                owners,
                tets,
                faces,
                count,
                candidates[j],
                workers,
                errors[j : j + 1],
                supported,
                statistics,
            ]
            if j == 0:
                wp.launch(
                    oracle.reference_search,
                    len(candidates[j]),
                    inputs=inputs,
                    device=device,
                    block_dim=block,
                )
            else:
                launcher.launch(inputs, block)
    for phase, source in enumerate([points, invalid, points]):
        wp.copy(device_points, wp.array(source, dtype=wp.vec3, device=device))
        wp.capture_launch(captured.graph)
        expected, actual = [value.numpy() for value in candidates]
        np.testing.assert_array_equal(actual, expected)
        error = errors.numpy()
        np.testing.assert_array_equal(error, [2, 2] if phase == 1 else [0, 0])
        queued = launcher.observe(phase, face_count)
        assert queued == launcher.capacity == 8192
        assert queued > 4096
        if phase != 1:
            # If the exact phase silently stops after its first 4,096 slots,
            # half these nonnegative winners incorrectly remain at -1.
            np.testing.assert_array_equal(
                actual.reshape(face_count, workers), np.tile(np.arange(64, 96), (face_count, 1))
            )
        else:
            assert np.all(actual.reshape(face_count, workers)[:, 0] == -1)


def test_split_backend_does_not_mutate_prepared_previous_input(tmp_path):
    points = np.random.default_rng(2297).normal(size=(32, 3)).astype(np.float32)
    moved = points.copy()
    moved[7] += np.array([0.125, -0.25, 0.0625], np.float32)
    writer = Writer(tmp_path / "immutable", name="prepared-input-immutability", provenance={})
    ids = np.arange(len(points), dtype=np.int64)
    writer.append(points, ids, reset=True)
    writer.append(moved, ids)
    writer.finalize()
    fixture = Fixture(writer.root)
    backend = WarpBackend("vtm-split-search")
    try:
        previous = backend.prepare(fixture.frame(0))
        current = backend.prepare(fixture.frame(1))
        original = previous["device_points"].numpy().copy()
        backend.update(previous)
        backend.synchronize()
        initial_mesh = backend.readback()
        assert initial_mesh.error is None
        backend.update(current)
        backend.synchronize()
        assert backend.readback().error is None
        np.testing.assert_array_equal(previous["device_points"].numpy(), original)
        np.testing.assert_array_equal(current["device_points"].numpy(), moved)
        backend.update(previous)
        backend.synchronize()
        restored = backend.readback()
        assert restored.error is None
        np.testing.assert_array_equal(restored.rows, initial_mesh.rows)
    finally:
        backend.close()
