"""Experimental bounded rollback at fresh invalid material edge stars (#287).

Production never imports this module. A local proposal is only a candidate:
ordinary reconstruction, objective, selection and quality guards still decide.
"""

import argparse
import hashlib
import json
import runpy
import subprocess
import sys
import time
from pathlib import Path

import warp as wp

import warpower
from warpower import _mesher
from warpower._selective_backtracking import LocalQualityBacktracking

wp.set_module_options({"enable_backward": False, "fast_math": False})


@wp.kernel
def eligible(
    mode: wp.array[int],
    recovered: wp.array[int],
    accepted: wp.array[int],
    rejections: wp.array[int],
    before: wp.array[int],
    reason: wp.array[int],
    selection_error: wp.array[int],
    mesh_error: wp.array[int],
    boundary_error: wp.array[int],
    extraction_error: wp.array[int],
    extraction_valid: wp.array[int],
    diagnostics: wp.array[int],
    used: wp.array[int],
    budget: int,
    ready: wp.array[int],
    statistics: wp.array[int],
):
    fresh = (
        mode[0] == 1
        and recovered[0] != 0
        and accepted[0] == 0
        and rejections[0] > before[0]
        and reason[0] == 3
        and selection_error[0] == 16
        and mesh_error[0] == 0
        and boundary_error[0] == 0
        and extraction_error[0] == 0
        and extraction_valid[0] == 0
        and diagnostics[0] == 2
    )
    statistics[2] += int(fresh)
    ready[0] = int(fresh and used[0] < budget)


@wp.kernel
def flag_edges(
    indices: wp.array[int],
    count: wp.array[int],
    slots: wp.array[int],
    occurrences: wp.array[int],
    balance: wp.array[int],
    bad: wp.array[int],
):
    e = wp.tid()
    if e < 3 * count[0]:
        slot = slots[e]
        if occurrences[slot] != 2 or balance[slot] != 0:
            next_e = 3 * (e // 3) + (e + 1) % 3
            wp.atomic_or(bad, indices[e], 1)
            wp.atomic_or(bad, indices[next_e], 1)


@wp.kernel
def expand_star(
    cells: wp.array2d[int], count: wp.array[int], bad: wp.array[int], neighborhood: wp.array[int]
):
    i = wp.tid()
    if i < count[0]:
        adjacent = int(0)
        for j in range(4):
            adjacent |= bad[cells[i, j]]
        if adjacent:
            for j in range(4):
                wp.atomic_or(neighborhood, cells[i, j], 1)


@wp.kernel
def block_moved(
    active: wp.array[int],
    candidates: wp.array[int],
    roles: wp.array[int],
    mask: wp.array[int],
    original: wp.array[wp.vec3],
    points: wp.array[wp.vec3],
    blocked: wp.array[int],
    any_blocked: wp.array[int],
    statistics: wp.array[int],
):
    i = wp.tid()
    if i < active[0] and candidates[i] and mask[i] and roles[i] != 3:
        p, old = points[i], original[i]
        if p[0] != old[0] or p[1] != old[1] or p[2] != old[2]:
            if wp.atomic_or(blocked, i, 1) == 0:
                wp.atomic_or(any_blocked, 0, 1)
                wp.atomic_add(statistics, 1, 1)


@wp.kernel
def invalidate_history(history: wp.array[int], rejections: wp.array[int]):
    # A topology failure must never train a quality-only scale predictor.
    history[0] = 0
    history[4] = 0
    history[5] = rejections[0]


@wp.kernel
def record_accept(pending: wp.array[int], accepted: wp.array[int], statistics: wp.array[int]):
    statistics[3] += int(pending[0] != 0 and accepted[0] != 0)
    pending[0] = 0


class LocalTopologyBacktracking(LocalQualityBacktracking):
    def __init__(self, core, budget, neighborhood):
        if (
            core.local_backtracking is not None
            or core.controls is None
            or core.active_count is None
            or core.selection.recovery_extraction is None
        ):
            raise ValueError(
                "Experiment requires a recovered scheduler core without local rollback"
            )
        if budget < 1 or neighborhood not in ["endpoints", "star"]:
            raise ValueError("Use a positive budget and endpoints/star neighborhood")
        super().__init__(core, budget)
        self.neighborhood = neighborhood
        self.bad = wp.zeros(core.n, dtype=int, device=core.device)
        self.star = wp.zeros(core.n, dtype=int, device=core.device)
        self.pending = wp.zeros(1, dtype=int, device=core.device)
        # Complete-solve totals: retries, blocked sites, fresh obstructions,
        # accepted iterations immediately following a local retry.
        self.statistics = wp.zeros(4, dtype=int, device=core.device)
        core.local_backtracking = self
        wp.load_module(module=__name__, device=core.device)

    def reset(self):
        # The enclosing complete-mesher recorder resets the aggregate counters.
        self.start_step()

    def start_step(self):
        super().start_step()
        if hasattr(self, "pending"):
            self.pending.zero_()

    def _mark(self):
        c = self.core
        t = c.selection.recovery_extraction._topology
        self.bad.zero_()
        self.any_blocked.zero_()
        self._launch(
            flag_edges,
            len(t.boundary_indices),
            [
                t.boundary_indices,
                t.boundary_count,
                t._edge_slots,
                t._edge_counts,
                t._edge_balance,
                self.bad,
            ],
        )
        candidates = self.bad
        if self.neighborhood == "star":
            self.star.zero_()
            self._launch(
                expand_star,
                len(c.full_tetrahedra),
                [c.full_tetrahedra, c.full_tetrahedra_count, self.bad, self.star],
            )
            candidates = self.star
        self._launch(
            block_moved,
            c.n,
            [
                c.active_count,
                candidates,
                c.boundary.roles,
                c.selection.mask,
                c._original,
                c.points,
                self.blocked,
                self.any_blocked,
                self.statistics,
            ],
        )
        wp.copy(self.ready, self.any_blocked)

    def _retry(self):
        super()._retry()
        self.pending.fill_(1)
        c = self.core
        if c.step_history is not None:
            self._launch(invalidate_history, 1, [c.step_history.history, c.selection.rejections])

    def finish(self, ordinary_finish):
        c, s = self.core, self.core.selection
        e = s.recovery_extraction
        self._launch(
            eligible,
            1,
            [
                c.controls.mode,
                c.material_recovery,
                c._accepted,
                s.rejections,
                self.before,
                s.last_error,
                s.error,
                c.mesh.error,
                c.boundary.error,
                e.error,
                e.valid,
                e.diagnostics,
                self.used,
                self.budget,
                self.ready,
                self.statistics,
            ],
        )
        wp.capture_if(self.ready, self._mark)

        def ordinary():
            self.blocked.zero_()
            ordinary_finish()
            self._launch(record_accept, 1, [self.pending, c._accepted, self.statistics])

        wp.capture_if(self.ready, self._retry, ordinary)


def install(mesh, budget, neighborhood):
    trial = LocalTopologyBacktracking(mesh.core, budget, neighborhood)
    original = mesh._record

    def record():
        trial.statistics.zero_()
        original()

    mesh._record = record
    with wp.ScopedDevice(mesh.device), wp.ScopedCapture() as cap:
        mesh._record()
    mesh.graph = cap.graph
    return trial


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--neighborhood", choices=["baseline", "endpoints", "star"], required=True)
    p.add_argument("--budget", type=int, default=1)
    p.add_argument("--driver", choices=["gallery", "mesher", "loop"], default="gallery")
    p.add_argument("--report", type=Path, required=True)
    args, remaining = p.parse_known_args()
    if args.report.exists():
        p.error("Report exists")
    base = _mesher.VariationalMeshingWorkspace
    objects = []

    class Measured(base):
        def __init__(self, domain, *a, **kw):
            before = wp.get_mempool_used_mem_current(domain.device)
            start = time.perf_counter()
            super().__init__(domain, *a, **kw)
            trial = (
                None
                if args.neighborhood == "baseline"
                else install(self, args.budget, args.neighborhood)
            )
            objects.append(
                (
                    self,
                    trial,
                    time.perf_counter() - start,
                    wp.get_mempool_used_mem_current(domain.device) - before,
                )
            )

    report = dict(
        scope=__doc__,
        completed=False,
        parameters=vars(args) | {"report": str(args.report)},
        source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), *sorted(Path("warpower").glob("*.py"))]
        },
    )
    try:
        warpower.VariationalMeshingWorkspace = _mesher.VariationalMeshingWorkspace = Measured
        sys.argv = [sys.argv[0], *remaining]
        module = {"gallery": "examples.vtm_gallery", "mesher": "benchmarks.mesher", "loop": "benchmarks.mesher_loop"}[args.driver]
        runpy.run_module(module, run_name="__main__")
        report["completed"] = True
    finally:
        warpower.VariationalMeshingWorkspace = _mesher.VariationalMeshingWorkspace = base
        report["observations"] = [
            dict(
                setup_seconds=seconds,
                pool_bytes=storage,
                statistics=None if trial is None else trial.statistics.numpy().tolist(),
            )
            for _, trial, seconds, storage in objects
        ]
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
