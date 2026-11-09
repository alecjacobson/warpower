"""Read-only device attribution of selective extraction failures (#258).

No observed duration is a performance result. Production never imports this
module. The observer only reads solver buffers and writes separate trace arrays.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import warp as wp

from benchmarks.tetrahedra import canonical
from examples.vtm_inputs import read_off
from warpower import SurfaceDomain, VariationalMeshingWorkspace

wp.set_module_options({"enable_backward": False, "fast_math": False})

COLUMNS = [
    "core_call",
    "iteration",
    "halving",
    "active",
    "full_count",
    "recovered_mode",
    "accepted",
    "selection_rejections",
    "fresh_rejection",
    "selection_error",
    "guard_reason",
    "retained_count",
    "extraction_error",
    "extraction_valid",
    "topology_diagnostics",
    "extraction_count",
    "containment_additions",
    "containment_rounds",
    "mesh_error",
    "boundary_error",
    "snapshot",
]


@wp.struct
class Inputs:
    mode: wp.array[int]
    iteration: wp.array[int]
    halving: wp.array[int]
    active: wp.array[int]
    full_count: wp.array[int]
    recovered: wp.array[int]
    accepted: wp.array[int]
    rejections: wp.array[int]
    selection_error: wp.array[int]
    reason: wp.array[int]
    retained: wp.array[int]
    extraction_error: wp.array[int]
    valid: wp.array[int]
    diagnostics: wp.array[int]
    extraction_count: wp.array[int]
    additions: wp.array[int]
    rounds: wp.array[int]
    mesh_error: wp.array[int]
    boundary_error: wp.array[int]
    alpha: wp.array[wp.float64]
    minimum: wp.array[wp.float64]
    previous_minimum: wp.array[wp.float64]


@wp.kernel
def begin_call(state: wp.array[int]):
    state[0] += 1
    state[1] = 0


@wp.kernel
def record(
    s: Inputs,
    state: wp.array[int],
    rows: wp.array2d[int],
    values: wp.array2d[wp.float64],
    slots: int,
):
    # State: call, prior rejection counter, event count, saved call, saved count,
    # trace overflow, snapshot overflow, current snapshot index.
    state[7] = -1
    if s.mode[0] != 1:
        return
    index = state[2]
    state[2] += 1
    fresh = s.rejections[0] > state[1]
    state[1] = s.rejections[0]
    if index >= rows.shape[0]:
        state[5] = 1
        return
    snapshot = int(-1)
    if fresh and s.selection_error[0] != 0 and s.recovered[0] != 0 and state[3] != state[0]:
        state[3] = state[0]
        snapshot = state[4]
        state[4] += 1
        if snapshot < slots:
            state[7] = snapshot
        else:
            state[6] = 1
            snapshot = -1
    rows[index, 0] = state[0]
    rows[index, 1] = s.iteration[0]
    rows[index, 2] = s.halving[0]
    rows[index, 3] = s.active[0]
    rows[index, 4] = s.full_count[0]
    rows[index, 5] = s.recovered[0]
    rows[index, 6] = s.accepted[0]
    rows[index, 7] = s.rejections[0]
    rows[index, 8] = int(fresh)
    rows[index, 9] = s.selection_error[0]
    rows[index, 10] = s.reason[0]
    rows[index, 11] = s.retained[0]
    # Recovered extraction fields are meaningful only for a fresh rejection
    # from that path; stale ordinary/global values are explicitly unavailable.
    for j in range(12, 18):
        rows[index, j] = -1
    if fresh and s.recovered[0] != 0:
        rows[index, 12] = s.extraction_error[0]
        rows[index, 13] = s.valid[0]
        rows[index, 14] = s.diagnostics[0]
        rows[index, 15] = s.extraction_count[0]
        rows[index, 16] = s.additions[0]
        rows[index, 17] = s.rounds[0]
    rows[index, 18] = s.mesh_error[0]
    rows[index, 19] = s.boundary_error[0]
    rows[index, 20] = snapshot
    values[index, 0] = s.alpha[0]
    values[index, 1] = s.minimum[0]
    values[index, 2] = s.previous_minimum[0]


@wp.kernel
def save_points(
    state: wp.array[int],
    active: wp.array[int],
    points: wp.array[wp.vec3],
    roles: wp.array[int],
    output: wp.array2d[wp.vec3],
    saved_roles: wp.array2d[int],
):
    i = wp.tid()
    slot = state[7]
    if slot >= 0 and i < active[0]:
        output[slot, i] = points[i]
        saved_roles[slot, i] = roles[i]


@wp.kernel
def save_cells(
    state: wp.array[int],
    count: wp.array[int],
    cells: wp.array2d[int],
    selected: wp.array[int],
    output: wp.array3d[int],
    saved_selected: wp.array2d[int],
):
    i = wp.tid()
    slot = state[7]
    if slot >= 0 and i < count[0]:
        for j in range(4):
            output[slot, i, j] = cells[i, j]
        saved_selected[slot, i] = selected[i]


class Observer:
    def __init__(self, mesh, *, events=8192, snapshots=16):
        c = mesh.core
        e = c.selection.recovery_extraction
        if e is None or c.controls is None or c.local_backtracking is not None:
            raise ValueError(
                "Observer requires a scheduler-owned recovered selective core without local rollback"
            )
        self.mesh = mesh
        self.slots = snapshots
        self.device = c.device
        with wp.ScopedDevice(c.device):
            self.state = wp.zeros(8, dtype=int)
            self.rows = wp.empty((events, len(COLUMNS)), dtype=int)
            self.values = wp.empty((events, 3), dtype=wp.float64)
            self.points = wp.empty((snapshots, c.n), dtype=wp.vec3)
            self.roles = wp.empty((snapshots, c.n), dtype=int)
            self.cells = wp.empty((snapshots, len(c.full_tetrahedra), 4), dtype=int)
            self.selected = wp.empty((snapshots, len(c.full_tetrahedra)), dtype=int)
            self.inputs = Inputs()
            source = dict(
                mode=c.controls.mode,
                iteration=c.iterations,
                halving=c._backtracks,
                active=c.active_count,
                full_count=c.full_tetrahedra_count,
                recovered=c.material_recovery,
                accepted=c._accepted,
                rejections=c.selection.rejections,
                selection_error=c.selection.error,
                reason=c.selection.last_error,
                retained=c.selection.retained_count,
                extraction_error=e.error,
                valid=e.valid,
                diagnostics=e.diagnostics,
                extraction_count=e.tetrahedra_count,
                additions=e.repaired_count,
                rounds=e.repair_rounds,
                mesh_error=c.mesh.error,
                boundary_error=c.boundary.error,
                alpha=c._alpha,
                minimum=c.selection.minimum,
                previous_minimum=c.selection.previous_minimum,
            )
            for name, value in source.items():
                setattr(self.inputs, name, value)
            wp.load_module(module=__name__, device=c.device)
            core_record, finish, whole_record = c._record, c._finish_attempt, mesh._record

            def core():
                wp.launch(begin_call, 1, inputs=[self.state])
                core_record()

            def observe():
                wp.launch(
                    record, 1, inputs=[self.inputs, self.state, self.rows, self.values, snapshots]
                )
                wp.launch(
                    save_points,
                    c.n,
                    inputs=[
                        self.state,
                        c.active_count,
                        c.points,
                        c.boundary.roles,
                        self.points,
                        self.roles,
                    ],
                )
                wp.launch(
                    save_cells,
                    len(c.full_tetrahedra),
                    inputs=[
                        self.state,
                        c.full_tetrahedra_count,
                        c.full_tetrahedra,
                        e.selected,
                        self.cells,
                        self.selected,
                    ],
                )
                finish()

            def whole():
                self.state.zero_()
                whole_record()

            c._record, c._finish_attempt, mesh._record = core, observe, whole
            with wp.ScopedCapture() as capture:
                mesh._record()
            mesh.graph = capture.graph
        mesh.failure_observer = self

    def export(self, folder, input_points, input_faces):
        state = self.state.numpy()
        count = min(int(state[2]), len(self.rows))
        rows, values = self.rows.numpy()[:count], self.values.numpy()[:count]
        np.savez_compressed(folder / "events.npz", rows=rows, values=values)
        p, r, c, s = (
            self.points.numpy(),
            self.roles.numpy(),
            self.cells.numpy(),
            self.selected.numpy(),
        )
        for row in rows:
            slot = int(row[-1])
            if slot >= 0:
                n, nt = int(row[3]), int(row[4])
                np.savez_compressed(
                    folder / f"failure-{slot}.npz",
                    points=p[slot, :n],
                    full_tetrahedra=c[slot, :nt],
                    selected=s[slot, :nt],
                    roles=r[slot, :n],
                    input_points=input_points,
                    input_faces=input_faces,
                )
        return dict(
            columns=COLUMNS,
            float_columns=["alpha", "minimum", "previous_minimum"],
            state=state.tolist(),
            rows=rows.tolist(),
            values=values.tolist(),
            overflow=bool(state[5] or state[6]),
        )


def final_arrays(mesh):
    n, nf, nt, nb = map(int, mesh.counts.numpy())
    return dict(
        points=mesh.points.numpy()[:n],
        full_tetrahedra=mesh.full_tetrahedra.numpy()[:nf],
        tetrahedra=mesh.tetrahedra.numpy()[:nt],
        boundary=mesh.boundary_faces.numpy()[:nb],
        roles=mesh.core.boundary.roles.numpy()[:n],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--query-backend", choices=["reference", "native"], default="reference")
    parser.add_argument(
        "--boundary-query-precision", choices=["float64", "float32", "mixed"], default="float64"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    points, faces = read_off(args.input)
    domain = SurfaceDomain(
        points,
        faces,
        surface_subdivisions=1,
        crease_subdivisions=1,
        feature_angle=120,
        corner_angle=45,
        query_backend=args.query_backend,
        validate_geometry=False,
    )
    mesh = VariationalMeshingWorkspace(
        domain,
        args.n,
        initial_count=256,
        seed=2026,
        quality_rounds=4,
        boundary_query_precision=args.boundary_query_precision,
    )
    mesh.run()
    assert mesh.valid.numpy()[0] == 1 and mesh.error.numpy()[0] == 0
    before = final_arrays(mesh)
    observer = Observer(mesh)
    mesh.run()
    assert mesh.valid.numpy()[0] == 1 and mesh.error.numpy()[0] == 0
    after = final_arrays(mesh)
    comparison = {
        k: bool(np.array_equal(canonical(before[k]), canonical(after[k])))
        if k in ["full_tetrahedra", "tetrahedra"]
        else bool(np.array_equal(before[k], after[k]))
        for k in before
    }
    for name, data in [("before", before), ("after", after)]:
        np.savez_compressed(
            args.output / (name + ".npz"), **data, input_points=points, input_faces=faces
        )
    report = observer.export(args.output, points, faces)
    report.update(
        scope=__doc__,
        comparison=comparison,
        source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), *sorted(Path("warpower").glob("*.py"))]
        },
        parameters={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                events=len(report["rows"]),
                snapshots=report["state"][4],
                overflow=report["overflow"],
                comparison=comparison,
            )
        ),
        flush=True,
    )
    assert not report["overflow"]


if __name__ == "__main__":
    main()
