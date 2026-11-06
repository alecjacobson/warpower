"""Measure complete Delaunay recovery of the difficult first hand-refinement batch."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

import warpower
from benchmarks.run import cpu_quota_delta, cpu_quota_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-expansions", type=int, default=16)
    parser.add_argument("--initial-factor", type=float, default=1.0)
    parser.add_argument("--symbolic-first", action="store_true")
    parser.add_argument("--facet-recovery", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--no-symbolic", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-mesh", type=Path)
    args = parser.parse_args()
    if args.repeats < 0 or not np.isfinite(args.initial_factor) or args.initial_factor < 1:
        parser.error("Require repeats >= 0 and a finite initial-factor >= 1")
    data = Path(__file__).parents[1] / "tests/data/hand-refined384.npz"
    points = np.load(data)["points"]
    expected = json.loads(data.with_suffix(".json").read_text())
    setup = time.perf_counter()
    workspace = warpower.DelaunayWorkspace(
        points,
        symbolic_max_expansions=args.max_expansions,
        symbolic_recovery=not args.no_symbolic,
        **({"facet_recovery": args.facet_recovery} if args.facet_recovery is not None else {}),
    )
    if args.symbolic_first:
        workspace._preferred.fill_(2)
    if workspace._symbolic is not None:
        workspace._symbolic._factor.fill_(args.initial_factor)
    elif args.symbolic_first or args.initial_factor != 1:
        parser.error("Symbolic strategy overrides require symbolic recovery")
    wp.synchronize()
    setup = time.perf_counter() - setup
    begin, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
    records = []
    canonical = None
    for replay in range(args.repeats + 1):
        quota = cpu_quota_snapshot()
        start = time.perf_counter()
        wp.record_event(begin)
        workspace.run()
        wp.record_event(end)
        wp.synchronize_event(end)
        wall = time.perf_counter() - start
        cpu = cpu_quota_delta(quota, cpu_quota_snapshot())
        record = dict(
            replay=replay, gpu_ms=wp.get_event_elapsed_time(begin, end), wall_seconds=wall, cpu=cpu
        )
        for name in [
            "error",
            "tetrahedra_count",
            "attempts",
            "certificate_error",
            "symbolic_error",
            "symbolic_expansions",
        ]:
            record[name] = int(getattr(workspace, name).numpy()[0])
        for name in [
            "facet_attempted",
            "facet_recovered",
            "facet_error",
            "facet_rounds",
            "facet_seeds",
        ]:
            if hasattr(workspace, name):
                record[name] = int(getattr(workspace, name).numpy()[0])
        record["final_box_factor"] = (
            float(workspace._symbolic._factor.numpy()[0])
            if workspace._symbolic is not None
            else None
        )
        records.append(record)
        if record["error"]:
            break
        cells = workspace.tetrahedra.numpy()[: record["tetrahedra_count"]]
        sorted_cells = np.sort(cells, axis=1)
        canonical = np.ascontiguousarray(sorted_cells[np.lexsort(sorted_cells[:, ::-1].T)])
        assert len(cells) == expected["expected_tetrahedra"]
        assert (
            hashlib.sha256(canonical.tobytes()).hexdigest()
            == expected["expected_canonical_connectivity_sha256"]
        )
    runtime = Path(warpower.__file__).parent
    result = dict(
        scope="Complete resident-device Delaunay recovery of 384 fixed original float32 sites, including all attempted strategies and box expansions. First replay and subsequent cached-strategy replays are separate. Setup, readback and connectivity validation are excluded from timing. This is not a full VTM solve or CPU comparison.",
        parameters={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        facet_recovery_effective=getattr(workspace, "facet_recovery", False),
        device=workspace.device.name,
        warp=wp.__version__,
        input_sha256=hashlib.sha256(points.tobytes()).hexdigest(),
        runtime_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(runtime.glob("*.py"))
        },
        setup_seconds=setup,
        replays=records,
        canonical_connectivity_matches=canonical is not None and not records[-1]["error"],
        median_additional_gpu_ms=float(np.median([r["gpu_ms"] for r in records[1:]]))
        if len(records) > 1
        else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if records[-1]["error"]:
        raise SystemExit("Delaunay recovery failed; diagnostics saved")
    if args.save_mesh:
        args.save_mesh.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save_mesh, points=points, tetrahedra=cells)


if __name__ == "__main__":
    main()
