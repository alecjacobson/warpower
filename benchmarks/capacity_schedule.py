"""Paired original-coordinate cell-capacity experiment; public defaults unchanged."""

import argparse
import hashlib
import json
import subprocess
import tempfile
import time
from itertools import product
from pathlib import Path

import numpy as np
import warp as wp

from benchmarks.datasets import generate
from benchmarks.run import cpu_quota_delta, cpu_quota_snapshot
from benchmarks.tetrahedra import canonical
from warpower import DelaunayWorkspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        default="uniform",
        choices=[
            "uniform",
            "gradient",
            "cluster5",
            "cluster10",
            "sphere",
            "lattice",
            "duplicates",
            "near-plane",
            "fitted-hand",
        ],
    )
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--checker", type=Path, required=True)
    parser.add_argument("--reference-connectivity", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 3 or args.n < 4:
        parser.error("Use at least three repetitions and four sites")
    if args.output.exists():
        parser.error("Output already exists")
    rng = np.random.default_rng(2026)
    if args.input:
        points = np.asarray(np.load(args.input)["points"], np.float32)
    elif args.case == "fitted-hand":
        points = np.load(Path(__file__).parents[1] / "tests/data/hand-fit256.npz")["points"]
    elif args.case == "lattice":
        points = np.array(list(product(range(round(args.n ** (1 / 3))), repeat=3)), np.float32)
    elif args.case in ["uniform", "gradient", "cluster5", "cluster10"]:
        points, _ = generate(args.n, args.case, seed=2026)
    else:
        points = rng.normal(size=(args.n, 3)).astype(np.float32)
        if args.case == "sphere":
            points /= np.linalg.norm(points, axis=1)[:, None]
        elif args.case == "duplicates":
            count = min(16, len(points) // 4)
            points[-count:] = points[:count]
        else:
            points[:, 2] *= np.float32(1e-5)
    points = np.asarray(points * np.float32(args.scale), np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(
        scope="Alternating complete certified fresh constructions; only the base FP64 capacity schedule differs. Setup, upload, output checking and independent CPU validation excluded from replay timing. No VTM or weighted-API promotion.",
        case=args.case,
        n=len(points),
        input=str(args.input),
        input_sha256=hashlib.sha256(points.tobytes()).hexdigest(),
        scale=args.scale,
        reference_connectivity=args.reference_connectivity,
        device=wp.get_device().name,
        warp=wp.__version__,
        source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), *sorted(Path("warpower").glob("*.py"))]
        },
        records={},
        validation_complete=False,
        eligible_comparison=False,
    )
    start = time.perf_counter()
    mesh = DelaunayWorkspace(points, cache_strategy=False)
    report["constructor_seconds"] = time.perf_counter() - start
    original = mesh._base._tet_capacities
    assert original == (128, 256, 512, 1024)
    graphs = {"128": mesh.graph}
    start = time.perf_counter()
    mesh._base._tet_capacities = (256, 512, 1024)
    with wp.ScopedCapture() as capture:
        mesh._record()
    graphs["256"] = capture.graph
    mesh._base._tet_capacities = original
    report["alternative_capture_seconds"] = time.perf_counter() - start
    report["additional_device_arrays_bytes"] = 0
    events = [wp.Event(enable_timing=True), wp.Event(enable_timing=True)]
    fields = [
        "error",
        "base_error",
        "certificate_error",
        "attempts",
        "recovered",
        "symbolic_attempted",
        "fast_attempted",
        "fast_accepted",
    ]
    results = {name: dict(samples_ms=[], cpu=[], diagnostics=[]) for name in graphs}
    for graph in graphs.values():
        wp.capture_launch(graph)
    wp.synchronize()
    outputs = {}
    for iteration in range(args.repeats):
        for name in list(graphs) if iteration % 2 == 0 else list(reversed(graphs)):
            before = cpu_quota_snapshot()
            wp.record_event(events[0])
            wp.capture_launch(graphs[name])
            wp.record_event(events[1])
            wp.synchronize_event(events[1])
            row = results[name]
            row["samples_ms"].append(wp.get_event_elapsed_time(*events))
            row["cpu"].append(cpu_quota_delta(before, cpu_quota_snapshot()))
            row["diagnostics"].append({key: int(getattr(mesh, key).numpy()[0]) for key in fields})
            count = int(mesh.tetrahedra_count.numpy()[0])
            cells = mesh.tetrahedra.numpy()[:count].copy()
            if name in outputs:
                assert np.array_equal(canonical(outputs[name]), canonical(cells))
            outputs[name] = cells
    for name, row in results.items():
        row["median_ms"] = float(np.median(row["samples_ms"]))
        row["successful_replays"] = all(d["error"] == 0 for d in row["diagnostics"])
        row["cpu_throttled"] = any(d["throttled_periods"] for r in row["cpu"] for d in r.values())
        row["tetrahedra"] = len(outputs[name])
        mesh_path = args.output.with_name(args.output.stem + "-" + name + ".npz")
        np.savez_compressed(mesh_path, points=points, tetrahedra=outputs[name])
        row["mesh_sha256"] = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
        report["records"] = results
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if row["successful_replays"]:
            with tempfile.TemporaryDirectory(
                prefix="capacity-check-", dir=args.output.parent
            ) as tmp:
                pp, tp = Path(tmp) / "points.txt", Path(tmp) / "tets.txt"
                np.savetxt(pp, points, fmt="%.17g", header=str(len(points)), comments="")
                np.savetxt(tp, outputs[name], fmt="%d", header=str(len(outputs[name])), comments="")
                command = [str(args.checker), str(pp), str(tp)]
                if args.reference_connectivity:
                    command += ["--cgal-connectivity"]
                check = subprocess.run(command, capture_output=True, text=True)
                row["independent"] = dict(
                    exit_code=check.returncode, stdout=check.stdout, stderr=check.stderr
                )
                row["independent_valid"] = (
                    check.returncode == 0 and json.loads(check.stdout)["valid"]
                )
        else:
            row["independent_valid"] = False
    report["records"] = results
    report["validation_complete"] = True
    report["same_connectivity"] = bool(
        np.array_equal(canonical(outputs["128"]), canonical(outputs["256"]))
    )
    report["eligible_comparison"] = all(
        r["successful_replays"] and r["independent_valid"] and not r["cpu_throttled"]
        for r in results.values()
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                case=args.case,
                n=len(points),
                eligible=report["eligible_comparison"],
                milliseconds={k: v["median_ms"] for k, v in results.items()},
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
