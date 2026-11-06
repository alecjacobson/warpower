"""Run a native CGAL Mesh_3/ODT executable and independently measure every output."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

import numpy as np

from benchmarks.mesh_quality import measure
from benchmarks.run import cpu_quota_delta, cpu_quota_snapshot


def read_surface(path):
    # Accept the explicit triangular OFF format exported by the gallery.
    lines = [line.split("#", 1)[0].strip() for line in path.read_text().splitlines()]
    lines = [line for line in lines if line]
    if lines[0] != "OFF":
        raise ValueError("Expected triangular ASCII OFF with counts on the second line")
    n, m, _ = map(int, lines[1].split())
    points = np.asarray([list(map(float, x.split())) for x in lines[2 : 2 + n]], np.float64)
    rows = [list(map(int, x.split())) for x in lines[2 + n : 2 + n + m]]
    if points.shape != (n, 3) or len(rows) != m or any(len(x) != 4 or x[0] != 3 for x in rows):
        raise ValueError("Expected a triangular surface")
    if not np.isfinite(points).all():
        raise ValueError("Surface positions must be finite")
    if not np.array_equal(points, points.astype(np.float32).astype(np.float64)):
        raise ValueError("Use the gallery float32 round-trip OFF for identical GPU/CPU coordinates")
    faces = np.asarray([x[1:] for x in rows], np.int32)
    if faces.min() < 0 or faces.max() >= n:
        raise ValueError("Invalid triangle indices")
    return points, faces


def read_mesh(path):
    with path.open() as f:
        n, m = map(int, f.readline().split())
        points = np.array([list(map(float, f.readline().split())) for _ in range(n)])
        tets = np.array([list(map(int, f.readline().split())) for _ in range(m)], np.int32)
    if points.shape != (n, 3) or tets.shape != (m, 4):
        raise ValueError("Malformed native mesh output")
    return points, tets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--size", type=float, required=True)
    parser.add_argument("--facet-distance", type=float, required=True)
    parser.add_argument("--odt-iterations", type=int, default=50)
    parser.add_argument("--convergence", type=float, default=0.02)
    parser.add_argument("--freeze-bound", type=float, default=0.01)
    parser.add_argument("--threads", type=int, choices=[1, 4], default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    points, faces = read_surface(args.surface)
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    # Measure all native executions before running the independent quality oracle.
    for i in range(args.repeats):
        mesh_path = args.output / f"mesh-{i}.txt"
        command = [
            str(args.binary.resolve()),
            str(args.surface.resolve()),
            str(mesh_path.resolve()),
            str(args.size),
            str(args.facet_distance),
            str(args.odt_iterations),
            str(args.threads),
            str(args.convergence),
            str(args.freeze_bound),
        ]
        before = cpu_quota_snapshot()
        result = subprocess.run(command, text=True, capture_output=True, check=True)
        quota = cpu_quota_delta(before, cpu_quota_snapshot())
        (args.output / f"run-{i}.stdout").write_text(result.stdout)
        (args.output / f"run-{i}.stderr").write_text(result.stderr)
        record = json.loads(result.stdout)
        record["command"] = command
        record["cpu_quota_delta"] = quota
        record["cpu_throttled"] = any(
            x["throttled_periods"] or x["throttled_usec"] for x in quota.values()
        )
        record["mesh_sha256"] = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
        records.append(record)
    quality = {}
    for i, record in enumerate(records):
        digest = record["mesh_sha256"]
        if digest not in quality:
            p, t = read_mesh(args.output / f"mesh-{i}.txt")
            quality[digest] = measure(p, t, points, faces)
        record["independent_quality"] = quality[digest]
    clean = [x["complete_seconds"] for x in records if not x["cpu_throttled"]]
    cpu_model = next(
        (
            x.split(":", 1)[1].strip()
            for x in Path("/proc/cpuinfo").read_text().splitlines()
            if x.startswith("model name")
        ),
        "unknown",
    )
    report = dict(
        scope="Fresh native CGAL Mesh_3 with feature protection and optional ODT. Complete time includes domain construction, meshing and optimization; file I/O, triangulation validation, serialization and independent quality checks are separate. No GPU speedup is inferred by this runner.",
        platform=platform.platform(),
        cpu_model=cpu_model,
        visible_cpu_count=os.cpu_count(),
        allowed_cpu_ids=sorted(os.sched_getaffinity(0)),
        cpu_quota_after_validation=cpu_quota_snapshot(),
        binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        surface_sha256=hashlib.sha256(args.surface.read_bytes()).hexdigest(),
        points_float32_sha256=hashlib.sha256(points.astype(np.float32).tobytes()).hexdigest(),
        parameters={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        median_complete_seconds=float(np.median([x["complete_seconds"] for x in records])),
        unthrottled_samples=len(clean),
        median_unthrottled_complete_seconds=float(np.median(clean)) if clean else None,
        runs=records,
    )
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
