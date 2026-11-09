"""Run a native CGAL Mesh_3/ODT executable and independently measure every output."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from benchmarks.mesh_quality import measure
from benchmarks.run import cpu_quota_delta, cpu_quota_snapshot
from benchmarks.surface_distance import NativeSurfaceDistance


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


def add_optimizer_arguments(parser):
    for name in ("perturb", "exude"):
        parser.add_argument("--" + name, action="store_true")
        parser.add_argument("--" + name + "-seconds", type=float, default=1.0)
        parser.add_argument("--" + name + "-bound", type=float, default=10.0)


def optimizer_arguments(args):
    values = []
    for name in ("perturb", "exude"):
        seconds, bound = getattr(args, name + "_seconds"), getattr(args, name + "_bound")
        if not np.isfinite(seconds) or seconds < 0:
            raise ValueError("--" + name + "-seconds must be finite and nonnegative")
        if not np.isfinite(bound) or not 0 <= bound <= 180:
            raise ValueError("--" + name + "-bound must be finite and in [0,180]")
        values.extend([str(int(getattr(args, name))), str(seconds), str(bound)])
    # Historical binaries still receive their original positional arguments.
    return values if args.perturb or args.exude else []


def check_optimizer_report(args, native):
    for name in ("perturb", "exude"):
        enabled = getattr(args, name)
        if bool(native.get(name + "_enabled", False)) != enabled:
            raise ValueError("Native " + name + " enable flag does not match requested policy")
        if enabled:
            for suffix, option in [("_time_limit", "_seconds"), ("_sliver_bound", "_bound")]:
                if native.get(name + suffix) != getattr(args, name + option):
                    raise ValueError("Native " + name + suffix + " does not match requested policy")
            if native.get(name + "_return_code", -1) < 0:
                raise ValueError("Native " + name + " did not report completion")
    if args.exude and native.get("triangulation_kind") != "weighted_regular":
        raise ValueError("Exudation must report a weighted regular triangulation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--size", type=float, required=True)
    parser.add_argument("--facet-distance", type=float, required=True)
    parser.add_argument("--odt-iterations", type=int, default=50)
    parser.add_argument("--convergence", type=float, default=0.02)
    parser.add_argument("--freeze-bound", type=float, default=0.01)
    parser.add_argument("--feature-angle", type=float, default=45.0)
    parser.add_argument("--topology", choices=["non_manifold", "manifold"], default="non_manifold")
    parser.add_argument("--threads", type=int, choices=[1, 4], default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=600.0)
    add_optimizer_arguments(parser)
    parser.add_argument(
        "--distance-binary",
        type=Path,
        help="Optional native AABB distance oracle for offline quality checks",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not np.isfinite(args.feature_angle) or not 0 < args.feature_angle < 180:
        parser.error("--feature-angle must be finite and strictly between 0 and 180 degrees")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if not np.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    try:
        optimizers = optimizer_arguments(args)
    except ValueError as error:
        parser.error(str(error))
    points, faces = read_surface(args.surface)
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        parser.error("--output must be an empty directory; existing evidence is never overwritten")
    records = []
    checkpoint = args.output / "runs.json"

    def save_runs(stage):
        checkpoint.write_text(json.dumps(dict(stage=stage, runs=records), indent=2) + "\n")

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
        # Keep the default command compatible with historical 45-degree binaries.
        if args.feature_angle != 45.0 or args.topology != "non_manifold" or optimizers:
            command.append(str(args.feature_angle))
        if args.topology != "non_manifold" or optimizers:
            command.append(args.topology)
        command.extend(optimizers)
        start = time.perf_counter()
        before = cpu_quota_snapshot()
        timed_out = False
        try:
            result = subprocess.run(command, text=True, capture_output=True, timeout=args.timeout)
        except subprocess.TimeoutExpired as error:
            timed_out = True

            def decode(value):
                return value.decode(errors="replace") if isinstance(value, bytes) else value or ""

            result = subprocess.CompletedProcess(
                command, 124, decode(error.stdout), decode(error.stderr)
            )
        quota = cpu_quota_delta(before, cpu_quota_snapshot())
        wall_seconds = time.perf_counter() - start
        (args.output / f"run-{i}.stdout").write_text(result.stdout)
        (args.output / f"run-{i}.stderr").write_text(result.stderr)
        record = dict(
            command=command,
            exit_code=result.returncode,
            wall_seconds=wall_seconds,
            timed_out=timed_out,
            cpu_quota_delta=quota,
            cpu_throttled=any(
                x["throttled_periods"] or x["throttled_usec"] for x in quota.values()
            ),
            native_valid=False,
            quality_valid=False,
        )
        if mesh_path.exists():
            record["mesh_sha256"] = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
        records.append(record)
        save_runs("native")
        try:
            if result.returncode != 0:
                raise RuntimeError(f"Native process exited {result.returncode}")
            native = json.loads(result.stdout)
            if native["feature_angle"] != args.feature_angle:
                raise ValueError("Native feature angle does not match requested policy")
            if native.get("topology", "non_manifold") != args.topology:
                raise ValueError("Native topology does not match requested policy")
            check_optimizer_report(args, native)
            if "mesh_sha256" not in record:
                raise ValueError("Native process did not produce a mesh")
            if "triangulation_kind" in native:
                weights_path = Path(str(mesh_path) + ".weights")
                weights = np.loadtxt(weights_path, skiprows=1, ndmin=1)
                with weights_path.open() as weight_file:
                    weight_count = int(weight_file.readline())
                if (
                    weights.shape != (native["vertices"],)
                    or weight_count != len(weights)
                    or not np.isfinite(weights).all()
                ):
                    raise ValueError("Malformed native vertex weights")
                if np.count_nonzero(weights) != native["nonzero_exported_weights"]:
                    raise ValueError("Native weight count disagrees with export")
                record["weights_sha256"] = hashlib.sha256(weights_path.read_bytes()).hexdigest()
            record.update(native)
            record["native_valid"] = True
            record["topology_native_reported"] = "topology" in native
        except Exception as error:
            record["native_error"] = dict(type=type(error).__name__, message=str(error))
        save_runs("native")
    # All timed native executions finish before independent validation starts.
    save_runs("validation")
    oracle = (
        NativeSurfaceDistance(args.distance_binary, args.output) if args.distance_binary else None
    )
    quality = {}
    errors = {}
    for i, record in enumerate(records):
        if not record["native_valid"]:
            continue
        digest = record["mesh_sha256"]
        if digest not in quality and digest not in errors:
            try:
                p, t = read_mesh(args.output / f"mesh-{i}.txt")
                quality[digest] = measure(p, t, points, faces, distance_query=oracle)
                if oracle is not None:
                    quality[digest]["distance_oracle"] = oracle.metadata
            except Exception as error:
                errors[digest] = dict(
                    type=type(error).__name__,
                    message=str(error),
                    stage="independent_material_quality_and_topology",
                )
        if digest in quality:
            record["independent_quality"] = quality[digest]
            record["quality_valid"] = True
        else:
            record["quality_error"] = errors[digest]
        save_runs("validation")
    valid = [x for x in records if x["native_valid"] and x["quality_valid"]]
    clean = [x["complete_seconds"] for x in valid if not x["cpu_throttled"]]
    native_valid = [x for x in records if x["native_valid"]]
    cpu_model = next(
        (
            x.split(":", 1)[1].strip()
            for x in Path("/proc/cpuinfo").read_text().splitlines()
            if x.startswith("model name")
        ),
        "unknown",
    )
    report = dict(
        scope="Fresh native CGAL Mesh_3 with feature protection, optional ODT, perturbation then exudation. Complete time includes domain construction, meshing and optimization; prepared-domain time excludes domain construction. File I/O, triangulation validation, serialization and independent quality checks are separate. No GPU speedup is inferred by this runner.",
        platform=platform.platform(),
        cpu_model=cpu_model,
        visible_cpu_count=os.cpu_count(),
        allowed_cpu_ids=sorted(os.sched_getaffinity(0)),
        cpu_quota_after_validation=cpu_quota_snapshot(),
        binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        surface_sha256=hashlib.sha256(args.surface.read_bytes()).hexdigest(),
        points_float32_sha256=hashlib.sha256(points.astype(np.float32).tobytes()).hexdigest(),
        parameters={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        complete=True,
        valid=len(valid) == args.repeats,
        successful_samples=len(valid),
        native_successful_samples=len(native_valid),
        median_native_complete_seconds=float(
            np.median([x["complete_seconds"] for x in native_valid])
        )
        if native_valid
        else None,
        native_unthrottled_samples=sum(not x["cpu_throttled"] for x in native_valid),
        median_complete_seconds=float(np.median([x["complete_seconds"] for x in valid]))
        if valid
        else None,
        unthrottled_samples=len(clean),
        median_unthrottled_complete_seconds=float(np.median(clean)) if clean else None,
        median_unthrottled_prepared_domain_seconds=float(
            np.median(
                [
                    x["prepared_domain_seconds"]
                    for x in valid
                    if not x["cpu_throttled"] and "prepared_domain_seconds" in x
                ]
            )
        )
        if any(not x["cpu_throttled"] and "prepared_domain_seconds" in x for x in valid)
        else None,
        runs=records,
    )
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    save_runs("complete")
    print(json.dumps(report), flush=True)
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
