"""Compare fresh GPU neighbor construction with explicit output contracts.

Run each backend/case in a separate process. CUDA event spans include device
idle gaps; use a separate Nsight capture to measure kernel execution time.
"""

import argparse
import ctypes
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from benchmarks.datasets import generate
from benchmarks.run import cpu_quota_delta, cpu_quota_snapshot


def gpu_snapshot():
    """Record GPU utilization and competing CUDA processes outside timing."""
    return {
        query: subprocess.check_output(
            ["nvidia-smi", f"--query-{query}={fields}", "--format=csv,noheader"],
            text=True,
            timeout=10,
        )
        .strip()
        .splitlines()
        for query, fields in (
            (
                "gpu",
                "name,uuid,driver_version,memory.used,utilization.gpu,temperature.gpu,clocks.sm,power.draw",
            ),
            ("compute-apps", "pid,process_name,used_memory"),
        )
    }


def main():
    """Measure warmed fresh construction and validate the final CSR."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("radfoam", "warpower", "paragram"), required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--n", type=int, default=1_000_000)
    parser.add_argument("--distribution", choices=("uniform", "cluster5"), default="uniform")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--module-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--profile", action="store_true", help="Capture measured calls with cudaProfilerStart/Stop"
    )
    parser.add_argument(
        "--save-graph",
        type=Path,
        help="Save final CSR and original-index permutation outside timing",
    )
    args = parser.parse_args()
    if args.warmups < 1 or args.repeats < 1:
        parser.error("warmups and repeats must be positive")
    if args.backend == "radfoam" and args.module_dir is None:
        parser.error("radfoam requires --module-dir")

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.input:
        with np.load(args.input) as data:
            points = np.ascontiguousarray(data["points"], dtype=np.float32)
        input_name = args.input.name
    else:
        points, _ = generate(args.n, args.distribution, args.seed)
        input_name = f"{args.distribution}-{args.n}-seed{args.seed}"
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be a finite N-by-3 array")

    report = {
        "complete": False,
        "backend": args.backend,
        "input": input_name,
        "n": len(points),
        "points_sha256": hashlib.sha256(points.tobytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "warpower_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "versions": {"python": platform.python_version(), "torch": torch.__version__},
        "cpu_threads": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "torch_threads": [torch.get_num_threads(), torch.get_num_interop_threads()],
        "warmups": [],
        "samples": [],
        "profiled": args.profile,
        "timing": "Synchronized fresh construction including allocation; resident input; preceding result destroyed before each call; setup, JIT, readback and validation excluded. CUDA events measure elapsed spans, including idle gaps, not summed kernel time.",
    }
    if args.backend == "warpower":
        import warp as wp

        import warpower

        wp.init()
        wp.set_mempool_release_threshold("cuda:0", 32 * 1024**3)
        device_points = wp.array(points, dtype=wp.vec3, device="cuda:0")
        construct = lambda: warpower.voronoi_diagram(device_points)
        synchronize = lambda: wp.synchronize_device("cuda:0")
        start_event = wp.Event("cuda:0", enable_timing=True)
        end_event = wp.Event("cuda:0", enable_timing=True)
        record_start = lambda: wp.record_event(start_event)
        record_end = lambda: wp.record_event(end_event)
        elapsed_ms = lambda: wp.get_event_elapsed_time(start_event, end_event)
        report.update(
            output_contract="Bounded Voronoi neighbor CSR in original input order; no tetrahedra",
            warp_retention_gib=32,
        )
        report["versions"]["warp"] = wp.__version__
    else:
        device_points = torch.from_numpy(points).cuda()
        synchronize = torch.cuda.synchronize
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        record_start = start_event.record
        record_end = end_event.record
        elapsed_ms = lambda: start_event.elapsed_time(end_event)
        if args.backend == "radfoam":
            sys.path.insert(0, str(args.module_dir.resolve()))
            import radfoam_headless as native

            construct = lambda: native.Triangulation(device_points)
            report.update(
                output_contract="Unbounded Delaunay neighbor CSR and tetrahedra in internal sorted order, plus original-index permutation; no original-order export",
                native_module_sha256=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
            )
        else:
            import paragram

            construct = lambda: paragram.voronoi_diagram(device_points)
            report.update(
                output_contract="Bounded Voronoi neighbor CSR in original input order; no tetrahedra"
            )

    runtime = None
    if args.profile:
        runtime_path = next(
            Path(sys.prefix).glob(
                "lib/python*/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12"
            )
        )
        runtime = ctypes.CDLL(str(runtime_path))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    result = None
    synchronize()
    report["gpu_before"] = gpu_snapshot()
    save()
    try:
        for repeat in range(-args.warmups, args.repeats):
            del result
            result = None
            synchronize()
            quota_before = cpu_quota_snapshot()
            if repeat == 0:
                report["gpu_measurement_start"] = gpu_snapshot()
                quota_before = cpu_quota_snapshot()
                if runtime is not None and runtime.cudaProfilerStart() != 0:
                    raise RuntimeError("cudaProfilerStart failed")
            wall_start = time.perf_counter()
            record_start()
            result = construct()
            record_end()
            synchronize()
            wall_ms = 1000 * (time.perf_counter() - wall_start)
            if repeat == args.repeats - 1 and runtime is not None:
                if runtime.cudaProfilerStop() != 0:
                    raise RuntimeError("cudaProfilerStop failed")
            quota = cpu_quota_delta(quota_before, cpu_quota_snapshot())
            sample = {
                "repeat": repeat,
                "wall_ms": wall_ms,
                "cuda_span_ms": elapsed_ms(),
                "cpu_quota": quota,
                "cpu_throttled": any(value["throttled_periods"] for value in quota.values()),
            }
            report["warmups" if repeat < 0 else "samples"].append(sample)
            save()
            print(json.dumps(sample), flush=True)

        report["gpu_after"] = gpu_snapshot()
        if args.backend == "radfoam":
            offsets = result.point_adjacency_offsets().cpu().numpy()
            adjacency = result.point_adjacency().cpu().numpy()
            permutation = result.permutation().cpu().numpy()
            report["tetrahedra"] = len(result.tets())
            np.testing.assert_array_equal(np.sort(permutation), np.arange(len(points)))
        else:
            offsets = (
                result.offsets.numpy()
                if args.backend == "warpower"
                else result.offsets.cpu().numpy()
            )
            adjacency = (
                result.adjacency.numpy()
                if args.backend == "warpower"
                else result.adjacency.cpu().numpy()
            )
            status = (
                result.status.numpy() if args.backend == "warpower" else result.status.cpu().numpy()
            )
            values, counts = np.unique(status, return_counts=True)
            report["status_counts"] = {
                str(value): int(count) for value, count in zip(values, counts)
            }
            report["failed_cells"] = int(np.count_nonzero((status != 0) & (status != 3)))
            permutation = np.arange(len(points), dtype=np.uint32)
            if args.backend == "warpower":
                report["diagnostics"] = result.diagnostics
        np.testing.assert_equal(offsets.shape, (len(points) + 1,))
        np.testing.assert_equal(offsets[0], 0)
        np.testing.assert_equal(offsets[-1], len(adjacency))
        np.testing.assert_array_less(-1, np.diff(offsets.astype(np.int64)))
        np.testing.assert_array_less(-1, adjacency.astype(np.int64))
        np.testing.assert_array_less(adjacency, len(points))
        input_after = (
            device_points.numpy() if args.backend == "warpower" else device_points.cpu().numpy()
        )
        np.testing.assert_array_equal(input_after, points)
        report.update(
            nnz=len(adjacency),
            max_degree=int(np.diff(offsets.astype(np.int64)).max()),
            structural_checks="CSR offset shape/endpoints/monotonicity, index bounds, permutation bijection and unchanged input pass; not a geometric certificate",
        )
        if args.save_graph:
            args.save_graph.parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.save_graph, offsets=offsets, adjacency=adjacency, permutation=permutation)
        report["summary"] = {
            metric: {
                "mean": float(np.mean([sample[metric] for sample in report["samples"]])),
                "median": float(np.median([sample[metric] for sample in report["samples"]])),
                "min": min(sample[metric] for sample in report["samples"]),
                "max": max(sample[metric] for sample in report["samples"]),
            }
            for metric in ("wall_ms", "cuda_span_ms")
        }
        report["cpu_throttled"] = any(sample["cpu_throttled"] for sample in report["samples"])
        report["complete"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
