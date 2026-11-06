"""Paired standalone-geometry regression gate against a persistent checkout."""

import argparse
import gc
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp
from scipy.stats import t as student_t

import warpower
from benchmarks.datasets import generate
from benchmarks.default_path import trace
from benchmarks.run import cpu_quota_delta, cpu_quota_snapshot
from benchmarks.tetrahedra import canonical


def load_baseline(path):
    name = "warpower_regression_baseline"
    spec = importlib.util.spec_from_file_location(
        name,
        path / "warpower/__init__.py",
        submodule_search_locations=[str(path / "warpower")],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check_outputs(a, b, adjacency, tetrahedra, graph):
    if graph:
        assert a.error.numpy()[0] == b.error.numpy()[0] == 0
    if adjacency:
        offsets = a.offsets.numpy()
        assert np.array_equal(offsets, b.offsets.numpy())
        x, y = a.adjacency.numpy(), b.adjacency.numpy()
        for start, end in zip(offsets[:-1], offsets[1:]):
            assert np.array_equal(np.sort(x[start:end]), np.sort(y[start:end]))
    if tetrahedra:
        na, nb = int(a.tetrahedra_count.numpy()[0]), int(b.tetrahedra_count.numpy()[0])
        assert na == nb
        assert np.array_equal(
            canonical(a.tetrahedra.numpy()[:na]), canonical(b.tetrahedra.numpy()[:nb])
        )


def normalized_trace(call):
    launches, allocations = trace(call)
    return [
        (name.removeprefix("warpower_regression_baseline.").removeprefix("warpower."), *rest)
        for name, *rest in launches
    ], allocations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--tet-repeats", type=int, default=10)
    parser.add_argument(
        "--quick", action="store_true", help="Smaller smoke cases, not release evidence"
    )
    args = parser.parse_args()
    if min(args.repeats, args.tet_repeats) < 3:
        parser.error("At least three paired repetitions are required")
    baseline = load_baseline(args.baseline_path.resolve())
    wp.set_mempool_release_threshold("cuda:0", 32 * 1024**3)
    large = 1000 if args.quick else 100000
    million = 2000 if args.quick else 1000000
    cases = [
        ("eager", large, "uniform", False, True, False),
        ("eager", large, "cluster5", False, True, False),
        ("eager", million, "uniform", False, True, False),
        ("eager", million, "uniform", True, True, False),
        ("graph", large, "uniform", False, True, False),
        ("graph", large, "uniform", True, True, False),
        ("graph", large, "uniform", False, False, True),
        ("graph", large, "uniform", True, False, True),
        ("graph", large, "uniform", False, True, True),
        ("graph", large, "uniform", True, True, True),
    ]
    records = []
    for mode, n, distribution, weighted, adjacency, tetrahedra in cases:
        if tetrahedra:
            rng = np.random.default_rng(2026)
            points = rng.uniform(-1, 1, (n, 3)).astype(np.float32)
            weights = rng.uniform(-0.0002, 0.0002, n).astype(np.float32)
        else:
            points, weights = generate(n, distribution, 42, weighted)
        p = wp.array(points, dtype=wp.vec3, device="cuda:0")
        w = wp.array(weights, device="cuda:0") if weighted else None
        workspaces, memory, calls = [], [], []
        for lib in (baseline, warpower):
            if mode == "graph":
                before_memory = wp.get_mempool_used_mem_current("cuda:0")
                workspace = lib.DiagramWorkspace(
                    p,
                    w,
                    adjacency=adjacency,
                    return_tetrahedra=tetrahedra,
                )
                memory.append(wp.get_mempool_used_mem_current("cuda:0") - before_memory)
                workspaces.append(workspace)
                calls.append(workspace.run)
            else:
                calls.append(
                    lambda lib=lib: lib.power_diagram(p, w) if weighted else lib.voronoi_diagram(p)
                )
        for _ in range(3):
            for call in calls:
                result = call()
                wp.synchronize()
                del result
        same_trace = None
        if mode == "eager":
            same_trace = normalized_trace(calls[0]) == normalized_trace(calls[1])
        times = [[], []]
        before = cpu_quota_snapshot()
        for repeat in range(args.tet_repeats if tetrahedra else args.repeats):
            for which in [0, 1] if repeat % 2 == 0 else [1, 0]:
                wp.synchronize()
                start = time.perf_counter()
                result = calls[which]()
                wp.synchronize()
                times[which].append(1000 * (time.perf_counter() - start))
                del result
        cpu = cpu_quota_delta(before, cpu_quota_snapshot())
        outputs = [call() for call in calls]
        wp.synchronize()
        check_outputs(*outputs, adjacency, tetrahedra, mode == "graph")
        relative = np.asarray(times[1]) / np.asarray(times[0]) - 1.0
        delta = (
            student_t.ppf(0.975, len(relative) - 1)
            * np.std(relative, ddof=1)
            / np.sqrt(len(relative))
        )
        interval = [float(np.mean(relative) - delta), float(np.mean(relative) + delta)]
        record = {
            "mode": mode,
            "n": n,
            "distribution": distribution,
            "weighted": weighted,
            "adjacency": adjacency,
            "tetrahedra": tetrahedra,
            "baseline_ms": times[0],
            "current_ms": times[1],
            "baseline_mean_ms": float(np.mean(times[0])),
            "current_mean_ms": float(np.mean(times[1])),
            "paired_relative_change_mean": float(np.mean(relative)),
            "paired_relative_change_95ci": interval,
            "regression_above_two_percent": interval[0] > 0.02,
            "identical_eager_launch_allocation_trace": same_trace,
            "persistent_workspace_bytes": memory,
            "outputs_equal": True,
            "cpu_throttled": any(v["throttled_periods"] for v in cpu.values()),
            "cpu_quota_delta": cpu,
        }
        records.append(record)
        print(
            json.dumps(
                {
                    k: v
                    for k, v in record.items()
                    if k
                    not in (
                        "baseline_ms",
                        "current_ms",
                        "cpu_quota_delta",
                    )
                }
            ),
            flush=True,
        )
        del outputs, calls, workspaces, call
        if mode == "graph":
            del workspace
        gc.collect()
    report = {
        "baseline_revision": subprocess.check_output(
            ["git", "-C", str(args.baseline_path), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "warmups": 3,
        "quick_smoke_only": args.quick,
        "comparison": "Alternating paired warmed calls, synchronized wall time, resident CUDA inputs",
        "gate": "Fail if any paired 95% interval lies entirely above +2%; inspect all changes",
        "source_sha256": {
            path.name: {
                "current": hashlib.sha256(path.read_bytes()).hexdigest(),
                "baseline": hashlib.sha256((args.baseline_path / path).read_bytes()).hexdigest()
                if (args.baseline_path / path).exists()
                else None,
            }
            for path in sorted(Path("warpower").glob("*.py"))
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if any(r["regression_above_two_percent"] or r["cpu_throttled"] for r in records):
        raise SystemExit("Investigate regression or CPU throttling before accepting this change")


if __name__ == "__main__":
    main()
