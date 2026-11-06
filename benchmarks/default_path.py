"""Paired default-call regression check against a pre-tetrahedra API revision."""

import argparse
import ast
import hashlib
import json
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import warp as wp

import warpower
from benchmarks.datasets import generate
from benchmarks.run import cpu_quota_delta, cpu_quota_snapshot


def trace(call):
    launches, allocations = [], []
    originals = {name: getattr(wp, name) for name in ("launch", "empty", "zeros", "full")}

    def launch(kernel, dim, *args, **kwargs):
        launches.append((kernel.module.name, kernel.key, dim, kwargs.get("block_dim")))
        return originals["launch"](kernel, dim, *args, **kwargs)

    def allocator(name):
        def wrapped(*args, **kwargs):
            shape = args[0] if args else kwargs.get("shape")
            allocations.append((name, shape, str(kwargs.get("dtype"))))
            return originals[name](*args, **kwargs)

        return wrapped

    try:
        wp.launch = launch
        for name in ("empty", "zeros", "full"):
            setattr(wp, name, allocator(name))
        result = call()
        wp.synchronize()
        del result
    finally:
        for name, original in originals.items():
            setattr(wp, name, original)
    return launches, allocations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-ref", default="72c679509bc7d3f6d3468ab12aa8dc0425dd7513")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = subprocess.check_output(
        ["git", "show", f"{args.baseline_ref}:warpower/_api.py"], text=True
    )
    current = Path("warpower/_api.py").read_text()

    def construction_ast(text):
        return ast.dump(
            next(
                n
                for n in ast.parse(text).body
                if isinstance(n, ast.FunctionDef) and n.name == "_build"
            )
        )

    assert construction_ast(source) == construction_ast(current), "Default _build changed"
    kernel_hashes = {}
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", args.baseline_ref, "warpower"], text=True
    ).splitlines()
    for path in paths:
        if path.endswith(".py") and path not in ("warpower/_api.py", "warpower/__init__.py"):
            old = subprocess.check_output(["git", "show", f"{args.baseline_ref}:{path}"])
            assert old == Path(path).read_bytes(), path
            kernel_hashes[path] = hashlib.sha256(old).hexdigest()
    baseline = types.ModuleType("warpower._default_baseline")
    baseline.__package__ = "warpower"
    sys.modules[baseline.__name__] = baseline
    exec(compile(source, "<baseline-api>", "exec"), baseline.__dict__)
    assert "warpower._tetrahedra" not in sys.modules
    assert "warpower._graph" not in sys.modules
    wp.set_mempool_release_threshold("cuda:0", 32 * 1024**3)
    records = []
    for distribution, n, weighted in [
        ("uniform", 100000, False),
        ("cluster5", 100000, False),
        ("uniform", 1000000, False),
        ("uniform", 1000000, True),
    ]:
        points, weights = generate(n, distribution, 42, weighted)
        p = wp.array(points, dtype=wp.vec3, device="cuda:0")
        w = wp.array(weights, device="cuda:0") if weighted else None
        calls = [
            (lambda lib=lib: lib.power_diagram(p, w) if weighted else lib.voronoi_diagram(p))
            for lib in (baseline, warpower)
        ]
        for _ in range(5):
            for call in calls:
                result = call()
                wp.synchronize()
                del result
        old_trace, new_trace = [trace(call) for call in calls]
        assert old_trace == new_trace, "GPU launch/allocation trace differs"
        times = [[], []]
        before = cpu_quota_snapshot()
        for k in range(args.repeats):
            for index in [0, 1] if k % 2 == 0 else [1, 0]:
                wp.synchronize()
                start = time.perf_counter()
                result = calls[index]()
                wp.synchronize()
                times[index].append(1000 * (time.perf_counter() - start))
                del result
        cpu = cpu_quota_delta(before, cpu_quota_snapshot())
        record = {
            "distribution": distribution,
            "n": n,
            "weighted": weighted,
            "baseline_mean_ms": float(np.mean(times[0])),
            "current_mean_ms": float(np.mean(times[1])),
            "baseline_median_ms": float(np.median(times[0])),
            "current_median_ms": float(np.median(times[1])),
            "baseline_times_ms": times[0],
            "current_times_ms": times[1],
            "identical_gpu_launches": len(old_trace[0]),
            "identical_allocation_calls": len(old_trace[1]),
            "cpu_throttled": any(v["throttled_periods"] for v in cpu.values()),
            "cpu_quota_delta": cpu,
        }
        records.append(record)
        print(
            json.dumps(
                {
                    k: v
                    for k, v in record.items()
                    if not k.endswith("times_ms") and k != "cpu_quota_delta"
                }
            ),
            flush=True,
        )
    assert "warpower._tetrahedra" not in sys.modules
    assert "warpower._graph" not in sys.modules
    report = {
        "baseline_ref": args.baseline_ref,
        "warmups_per_implementation": 5,
        "repeats_per_implementation": args.repeats,
        "mempool_retained_gib": 32,
        "identical_default_construction_ast": True,
        "unchanged_source_sha256": kernel_hashes,
        "optional_module_loaded": False,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
