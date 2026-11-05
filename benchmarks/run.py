"""Run in separate processes for clean comparisons. See --help."""

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from benchmarks.datasets import generate, generate_weights
from benchmarks.memory import MemorySampler


def cpu_quota_snapshot():
    """Read visible cgroup-v2 CPU limits, including limiting ancestors."""
    records = {}
    try:
        line = next(
            line
            for line in Path("/proc/self/cgroup").read_text().splitlines()
            if line.startswith("0::")
        )
        current = Path("/sys/fs/cgroup") / line[3:].lstrip("/")
        root = Path("/sys/fs/cgroup")
        while current == root or root in current.parents:
            maximum = current / "cpu.max"
            if maximum.exists():
                quota, period = maximum.read_text().split()
                if quota != "max":
                    stats = dict(
                        line.split() for line in (current / "cpu.stat").read_text().splitlines()
                    )
                    # Hierarchy depth identifies the limit without publishing
                    # container identifiers from the host's cgroup paths.
                    depth = len(current.relative_to(root).parts)
                    records[str(depth)] = {
                        "cpu_cores": int(quota) / int(period),
                        "throttled_periods": int(stats["nr_throttled"]),
                        "throttled_usec": int(stats["throttled_usec"]),
                    }
            current = current.parent
    except (OSError, ValueError, KeyError, StopIteration):
        pass
    return records


def cpu_quota_delta(before, after):
    return {
        level: {
            "cpu_cores": record["cpu_cores"],
            "throttled_periods": record["throttled_periods"] - before[level]["throttled_periods"],
            "throttled_usec": record["throttled_usec"] - before[level]["throttled_usec"],
        }
        for level, record in after.items()
        if level in before
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["paragram", "warpower"], required=True)
    parser.add_argument("--distribution", default="uniform")
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weighted", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--input", type=Path, help="NPZ with points and optional weights")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-diagram", type=Path)
    parser.add_argument(
        "--mempool-retain-mb",
        type=int,
        default=0,
        help="Warp allocator reserve policy (no geometry preallocation)",
    )
    args = parser.parse_args()
    if args.input:
        data = np.load(args.input)
        points = data["points"].astype(np.float32)
        weights = None
        if args.weighted:
            weights = (
                data["weights"].astype(np.float32)
                if "weights" in data
                else generate_weights(points, args.seed)
            )
    else:
        points, weights = generate(args.n, args.distribution, args.seed, args.weighted)

    if args.backend == "paragram":
        import paragram as lib
        import torch

        p = torch.from_numpy(points).cuda()
        w = torch.from_numpy(weights).cuda() if weights is not None else None
        synchronize = torch.cuda.synchronize
        as_numpy = lambda a: a.cpu().numpy()
        versions = {
            "torch": torch.__version__,
            "paragram_commit": "4f36b1f2c8738912c5efc056a06023d4c1d2da07",
        }
    else:
        import warp as wp

        import warpower as lib

        wp.init()
        wp.set_mempool_release_threshold("cuda:0", args.mempool_retain_mb * 1024**2)
        p = wp.array(points, dtype=wp.vec3, device="cuda:0")
        w = wp.array(weights, device="cuda:0") if weights is not None else None
        synchronize = wp.synchronize
        as_numpy = lambda a: a.numpy()
        versions = {
            "warp": wp.__version__,
            "warpower_source_sha256": hashlib.sha256(
                b"".join(p.read_bytes() for p in sorted(Path("warpower").glob("*.py")))
            ).hexdigest(),
            "warpower_dirty": bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain", "--", "warpower"], text=True
                ).strip()
            ),
            "warpower_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
        }

    def build():
        return lib.voronoi_diagram(p) if w is None else lib.power_diagram(p, w)

    versions["benchmark_source_sha256"] = hashlib.sha256(
        b"".join(
            (Path("benchmarks") / name).read_bytes()
            for name in ("datasets.py", "memory.py", "run.py")
        )
    ).hexdigest()
    memory = MemorySampler()
    memory.start()
    for _ in range(args.warmup):
        result = build()
        synchronize()
        del result
    times = []
    cpu_before = cpu_quota_snapshot()
    for _ in range(args.repeats):
        synchronize()
        start = time.perf_counter()
        result = build()
        synchronize()
        times.append(time.perf_counter() - start)
        if len(times) < args.repeats:
            del result
    cpu_limits = cpu_quota_delta(cpu_before, cpu_quota_snapshot())
    memory_record = memory.finish()
    adjacency, offsets, status = (
        as_numpy(getattr(result, name)) for name in ("adjacency", "offsets", "status")
    )
    unique, counts = np.unique(status, return_counts=True)
    record = {
        "backend": args.backend,
        "points_sha256": hashlib.sha256(points.tobytes()).hexdigest(),
        "weights_sha256": hashlib.sha256(weights.tobytes()).hexdigest()
        if weights is not None
        else None,
        "distribution": args.distribution,
        "n": len(points),
        "seed": args.seed,
        "weighted": args.weighted,
        "memory": memory_record,
        "cpu_quota_during_measurement": cpu_limits,
        "cpu_throttled": any(r["throttled_periods"] > 0 for r in cpu_limits.values()),
        "mempool_retain_mb": args.mempool_retain_mb if args.backend == "warpower" else None,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "seconds": times,
        "mean_seconds": float(np.mean(times)),
        "median_seconds": float(np.median(times)),
        "nnz": len(adjacency),
        "status_counts": dict(zip(map(str, unique), map(int, counts))),
        "timing": "synchronized wall time; construction allocations included; GPU inputs resident; compilation excluded",
        "gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True,
        ).strip(),
        "python": platform.python_version(),
        "versions": versions,
    }
    if hasattr(result, "diagnostics"):
        record["diagnostics"] = result.diagnostics
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    if args.save_diagram:
        np.savez_compressed(args.save_diagram, adjacency=adjacency, offsets=offsets, status=status)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
