"""Run in separate processes for clean comparisons. See --help."""

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from benchmarks.datasets import generate


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
    args = parser.parse_args()
    if args.input:
        data = np.load(args.input)
        points = data["points"].astype(np.float32)
        weights = data["weights"].astype(np.float32) if args.weighted else None
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
        p = wp.array(points, dtype=wp.vec3, device="cuda:0")
        w = wp.array(weights, device="cuda:0") if weights is not None else None
        synchronize = wp.synchronize
        as_numpy = lambda a: a.numpy()
        versions = {
            "warp": wp.__version__,
            "warpower_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
        }

    def build():
        return lib.voronoi_diagram(p) if w is None else lib.power_diagram(p, w)

    for _ in range(args.warmup):
        result = build()
        synchronize()
        del result
    times = []
    for _ in range(args.repeats):
        synchronize()
        start = time.perf_counter()
        result = build()
        synchronize()
        times.append(time.perf_counter() - start)
        if len(times) < args.repeats:
            del result
    adjacency, offsets, status = (
        as_numpy(getattr(result, name)) for name in ("adjacency", "offsets", "status")
    )
    unique, counts = np.unique(status, return_counts=True)
    record = {
        "backend": args.backend,
        "distribution": args.distribution,
        "n": len(points),
        "seed": args.seed,
        "weighted": args.weighted,
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
