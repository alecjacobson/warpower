"""Paired eager versus reusable CUDA graph construction timings."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

import warpower
from benchmarks.datasets import generate
from benchmarks.run import cpu_quota_delta, cpu_quota_snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--tet-repeats", type=int, default=5)
    parser.add_argument("--only", choices=["all", "diagrams", "tetrahedra"], default="all")
    args = parser.parse_args()
    wp.set_mempool_release_threshold("cuda:0", 32 * 1024**3)
    records = []
    cases = [
        (100000, "uniform", False, False),
        (100000, "cluster5", False, False),
        (1000000, "uniform", False, False),
        (1000000, "uniform", True, False),
        (1000, "uniform", False, True),
        (10000, "uniform", False, True),
        (100000, "uniform", False, True),
        (100000, "uniform", True, True),
    ]
    for n, distribution, weighted, tets in cases:
        if args.only == "diagrams" and tets or args.only == "tetrahedra" and not tets:
            continue
        if tets:
            rng = np.random.default_rng(2026)
            points = rng.uniform(-1, 1, (n, 3)).astype(np.float32)
            bound = 0.0002 if n >= 100000 else 0.002
            weights = rng.uniform(-bound, bound, n).astype(np.float32) if weighted else None
        else:
            points, weights = generate(n, distribution, 42, weighted)
        p = wp.array(points, dtype=wp.vec3, device="cuda:0")
        w = wp.array(weights, device="cuda:0") if weighted else None
        workspace = warpower.DiagramWorkspace(p, w, return_tetrahedra=tets)

        def eager():
            if weighted:
                return warpower.power_diagram(p, w, return_tetrahedra=tets)
            return warpower.voronoi_diagram(p, return_tetrahedra=tets)

        calls = [eager, workspace.run]
        for _ in range(3):
            for call in calls:
                result = call()
                wp.synchronize()
                del result
        assert workspace.error.numpy()[0] == 0
        times = [[], []]
        before = cpu_quota_snapshot()
        for repeat in range(args.tet_repeats if tets else args.repeats):
            for which in [0, 1] if repeat % 2 == 0 else [1, 0]:
                wp.synchronize()
                start = time.perf_counter()
                result = calls[which]()
                wp.synchronize()
                times[which].append(1000 * (time.perf_counter() - start))
                del result
        cpu = cpu_quota_delta(before, cpu_quota_snapshot())
        assert workspace.error.numpy()[0] == 0
        expected = eager()
        offsets = workspace.offsets.numpy()
        assert np.array_equal(offsets, expected.offsets.numpy())
        actual = workspace.adjacency.numpy()
        reference = expected.adjacency.numpy()
        for i in range(n):
            start, end = offsets[i : i + 2]
            assert np.array_equal(np.sort(actual[start:end]), np.sort(reference[start:end]))
        if tets:
            from benchmarks.tetrahedra import canonical

            count = int(workspace.tetrahedra_count.numpy()[0])
            assert np.array_equal(
                canonical(workspace.tetrahedra.numpy()[:count]),
                canonical(expected.tetrahedra.numpy()),
            )
        record = {
            "n": n,
            "distribution": distribution,
            "weighted": weighted,
            "tetrahedra": tets,
            "eager_ms": times[0],
            "graph_ms": times[1],
            "eager_mean_ms": float(np.mean(times[0])),
            "graph_mean_ms": float(np.mean(times[1])),
            "speedup": float(np.mean(times[0]) / np.mean(times[1])),
            "output_matches_eager": True,
            "cpu_quota_delta": cpu,
            "cpu_throttled": any(v["throttled_periods"] for v in cpu.values()),
        }
        if tets:
            record["tetrahedra_count"] = int(workspace.tetrahedra_count.numpy()[0])
            record["warm_replay_expansions"] = int(workspace.tetrahedra_expansions.numpy()[0])
        records.append(record)
        print(
            json.dumps(
                {
                    k: v
                    for k, v in record.items()
                    if k not in ("eager_ms", "graph_ms", "cpu_quota_delta")
                }
            ),
            flush=True,
        )
        del expected, workspace, calls
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "warmups": 3,
                "mempool_retained_gib": 32,
                "source_sha256": {
                    str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(Path("warpower").glob("*.py"))
                },
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
