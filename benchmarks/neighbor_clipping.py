"""Isolated full-hull clipping reuse experiment (#272); public defaults unchanged."""

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import warp as wp
from scipy.spatial import Delaunay

from benchmarks._neighbor_clipping import install
from benchmarks.run import cpu_quota_delta, cpu_quota_snapshot
from benchmarks.tetrahedra import canonical
from warpower import DelaunayWorkspace
from warpower._wall_reuse import WallReuseDelaunayWorkspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--strategies", nargs="+", default=["fresh", "larger", "neighbors", "fragment"]
    )
    parser.add_argument(
        "--movement", type=float, default=0.0, help="Displacement as fraction of bbox diagonal"
    )
    args = parser.parse_args()
    rng = np.random.default_rng(2026)
    if args.input:
        points = np.asarray(np.load(args.input)["points"], np.float32)
    else:
        points = rng.uniform(-1, 1, (args.n, 3)).astype(np.float32)
    moved = points.copy()
    if args.movement:
        moved += rng.normal(size=points.shape).astype(np.float32) * np.float32(
            args.movement * np.linalg.norm(np.ptp(points, axis=0))
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(
        n=len(points),
        movement=args.movement,
        input=str(args.input),
        input_sha256=hashlib.sha256(points.tobytes()).hexdigest(),
        moved_sha256=hashlib.sha256(moved.tobytes()).hexdigest(),
        source_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                Path(__file__).with_name("_neighbor_clipping.py"),
                *sorted(Path("warpower").glob("*.py")),
            ]
        },
        scope="Preallocated full certified construction, with cache ordering/build/update included. Setup, uploads, readback and independent SciPy comparison excluded. Each measured trial starts with the same previous point set and cache, then constructs the same moved input. Not a complete VTM solve.",
        records=[],
    )
    expected = canonical(Delaunay(moved.astype(np.float64)).simplices)
    before = wp.Event(enable_timing=True)
    after = wp.Event(enable_timing=True)
    for strategy in args.strategies:
        assert strategy in ("fresh", "larger", "neighbors", "neighbors-larger", "fragment")
        start = time.perf_counter()
        cls = WallReuseDelaunayWorkspace if strategy == "fragment" else DelaunayWorkspace
        mesh = cls(points)
        cache = None
        if strategy.startswith("neighbors"):
            cache = install(mesh, starting_capacity=256 if strategy.endswith("larger") else 128)
        elif strategy == "larger":
            mesh._base._tet_capacities = (256, 512, 1024)
            with wp.ScopedCapture() as capture:
                mesh._record()
            mesh.graph = capture.graph
        setup = time.perf_counter() - start
        original = wp.array(points, dtype=wp.vec3, device=mesh.device)
        current = wp.array(moved, dtype=wp.vec3, device=mesh.device)
        samples = []
        quota = cpu_quota_snapshot()
        for iteration in range(args.repeats + 1):
            # Reestablish the same previous input for each measured reconstruction.
            wp.copy(mesh.points, original)
            if cache is not None:
                cache.clear()
            if strategy == "fragment":
                mesh.clear_reuse()
            mesh._preferred.zero_()
            mesh.run()
            wp.copy(mesh.points, current)
            wp.record_event(before)
            mesh.run()
            wp.record_event(after)
            wp.synchronize_event(after)
            duration = wp.get_event_elapsed_time(before, after)
            if iteration:
                samples.append(duration)
        cpu = cpu_quota_delta(quota, cpu_quota_snapshot())
        count = int(mesh.tetrahedra_count.numpy()[0])
        output = canonical(mesh.tetrahedra.numpy()[:count])
        stats = {
            name: getattr(mesh, name).numpy().tolist()
            for name in (
                "error",
                "base_error",
                "certificate_error",
                "attempts",
                "recovered",
                "base_attempted",
                "symbolic_attempted",
                "symbolic_recovered",
            )
        }
        if strategy == "fragment":
            stats.update(
                {
                    name: getattr(mesh, name).numpy().tolist()
                    for name in (
                        "reuse_attempted",
                        "reuse_accepted",
                        "reuse_error",
                        "reuse_seeds",
                        "reuse_rounds",
                    )
                }
            )
        if cache:
            stats.update(
                cache_bytes=cache.bytes,
                truncated_edge_visits=cache.overflows.numpy().tolist(),
                neighbor_counts_percentiles=np.percentile(
                    cache.counts.numpy(), [0, 50, 95, 100]
                ).tolist(),
            )
        row = dict(
            strategy=strategy,
            setup_seconds=setup,
            samples_ms=samples,
            median_ms=float(np.median(samples)),
            tetrahedra=count,
            scipy_equal=bool(np.array_equal(expected, output)),
            cpu=cpu,
            stats=stats,
        )
        report["records"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)
        del mesh, original, current, cache
        gc.collect()
        wp.synchronize()


if __name__ == "__main__":
    main()
