"""Measure fixed-hull ODT; this is not yet a full domain-mesher benchmark."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import warp as wp
from scipy.spatial import ConvexHull

import warpower
from benchmarks.run import cpu_quota_delta, cpu_quota_snapshot
from benchmarks.tetrahedra import canonical


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000])
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reference", action="store_true", help="Compare final connectivity with SciPy"
    )
    parser.add_argument(
        "--save-dir", type=Path, help="Save final sites/connectivity outside timing"
    )
    args = parser.parse_args()
    wp.set_mempool_release_threshold("cuda:0", 32 * 1024**3)
    records = []
    for n in args.sizes:
        source = np.random.default_rng(2026).uniform(-1, 1, (n, 3)).astype(np.float32)
        original = wp.array(source, dtype=wp.vec3, device="cuda:0")
        points = wp.clone(original)
        wp.synchronize()
        start = time.perf_counter()
        workspace = warpower.ODTWorkspace(points, max_iterations=args.iterations)
        with wp.ScopedCapture() as capture:
            wp.copy(points, original)
            workspace.run()
        wp.synchronize()
        setup = time.perf_counter() - start
        wp.capture_launch(capture.graph)
        wp.synchronize()
        initial = workspace.energy_history.numpy()[0]
        times, outcomes = [], []
        before = cpu_quota_snapshot()
        for _ in range(args.repeats):
            start = time.perf_counter()
            wp.capture_launch(capture.graph)
            wp.synchronize()
            times.append(time.perf_counter() - start)
            outcomes.append(
                {
                    "accepted_iterations": int(workspace.iterations.numpy()[0]),
                    "error": int(workspace.error.numpy()[0]),
                    "termination": int(workspace.termination.numpy()[0]),
                    "rejected_steps": int(workspace.rejected_steps.numpy()[0]),
                }
            )
        cpu = cpu_quota_delta(before, cpu_quota_snapshot())
        assert workspace.error.numpy()[0] == 0
        actual = points.numpy()
        hull = ConvexHull(source)
        assert np.array_equal(actual[hull.vertices], source[hull.vertices])
        assert np.isclose(workspace.volume.numpy()[0], hull.volume, rtol=1e-10)
        steps = int(workspace.iterations.numpy()[0])
        history = workspace.energy_history.numpy()[: steps + 1]
        assert np.all(np.diff(history) <= initial * 2e-12)
        record = {
            "n": n,
            "tetrahedra": int(workspace.tetrahedra_count.numpy()[0]),
            "setup_seconds": setup,
            "warm_replay_seconds": times,
            "replay_outcomes": outcomes,
            "mean_replay_seconds": float(np.mean(times)),
            "requested_iterations": args.iterations,
            "accepted_iterations": steps,
            "rejected_steps": int(workspace.rejected_steps.numpy()[0]),
            "termination": int(workspace.termination.numpy()[0]),
            "last_geometry_error": int(workspace.last_geometry_error.numpy()[0]),
            "energy_history": history.tolist(),
            "energy_reduction_fraction": float(1 - history[-1] / initial),
            "minimum_radius_ratio": float(workspace.minimum_radius_ratio.numpy()[0]),
            "boundary_unchanged": True,
            "volume_matches_hull": True,
            "cpu_throttled": any(v["throttled_periods"] for v in cpu.values()),
            "cpu_quota_delta": cpu,
        }
        if args.reference or args.save_dir:
            count = int(workspace.tetrahedra_count.numpy()[0])
            tets = workspace.tetrahedra.numpy()[:count]
            if args.reference:
                from scipy.spatial import Delaunay

                matches = np.array_equal(canonical(tets), canonical(Delaunay(actual).simplices))
                record["complete_tet_set_matches_scipy"] = bool(matches)
                assert matches, "Optimized mesh differs from the independent reference"
            if args.save_dir:
                args.save_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    args.save_dir / f"optimized-{n}.npz", points=actual, tetrahedra=tets
                )
        records.append(record)
        print(
            json.dumps(
                {k: v for k, v in record.items() if k not in ("energy_history", "cpu_quota_delta")}
            ),
            flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "scope": "Uniform fixed-hull ODT, seed 2026, float32 sites / float64 arithmetic; not full domain meshing",
                "source_sha256": {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in Path("warpower").glob("*.py")
                },
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
