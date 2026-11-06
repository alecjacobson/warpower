"""Paired resident-device timing and independent hull/connectivity checks.

Host validation and setup are outside timing. Requires the development extra.
The small case verifies that the automatic cutoff preserves the original path.
"""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import warp as wp

from benchmarks.run import cpu_quota_delta, cpu_quota_snapshot
from tests.test_tetrahedra import check_mesh, tet_set
from warpower import DelaunayWorkspace


def main():
    parser = argparse.ArgumentParser(
        description="Compare original and hybrid certified Delaunay on four distributions"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    rng = np.random.default_rng(8721)
    records = []
    for case, n in [
        ("uniform8192", 8192),
        ("uniform32768", 32768),
        ("clustered32768", 32768),
        ("anisotropic32768", 32768),
    ]:
        p = rng.uniform(-1, 1, (n, 3)).astype(np.float32)
        if case.startswith("clustered"):
            centers = np.array(
                [[0, 0, 0], [0.7, 0, 0], [-0.7, 0, 0], [0, 0.7, 0], [0, 0, 0.7]], np.float32
            )
            p = (centers[np.arange(n) % 5] + rng.normal(0, 0.03, (n, 3))).astype(np.float32)
        if case.startswith("anisotropic"):
            p[:, 2] *= np.float32(0.05)
        comparisons = []
        connectivity = []
        for enabled in (False, True):
            w = DelaunayWorkspace(p, fast_candidates=enabled)
            begin, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
            initial_quota = cpu_quota_snapshot()
            wp.record_event(begin)
            w.run()
            wp.record_event(end)
            wp.synchronize_event(end)
            first = wp.get_event_elapsed_time(begin, end)
            initial_cpu = cpu_quota_delta(initial_quota, cpu_quota_snapshot())
            assert w.error.numpy()[0] == 0, (
                case,
                enabled,
                w.base_error.numpy(),
                w.symbolic_error.numpy(),
            )
            first_diagnostics = dict(
                fast_attempted=int(w.fast_attempted.numpy()[0]),
                fast_error=int(w.fast_error.numpy()[0]),
                auxiliary_attempts=int(w.attempts.numpy()[0]),
                symbolic_recovered=int(w.symbolic_recovered.numpy()[0]),
            )
            w.run()
            wp.synchronize()
            quota = cpu_quota_snapshot()
            samples = []
            for _ in range(args.repeats):
                wp.record_event(begin)
                w.run()
                wp.record_event(end)
                wp.synchronize_event(end)
                samples.append(wp.get_event_elapsed_time(begin, end))
            cpu = cpu_quota_delta(quota, cpu_quota_snapshot())
            assert not any(r["throttled_periods"] for r in cpu.values())
            count = int(w.tetrahedra_count.numpy()[0])
            t = w.tetrahedra.numpy()[:count]
            check_mesh(p, t)
            connectivity.append(tet_set(t))
            comparisons.append(
                dict(
                    fast_enabled=enabled,
                    first_ms=first,
                    initial_cpu_quota_delta=initial_cpu,
                    first_diagnostics=first_diagnostics,
                    median_ms=float(np.median(samples)),
                    samples_ms=samples,
                    cpu=cpu,
                    count=count,
                    error=int(w.error.numpy()[0]),
                    fast_attempted=int(w.fast_attempted.numpy()[0]),
                    fast_accepted=int(w.fast_accepted.numpy()[0]),
                    fast_error=int(w.fast_error.numpy()[0]),
                    fast_repaired_cells=int(w.fast_repaired_cells.numpy()[0]),
                    auxiliary_attempts=int(w.attempts.numpy()[0]),
                    symbolic_recovered=int(w.symbolic_recovered.numpy()[0]),
                )
            )
            del w
            gc.collect()
        assert connectivity[0] == connectivity[1], case
        row = dict(case=case, n=n, seed=8721, canonical_cells_equal=True, comparisons=comparisons)
        records.append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                dict(
                    scope="Warmed resident CUDA-event replays, current exact certificate and independent convex-hull/topology checks, original/hybrid canonical cell equivalence. Setup and host validation excluded.",
                    records=records,
                ),
                indent=2,
            )
            + "\n"
        )
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
