"""Paired current-cell continuation experiment; no public path is changed."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import warp as wp
from scipy.spatial import Delaunay

from benchmarks._cell_continuation import install, original_schedule
from benchmarks.run import cpu_quota_delta, cpu_quota_snapshot
from benchmarks.tetrahedra import canonical
from warpower import DelaunayWorkspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--slots", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("Use at least three paired samples")
    if args.output.exists():
        parser.error("Output already exists")
    points = (
        np.asarray(np.load(args.input)["points"], np.float32)
        if args.input
        else np.random.default_rng(2026).uniform(-1, 1, (args.n, 3)).astype(np.float32)
    )
    expected = canonical(Delaunay(points.astype(float)).simplices)
    start = time.perf_counter()
    mesh = DelaunayWorkspace(points, cache_strategy=False)
    original_schedule(mesh)
    baseline = mesh.graph
    setup = time.perf_counter() - start
    start = time.perf_counter()
    cache = install(mesh, slots=args.slots)
    setup_continuation = time.perf_counter() - start
    graphs = {"baseline": baseline, "continuation": mesh.graph}
    events = [wp.Event(enable_timing=True), wp.Event(enable_timing=True)]
    records = {name: dict(samples_ms=[], cpu=[], statistics=[], base_errors=[]) for name in graphs}
    for name, graph in graphs.items():
        wp.capture_launch(graph)
        wp.synchronize()
        assert mesh.error.numpy()[0] == 0
        assert np.array_equal(
            expected, canonical(mesh.tetrahedra.numpy()[: mesh.tetrahedra_count.numpy()[0]])
        )
    for iteration in range(args.repeats):
        order = list(graphs) if iteration % 2 == 0 else list(reversed(graphs))
        for name in order:
            before = cpu_quota_snapshot()
            wp.record_event(events[0])
            wp.capture_launch(graphs[name])
            wp.record_event(events[1])
            wp.synchronize_event(events[1])
            row = records[name]
            row["samples_ms"].append(wp.get_event_elapsed_time(*events))
            row["cpu"].append(cpu_quota_delta(before, cpu_quota_snapshot()))
            row["statistics"].append(
                cache.data.statistics.numpy().tolist() if name == "continuation" else None
            )
            row["base_errors"].append(int(mesh.base_error.numpy()[0]))
            assert mesh.error.numpy()[0] == 0
            assert np.array_equal(
                expected, canonical(mesh.tetrahedra.numpy()[: mesh.tetrahedra_count.numpy()[0]])
            )
    for row in records.values():
        row["median_ms"] = float(np.median(row["samples_ms"]))
    report = dict(
        scope="Alternating preallocated complete certified construction; same scratch/output arrays. Excludes setup/upload/independent readback. Every measured output matches independent SciPy; no complete VTM claim.",
        n=len(points),
        device=wp.get_device().name,
        warp=wp.__version__,
        input_sha256=hashlib.sha256(points.tobytes()).hexdigest(),
        runtime_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path("warpower").glob("*.py"))
        },
        experiment_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), Path(__file__).with_name("_cell_continuation.py")]
        },
        baseline_setup_seconds=setup,
        continuation_setup_seconds=setup_continuation,
        added_bytes=cache.bytes,
        records=records,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
