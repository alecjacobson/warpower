"""Resident sampled sizing-field rebuild and query timings, with a CPU oracle."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

import warpower
from benchmarks.run import cpu_quota_delta, cpu_quota_snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--queries", type=int, default=100000)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.samples, args.queries, args.repeats) < 1:
        parser.error("Counts and repetitions must be positive")
    rng = np.random.default_rng(args.seed)
    samples = rng.uniform(-1, 1, (args.samples, 3)).astype(np.float32)
    values = rng.uniform(0.1, 1, args.samples).astype(np.float32)
    queries = rng.uniform(-1.2, 1.2, (args.queries, 3)).astype(np.float32)
    start = time.perf_counter()
    field = warpower.SizingField(samples, wp.array(values, device="cuda:0"), gradation=0.5)
    query = warpower.SizingQueryWorkspace(field, queries)
    with wp.ScopedCapture() as capture:
        field.refit()
        query.run()
    with wp.ScopedCapture() as query_capture:
        query.run()
    wp.synchronize()
    setup = time.perf_counter() - start
    for _ in range(3):
        wp.capture_launch(capture.graph)
    wp.synchronize()
    begin, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
    timings = {}
    cpu_before = cpu_quota_snapshot()
    for name, graph in (("refit_and_queries", capture.graph), ("queries", query_capture.graph)):
        timings[name] = []
        for _ in range(args.repeats):
            wp.record_event(begin)
            wp.capture_launch(graph)
            wp.record_event(end)
            wp.synchronize_event(end)
            timings[name].append(wp.get_event_elapsed_time(begin, end))
    cpu = cpu_quota_delta(cpu_before, cpu_quota_snapshot())
    assert field.error.numpy()[0] == query.error.numpy()[0] == 0
    check_n = min(64, args.queries)
    distance = np.linalg.norm(
        queries[:check_n].astype(float)[:, None] - samples.astype(float)[None], axis=2
    )
    reference = np.min(values.astype(float)[None] + 0.5 * distance, axis=1)
    actual = query.values.numpy()[:check_n]
    np.testing.assert_allclose(actual, reference, rtol=8e-8)
    result = dict(
        device=field.device.name,
        warp=wp.__version__,
        seed=args.seed,
        samples=args.samples,
        queries=args.queries,
        gradation=0.5,
        setup_seconds=setup,
        gpu_milliseconds=timings,
        median_gpu_milliseconds={key: float(np.median(value)) for key, value in timings.items()},
        oracle_queries=check_n,
        maximum_relative_oracle_error=float(np.max(np.abs(actual - reference) / reference)),
        cpu_quota_delta=cpu,
        cpu_throttled=any(v["throttled_periods"] for v in cpu.values()),
        scope="Resident float32 sample positions/values and query positions; sampled cone envelope evaluated in float64, rounded float32 output; no initialization, optimization or extraction",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
