"""Measure and optionally trace captured surface refits and signed-distance queries."""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

import warpower
from benchmarks.run import cpu_quota_delta, cpu_quota_snapshot
from tests.test_domain import consumer, torus


@wp.kernel
def deform(source: wp.array[wp.vec3], output: wp.array[wp.vec3], phase: wp.array[int]):
    i = wp.tid()
    output[i] = source[i] * (1.0 + 0.01 * float(phase[0] % 7))


@wp.kernel
def advance(phase: wp.array[int]):
    phase[0] += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=100000)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dot", type=Path)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.queries < 1 or args.repeats < 1:
        parser.error("queries and repeats must be positive")
    p, f = torus(96, 48)
    source = wp.array(p, dtype=wp.vec3, device="cuda:0")
    points = wp.clone(source)
    domain = warpower.SurfaceDomain(points, f, surface_subdivisions=2)
    queries = np.random.default_rng(81).uniform(-3, 3, (args.queries, 3)).astype(np.float32)
    query = warpower.DomainQueryWorkspace(domain, queries)
    phase = wp.zeros(1, dtype=int, device="cuda:0")
    result = wp.empty(args.queries, dtype=float, device="cuda:0")
    wp.load_module(module=__name__, device="cuda:0")
    wp.load_module(module="tests.test_domain", device="cuda:0")
    graphs = {}
    with wp.ScopedCapture() as capture:
        query.run()
    graphs["queries"] = capture.graph
    with wp.ScopedCapture() as capture:
        wp.launch(deform, len(p), inputs=[source, points, phase])
        domain.refit()
        query.run()
        wp.launch(consumer, args.queries, inputs=[query.signed_distance, query.inside, result])
        wp.launch(advance, 1, inputs=[phase])
    graphs["refit_queries_consumer"] = capture.graph
    report = {
        "device": wp.get_device("cuda:0").name,
        "warp": wp.__version__,
        "vertices": len(p),
        "triangles": len(f),
        "queries": args.queries,
        "surface_samples": len(domain.surface_points),
        "timings_ms": {},
    }
    for name, graph in graphs.items():
        for _ in range(3):
            wp.capture_launch(graph)
        wp.synchronize()
        start, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
        cpu_before = cpu_quota_snapshot()
        values = []
        for _ in range(args.repeats):
            wp.record_event(start)
            wp.capture_launch(graph)
            wp.record_event(end)
            wp.synchronize_event(end)
            values.append(wp.get_event_elapsed_time(start, end))
        report["timings_ms"][name] = {"samples": values, "median": float(np.median(values))}
        cpu = cpu_quota_delta(cpu_before, cpu_quota_snapshot())
        report["timings_ms"][name]["cpu_quota_delta"] = cpu
        report["timings_ms"][name]["cpu_throttled"] = any(
            v["throttled_periods"] for v in cpu.values()
        )
        assert domain.error.numpy()[0] == 0
        assert query.error.numpy()[0] == 0
        assert np.all(query.valid.numpy())
    if args.dot:
        args.dot.parent.mkdir(parents=True, exist_ok=True)
        wp.capture_debug_dot_print(graphs["refit_queries_consumer"], str(args.dot), verbose=True)
    if args.profile:
        import torch

        torch.cuda.cudart().cudaProfilerStart()
        for _ in range(3):
            wp.capture_launch(graphs["refit_queries_consumer"])
        wp.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    report["domain_error"] = int(domain.error.numpy()[0])
    report["query_error"] = int(query.error.numpy()[0])
    report["all_queries_valid"] = bool(np.all(query.valid.numpy()))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
