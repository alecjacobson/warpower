"""Time a bounded tangential quality search from a fixed fitted torus."""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

import warpower
from benchmarks.run import cpu_quota_delta, cpu_quota_snapshot
from benchmarks.selective import metrics
from tests.test_domain import torus
from tests.test_extraction import independent_topology
from tests.test_tetrahedra import check_mesh, reference, tet_set
from tests.test_vtm_core import torus_sites
from warpower._jitter import BoundaryJitterWorkspace
from warpower._vtm import VTMWorkspace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--warmstart-iterations", type=int, default=30)
    parser.add_argument("--attempts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--dot", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.n < 4 or args.warmstart_iterations < 0 or args.repeats < 1:
        parser.error("n>=4, warmstart-iterations>=0, repeats>=1 required")
    domain = warpower.SurfaceDomain(*torus(48, 24), surface_subdivisions=4)
    warm = VTMWorkspace(
        domain, torus_sites(args.n, args.seed), max_iterations=args.warmstart_iterations
    ).run()
    assert warm.error.numpy()[0] == 0
    before = metrics(
        warpower.ExtractionWorkspace(
            domain,
            warm.points,
            warm.full_tetrahedra,
            warm.full_tetrahedra_count,
            source_error=warm.error,
        ).run()
    )
    core = VTMWorkspace(domain, wp.clone(warm.points), max_iterations=0)
    jitter = BoundaryJitterWorkspace(core, seed=args.seed, max_attempts=args.attempts)
    with wp.ScopedCapture() as cap:
        wp.copy(core.points, warm.points)
        core.run()
        jitter.run()
    for _ in range(2):
        wp.capture_launch(cap.graph)
    wp.synchronize()
    samples = []
    begin, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
    quota = cpu_quota_snapshot()
    for _ in range(args.repeats):
        wp.record_event(begin)
        wp.capture_launch(cap.graph)
        wp.record_event(end)
        wp.synchronize_event(end)
        samples.append(wp.get_event_elapsed_time(begin, end))
    cpu = cpu_quota_delta(quota, cpu_quota_snapshot())
    if args.dot:
        args.dot.parent.mkdir(parents=True, exist_ok=True)
        wp.capture_debug_dot_print(cap.graph, str(args.dot), verbose=True)
    if args.profile:
        import torch

        torch.cuda.cudart().cudaProfilerStart()
        for _ in range(3):
            wp.capture_launch(cap.graph)
        wp.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    assert jitter.error.numpy()[0] == core.error.numpy()[0] == 0
    result = dict(
        scope="Resident fitted-point reset, certified reconstruction/statistics refresh, one bounded jitter search including extraction and diagnostics. Initial fitting, sizing, sampling and other mesher stages excluded.",
        device=core.device.name,
        warp=wp.__version__,
        n=args.n,
        seed=args.seed,
        initial_global_iterations=args.warmstart_iterations,
        max_attempts=args.attempts,
        attempts=int(jitter.attempts.numpy()[0]),
        accepted=int(jitter.accepted.numpy()[0]),
        status=int(jitter.status.numpy()[0]),
        rejection_reason=int(jitter.rejection_reason.numpy()[0]),
        last_errors=jitter.last_errors.numpy().tolist(),
        site=int(jitter.site.numpy()[0]),
        proposed_distance=float(jitter.proposed_distance.numpy()[0]),
        proposed_limit=float(jitter.proposed_limit.numpy()[0]),
        before=before,
        after=metrics(jitter.extraction),
        quality_deficit_before=float(jitter.previous_metrics.numpy()[1]),
        quality_deficit_after=float(jitter.metrics.numpy()[1]),
        gpu_milliseconds=samples,
        median_gpu_milliseconds=float(np.median(samples)),
        cpu_quota_delta=cpu,
        cpu_throttled=any(v["throttled_periods"] for v in cpu.values()),
    )
    if args.reference:
        p = core.points.numpy()
        full = jitter.full_tetrahedra.numpy()[: jitter.full_tetrahedra_count.numpy()[0]]
        check_mesh(p, full)
        assert tet_set(full) == reference(p)
        _, components, chi = independent_topology(
            p, jitter.tetrahedra.numpy()[: jitter.tetrahedra_count.numpy()[0]]
        )
        assert components == result["after"]["components"] and chi == sum(
            result["after"]["shell_euler"]
        )
        result["independent_delaunay_and_topology_match"] = True
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
