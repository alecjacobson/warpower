"""Time one interior teleportation/insertion transaction on a fitted sharp cube."""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

import warpower
from benchmarks.run import cpu_quota_delta, cpu_quota_snapshot
from benchmarks.selective import metrics
from tests.test_delaunay_recovery import exact_empty
from tests.test_domain import cube
from tests.test_extraction import independent_topology
from tests.test_tetrahedra import check_mesh, reference
from warpower._teleport import InteriorImprovementWorkspace
from warpower._vtm import VTMWorkspace


def interior_metrics(extract, core):
    result = metrics(extract)
    tets = extract.tetrahedra.numpy()[: result["tetrahedra"]]
    inside = (core.boundary.roles.numpy()[tets] == 0).all(axis=1)
    q = extract.radius_ratio.numpy()[: len(tets)][inside]
    result["minimum_interior_radius_ratio"] = float(q.min()) if len(q) else None
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--warmstart-iterations", type=int, default=5)
    parser.add_argument("--mode", choices=["teleport", "insert"], default="teleport")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--dot", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.n < 4 or args.warmstart_iterations < 0 or args.repeats < 1:
        parser.error("n>=4, warmstart-iterations>=0, repeats>=1 required")
    domain = warpower.SurfaceDomain(*cube(), surface_subdivisions=6)
    source = np.full((args.n + 1, 3), np.nan, np.float32)
    source[: args.n] = np.random.default_rng(args.seed).uniform(-0.9, 0.9, (args.n, 3))
    initial_count = wp.full(1, args.n, dtype=int, device="cuda:0")
    warm = VTMWorkspace(
        domain, source, active_count=initial_count, max_iterations=args.warmstart_iterations
    ).run()
    assert warm.error.numpy()[0] == 0
    initial_extract = warpower.ExtractionWorkspace(
        domain,
        warm.points,
        warm.full_tetrahedra,
        warm.full_tetrahedra_count,
        source_error=warm.error,
    ).run()
    before = interior_metrics(initial_extract, warm)
    count = wp.full(1, args.n, dtype=int, device=warm.device)
    core = VTMWorkspace(domain, wp.clone(warm.points), active_count=count, max_iterations=0)
    jitter = InteriorImprovementWorkspace(
        core, target_count=args.n + 1 if args.mode == "insert" else None
    )
    with wp.ScopedCapture() as cap:
        wp.copy(core.points, warm.points)
        count.fill_(args.n)
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
        scope="Resident fitted-point/count reset, certified reconstruction/statistics refresh, one interior quality transaction including extraction and diagnostics. Initial fitting, sizing, sampling and other mesher stages excluded.",
        device=core.device.name,
        warp=wp.__version__,
        n=args.n,
        seed=args.seed,
        initial_global_iterations=args.warmstart_iterations,
        mode=args.mode,
        point_count=int(jitter.point_count.numpy()[0]),
        inserted=int(jitter.inserted.numpy()[0]),
        attempts=int(jitter.attempts.numpy()[0]),
        accepted=int(jitter.accepted.numpy()[0]),
        status=int(jitter.status.numpy()[0]),
        rejection_reason=int(jitter.rejection_reason.numpy()[0]),
        last_errors=jitter.last_errors.numpy().tolist(),
        site=int(jitter.site.numpy()[0]),
        source_tetrahedron=jitter.canonical_tetrahedron.numpy().tolist(),
        dual_volume_error=int(jitter.cells.error.numpy()[0]),
        before=before,
        after=interior_metrics(jitter.extraction, core),
        quality_deficit_before=float(jitter.previous_metrics.numpy()[1]),
        quality_deficit_after=float(jitter.metrics.numpy()[1]),
        gpu_milliseconds=samples,
        median_gpu_milliseconds=float(np.median(samples)),
        cpu_quota_delta=cpu,
        cpu_throttled=any(v["throttled_periods"] for v in cpu.values()),
    )
    if args.reference:
        p = core.points.numpy()[: jitter.point_count.numpy()[0]]
        full = jitter.full_tetrahedra.numpy()[: jitter.full_tetrahedra_count.numpy()[0]]
        check_mesh(p, full)
        expected = reference(p)
        different = np.array([tet for tet in full if tuple(sorted(tet)) not in expected], np.int32)
        if len(different):
            exact_empty(p, different)
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
