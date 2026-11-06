"""Captured progressive torus refinement; initialization and full mesher excluded."""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

import warpower
from benchmarks.run import cpu_quota_delta, cpu_quota_snapshot
from tests.test_domain import torus
from tests.test_extraction import independent_topology
from tests.test_tetrahedra import check_mesh, reference, tet_set
from tests.test_vtm_core import torus_sites
from warpower._refinement import RefinementWorkspace
from warpower._vtm import VTMWorkspace


@wp.kernel
def record_stage(
    stage: int,
    active: wp.array[int],
    inserted: wp.array[int],
    eligible: wp.array[int],
    error: wp.array[int],
    core_error: wp.array[int],
    records: wp.array2d[int],
):
    records[stage, 0] = active[0]
    records[stage, 1] = inserted[0]
    records[stage, 2] = eligible[0]
    records[stage, 3] = error[0]
    records[stage, 4] = core_error[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--initial", type=int, default=256)
    parser.add_argument("--stages", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--polish-rounds", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dot", type=Path)
    args = parser.parse_args()
    if (
        not 4 <= args.initial <= args.n
        or min(args.stages, args.repeats) < 1
        or min(args.iterations, args.polish_rounds) < 0
    ):
        parser.error(
            "4 <= initial <= n; stages/repeats >= 1; iterations/polish-rounds >= 0 required"
        )
    domain = warpower.SurfaceDomain(*torus(48, 24), surface_subdivisions=4)
    data = np.full((args.n, 3), np.nan, np.float32)
    data[: args.initial] = torus_sites(args.initial, args.seed)
    source = wp.array(data, dtype=wp.vec3, device="cuda:0")
    points = wp.clone(source)
    active = wp.array([args.initial], dtype=int, device="cuda:0")
    core = VTMWorkspace(domain, points, active_count=active, max_iterations=args.iterations)
    refine = RefinementWorkspace(core, args.n)
    extract = warpower.ExtractionWorkspace(
        domain, points, core.full_tetrahedra, core.full_tetrahedra_count, source_error=core.error
    )
    records = wp.zeros((args.stages, 5), dtype=int, device="cuda:0")
    wp.load_module(module=__name__, device="cuda:0")
    with wp.ScopedCapture() as capture:
        wp.copy(points, source)
        active.fill_(args.initial)
        core.run()
        for stage in range(args.stages):
            refine.run()
            core.run()
            wp.launch(
                record_stage,
                1,
                inputs=[
                    stage,
                    active,
                    refine.inserted,
                    refine.eligible,
                    refine.error,
                    core.error,
                    records,
                ],
            )
        for _ in range(args.polish_rounds):
            core.run()
        extract.run()
    wp.capture_launch(capture.graph)
    wp.synchronize()
    samples = []
    begin, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
    before = cpu_quota_snapshot()
    for _ in range(args.repeats):
        wp.record_event(begin)
        wp.capture_launch(capture.graph)
        wp.record_event(end)
        wp.synchronize_event(end)
        samples.append(wp.get_event_elapsed_time(begin, end))
    cpu = cpu_quota_delta(before, cpu_quota_snapshot())
    if args.dot:
        args.dot.parent.mkdir(parents=True, exist_ok=True)
        wp.capture_debug_dot_print(capture.graph, str(args.dot), verbose=True)
    if args.profile:
        import torch

        torch.cuda.cudart().cudaProfilerStart()
        for _ in range(3):
            wp.capture_launch(capture.graph)
        wp.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    assert np.all(records.numpy()[:, 3:] == 0), records.numpy()
    assert core.error.numpy()[0] == extract.error.numpy()[0] == 0
    assert active.numpy()[0] == args.n, "Stage budget did not reach requested target"
    nt = int(extract.tetrahedra_count.numpy()[0])
    nf = int(extract.boundary_count.numpy()[0])
    shells = np.unique(extract.shell_labels.numpy()[:nf])
    result = dict(
        scope="Resident initial point copy, active-count reset, initial optimization, fixed refinement/optimization stages, domain extraction; sampling, sizing, and remaining mesher stages excluded",
        device=points.device.name,
        warp=wp.__version__,
        case="torus",
        seed=args.seed,
        initial=args.initial,
        target=args.n,
        iterations_per_stage=args.iterations,
        polish_rounds=args.polish_rounds,
        stage_columns=["active", "inserted", "eligible", "refinement_error", "core_error"],
        stages=records.numpy().tolist(),
        full_hull_tetrahedra=int(core.full_tetrahedra_count.numpy()[0]),
        tetrahedra=nt,
        diagnostics=int(extract.diagnostics.numpy()[0]),
        components=int(extract.component_count.numpy()[0]),
        shell_euler=extract.shell_euler.numpy()[shells].tolist(),
        volume=float(extract.volume.numpy()[0]),
        domain_volume=float(extract.domain_volume.numpy()[0]),
        surface_rms_distance=float(extract.surface_rms_distance.numpy()[0]),
        surface_max_sample_distance=float(extract.surface_max_sample_distance.numpy()[0]),
        surface_metric="Directed three-point quadrature per extracted face to input triangles",
        radius_ratio_percentiles=dict(
            zip(
                ["0", "1", "5", "50", "95", "100"],
                np.percentile(extract.radius_ratio.numpy()[:nt], [0, 1, 5, 50, 95, 100]).tolist(),
            )
        ),
        gpu_milliseconds=samples,
        median_gpu_milliseconds=float(np.median(samples)),
        cpu_quota_delta=cpu,
        cpu_throttled=any(v["throttled_periods"] for v in cpu.values()),
    )
    if args.reference:
        p = points.numpy()
        full = core.full_tetrahedra.numpy()[: core.full_tetrahedra_count.numpy()[0]]
        check_mesh(p, full)
        assert tet_set(full) == reference(p)
        _, components, chi = independent_topology(p, extract.tetrahedra.numpy()[:nt])
        assert components == result["components"] and chi == sum(result["shell_euler"])
        result["independent_delaunay_and_topology_match"] = True
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
