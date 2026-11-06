"""Compare further global/selective optimization from the same fitted torus."""

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
from warpower._vtm import VTMWorkspace


def use_full_query_reference(core):
    """Reproduce the initial implementation's unnecessary full distance queries."""
    from warpower import _extraction, _selective

    selected = core.selection
    centers = wp.empty(selected._m, dtype=wp.vec3d, device=core.device)
    radii = wp.empty(selected._m, dtype=wp.float64, device=core.device)
    distances = wp.empty_like(radii)
    volumes = wp.empty_like(radii)

    def evaluate():
        selected.retained_count.zero_()
        selected.minimum.fill_(1)
        selected._launch(
            _extraction.input_ready,
            1,
            [
                core.domain.error,
                core.mesh.error,
                core.full_tetrahedra_count,
                selected._m,
                selected.error,
            ],
        )
        selected._launch(
            _extraction.classify,
            selected._m,
            [
                core.domain.tree,
                core.points,
                core.full_tetrahedra,
                core.full_tetrahedra_count,
                selected.tolerance,
                selected.error,
                centers,
                radii,
                distances,
                selected.keep,
                volumes,
                selected.quality,
            ],
        )
        selected._launch(
            _selective.summarize,
            selected._m,
            [
                selected.keep,
                selected.quality,
                selected.error,
                selected.retained_count,
                selected.minimum,
            ],
        )

    # The parent capture below records the selected evaluator inline. This is
    # benchmark-only wiring, with all reference storage allocated before capture.
    selected.evaluate = evaluate


def metrics(extract):
    assert extract.error.numpy()[0] == 0
    count = int(extract.tetrahedra_count.numpy()[0])
    shells = np.unique(extract.shell_labels.numpy()[: extract.boundary_count.numpy()[0]])
    return dict(
        tetrahedra=count,
        diagnostics=int(extract.diagnostics.numpy()[0]),
        components=int(extract.component_count.numpy()[0]),
        shell_euler=extract.shell_euler.numpy()[shells].tolist(),
        volume=float(extract.volume.numpy()[0]),
        surface_rms_distance=float(extract.surface_rms_distance.numpy()[0]),
        radius_ratio_percentiles=dict(
            zip(
                ["0", "1", "5", "50", "95", "100"],
                np.percentile(
                    extract.radius_ratio.numpy()[:count], [0, 1, 5, 50, 95, 100]
                ).tolist(),
            )
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--warmstart-iterations", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--mode", choices=["selective", "global"], default="selective")
    parser.add_argument("--full-query-reference", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--dot", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.n < 4 or min(args.iterations, args.warmstart_iterations) < 0 or args.repeats < 1:
        parser.error("n>=4, iterations>=0 and repeats>=1 required")
    if args.full_query_reference and args.mode != "selective":
        parser.error("full-query-reference requires selective mode")
    domain = warpower.SurfaceDomain(*torus(48, 24), surface_subdivisions=4)
    warm = VTMWorkspace(
        domain, torus_sites(args.n, args.seed), max_iterations=args.warmstart_iterations
    ).run()
    assert warm.error.numpy()[0] == 0
    initial_extract = warpower.ExtractionWorkspace(
        domain,
        warm.points,
        warm.full_tetrahedra,
        warm.full_tetrahedra_count,
        source_error=warm.error,
    ).run()
    before = metrics(initial_extract)
    points = wp.clone(warm.points)
    core = VTMWorkspace(
        domain,
        points,
        max_iterations=args.iterations,
        selective_threshold=args.threshold if args.mode == "selective" else None,
    )
    if args.full_query_reference:
        use_full_query_reference(core)
    extract = warpower.ExtractionWorkspace(
        domain, points, core.full_tetrahedra, core.full_tetrahedra_count, source_error=core.error
    )
    with wp.ScopedCapture() as cap:
        wp.copy(points, warm.points)
        core.run()
        extract.run()
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
    assert core.error.numpy()[0] == 0
    result = dict(
        scope="Resident fitted-point reset, further optimization, extraction and quality diagnostics; initial global fitting, sampling, sizing, and other mesher stages excluded",
        device=points.device.name,
        warp=wp.__version__,
        mode=args.mode,
        n=args.n,
        seed=args.seed,
        initial_global_iterations=args.warmstart_iterations,
        requested_further_iterations=args.iterations,
        iterations=int(core.iterations.numpy()[0]),
        termination=int(core.termination.numpy()[0]),
        rejected_steps=int(core.rejected_steps.numpy()[0]),
        before=before,
        after=metrics(extract),
        gpu_milliseconds=samples,
        median_gpu_milliseconds=float(np.median(samples)),
        cpu_quota_delta=cpu,
        cpu_throttled=any(v["throttled_periods"] for v in cpu.values()),
    )
    if core.selection is not None:
        result["retention_classifier"] = (
            "full-query-reference" if args.full_query_reference else "retention-only"
        )
        result["threshold"] = args.threshold
        result["last_selected_vertices"] = int(core.selection.selected_count.numpy()[0])
        result["quality_rejections"] = int(core.selection.rejections.numpy()[0])
    if args.reference:
        p = points.numpy()
        full = core.full_tetrahedra.numpy()[: core.full_tetrahedra_count.numpy()[0]]
        check_mesh(p, full)
        assert tet_set(full) == reference(p)
        _, components, chi = independent_topology(
            p, extract.tetrahedra.numpy()[: extract.tetrahedra_count.numpy()[0]]
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
