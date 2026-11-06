"""Boundary/interior core benchmark; independent domain extraction is excluded."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

import warpower
from benchmarks.run import cpu_quota_delta, cpu_quota_snapshot
from tests.test_domain import cube, torus
from tests.test_tetrahedra import check_mesh, reference, tet_set
from tests.test_vtm_core import consume, produce, torus_sites
from warpower._vtm import VTMWorkspace


@wp.kernel
def advance(phase: wp.array[int]):
    phase[0] += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["torus", "cube"], default="torus")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--dot", type=Path)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--corner-only", action="store_true", help="Skip the cube's full boundary prefit"
    )
    args = parser.parse_args()
    if args.n < 8 or args.iterations < 0 or args.repeats < 1:
        parser.error("n>=8, iterations>=0 and repeats>=1 required")
    start = time.perf_counter()
    if args.case == "torus":
        source = torus_sites(args.n, args.seed)
        domain = warpower.SurfaceDomain(*torus(48, 24), surface_subdivisions=4)
    else:
        source = np.random.default_rng(args.seed).uniform(-0.9, 0.9, (args.n, 3)).astype(np.float32)
        domain = warpower.SurfaceDomain(*cube(), surface_subdivisions=16)
    base = wp.array(source, dtype=wp.vec3, device="cuda:0")
    points = wp.clone(base)
    core = VTMWorkspace(domain, points, max_iterations=args.iterations)
    boundary_init = (
        warpower.BoundaryWorkspace(domain, points)
        if args.case == "cube" and not args.corner_only
        else None
    )
    phase = wp.zeros(1, dtype=int, device="cuda:0")
    volume = wp.zeros(1, dtype=wp.float64, device="cuda:0")
    wp.load_module(module=__name__, device="cuda:0")
    wp.load_module(module="tests.test_vtm_core", device="cuda:0")
    with wp.ScopedCapture() as capture:
        wp.launch(produce, len(base), inputs=[base, points, phase])
        if boundary_init is not None:
            boundary_init.run()
            boundary_init.apply()
        core.run()
        volume.zero_()
        wp.launch(
            consume,
            len(core.full_tetrahedra),
            inputs=[points, core.full_tetrahedra, core.full_tetrahedra_count, core.error, volume],
        )
        wp.launch(advance, 1, inputs=[phase])
    wp.synchronize()
    setup = time.perf_counter() - start
    for _ in range(3):
        wp.capture_launch(capture.graph)
    wp.synchronize()
    assert core.error.numpy()[0] == 0, core.mesh.error.numpy()
    samples = []
    begin, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
    cpu_before = cpu_quota_snapshot()
    for _ in range(args.repeats):
        wp.record_event(begin)
        wp.capture_launch(capture.graph)
        wp.record_event(end)
        wp.synchronize_event(end)
        samples.append(wp.get_event_elapsed_time(begin, end))
        assert core.error.numpy()[0] == 0, core.mesh.error.numpy()
    cpu = cpu_quota_delta(cpu_before, cpu_quota_snapshot())
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
    n_tets = int(core.full_tetrahedra_count.numpy()[0])
    n_iterations = int(core.iterations.numpy()[0])
    result = dict(
        case=args.case,
        seed=args.seed,
        n=args.n,
        device=core.device.name,
        warp=wp.__version__,
        requested_iterations=args.iterations,
        iterations=n_iterations,
        error=int(core.error.numpy()[0]),
        termination=int(core.termination.numpy()[0]),
        rejected=int(core.rejected_steps.numpy()[0]),
        last_geometry_error=int(core.last_geometry_error.numpy()[0]),
        last_boundary_error=int(core.last_boundary_error.numpy()[0]),
        surface_samples=len(domain.surface_points),
        feature_edges=int(domain.feature_count.numpy()[0]),
        corners=int(domain.corner_count.numpy()[0]),
        full_boundary_prefit=boundary_init is not None,
        roles=core.boundary.role_counts.numpy().tolist(),
        full_hull_tetrahedra=n_tets,
        full_hull_volume=float(core.volume.numpy()[0]),
        minimum_full_hull_radius_ratio=float(core.minimum_radius_ratio.numpy()[0]),
        energy_history_columns=["full_hull_odt", "sampled_surface", "sampled_crease"],
        energy_history=core.energy_history.numpy()[: n_iterations + 1].tolist(),
        gpu_milliseconds=samples,
        median_gpu_milliseconds=float(np.median(samples)),
        setup_seconds=setup,
        cpu_quota_delta=cpu,
        cpu_throttled=any(v["throttled_periods"] for v in cpu.values()),
        scope="Resident producer, optional boundary initialization, uniform fixed-budget VTM core and volume consumer; full-hull connectivity, no domain extraction; three cycling input translations",
    )
    assert result["error"] == 0
    assert np.isclose(volume.numpy()[0], core.volume.numpy()[0], rtol=1e-12)
    if args.reference or args.save_dir:
        p = core.points.numpy()
        tets = core.full_tetrahedra.numpy()[:n_tets]
        if args.reference:
            check_mesh(p, tets)
            result["matches_scipy"] = tet_set(tets) == reference(p)
            assert result["matches_scipy"]
        if args.save_dir:
            args.save_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.save_dir / (args.case + ".npz"), points=p, tetrahedra=tets, initial=source
            )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
