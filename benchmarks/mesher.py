"""Measure the complete captured VTM pipeline, including initialization and diagnostics."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

import warpower
from benchmarks.run import cpu_quota_delta, cpu_quota_snapshot
from benchmarks.selective import metrics
from tests.test_domain import combine, cube, torus
from tests.test_mesher import validate


def surface(case):
    if case == "torus":
        return torus(48, 24)
    if case == "cavity":
        inner = cube(0.4)
        return combine(cube(), (inner[0], inner[1][:, ::-1].copy()))
    if case == "disconnected":
        return combine(cube(0.7, (-2, 0, 0)), cube(0.7, (2, 0, 0)))
    return cube()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case", choices=["cube", "torus", "cavity", "disconnected"], default="torus"
    )
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--capacity", type=int, help="Reserved site capacity (defaults to --n)")
    parser.add_argument("--initial", type=int, default=256)
    parser.add_argument("--growth-iterations", type=int, default=5)
    parser.add_argument("--polish-iterations", type=int, default=20)
    parser.add_argument("--selective-iterations", type=int, default=10)
    parser.add_argument("--quality-rounds", type=int, default=4)
    parser.add_argument("--local-topology-retries", type=int, default=0)
    parser.add_argument("--subdivisions", type=int, default=4)
    parser.add_argument("--automatic-sizing", action="store_true")
    parser.add_argument("--static-domain", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--dot", type=Path)
    parser.add_argument("--save-mesh", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    wp.init()
    device = wp.get_device("cuda:0")
    memory_before = wp.get_mempool_used_mem_current(device)
    setup_start = time.perf_counter()
    domain = warpower.SurfaceDomain(*surface(args.case), surface_subdivisions=args.subdivisions)
    mesh = warpower.VariationalMeshingWorkspace(
        domain,
        args.n if args.capacity is None else args.capacity,
        target_count=args.n,
        initial_count=args.initial,
        seed=args.seed,
        growth_iterations=args.growth_iterations,
        polish_iterations=args.polish_iterations,
        selective_iterations=args.selective_iterations,
        quality_rounds=args.quality_rounds,
        local_topology_retries=args.local_topology_retries,
        automatic_sizing=args.automatic_sizing,
    )
    with wp.ScopedCapture() as cap:
        if not args.static_domain:
            domain.refit()
        mesh.run()
    wp.synchronize()
    setup_seconds = time.perf_counter() - setup_start
    memory = wp.get_mempool_used_mem_current(device) - memory_before
    begin, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
    initial_quota = cpu_quota_snapshot()
    first_start = time.perf_counter()
    wp.record_event(begin)
    wp.capture_launch(cap.graph)
    wp.record_event(end)
    wp.synchronize_event(end)
    first_wall = time.perf_counter() - first_start
    first_gpu = wp.get_event_elapsed_time(begin, end)
    wp.capture_launch(cap.graph)
    wp.synchronize()
    initial_cpu = cpu_quota_delta(initial_quota, cpu_quota_snapshot())
    quota = cpu_quota_snapshot()
    samples = []
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
    result = dict(
        scope="Complete resident-device meshing: input topology, optional automatic sizing, fresh initial sampling, progressive growth, global fitting and recovery, optional quality rounds, final extraction and diagnostics. Allocation, compilation, upload and serialization excluded from replay; setup reported separately.",
        domain_refit_included=not args.static_domain,
        parameters={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        device=device.name,
        warp=wp.__version__,
        setup_seconds=setup_seconds,
        persistent_workspace_pool_bytes=memory,
        first_replay_wall_seconds=first_wall,
        first_replay_gpu_milliseconds=first_gpu,
        gpu_milliseconds=samples,
        median_gpu_milliseconds=float(np.median(samples)),
        cpu_quota_delta=cpu,
        cpu_throttled=any(v["throttled_periods"] for v in cpu.values()),
        initial_replays_cpu_quota_delta=initial_cpu,
        error=int(mesh.error.numpy()[0]),
        valid=int(mesh.valid.numpy()[0]),
        status=int(mesh.status.numpy()[0]),
        counts=mesh.counts.numpy().tolist(),
        rounds=mesh.rounds.numpy().tolist(),
        optimizer_steps=mesh.steps.numpy().tolist(),
        quality_moves=mesh.moves.numpy().tolist(),
        refinement_rejections=int(mesh.refinement_rejections.numpy()[0]),
        quality_rejections=int(mesh.quality_rejections.numpy()[0]),
        last_errors=mesh.last_errors.numpy().tolist(),
        last_bad_extraction=int(mesh.last_bad_extraction.numpy()[0]),
        topology=mesh.topology.numpy().tolist(),
    )
    if hasattr(mesh.core.mesh, "route_calls"):
        result["reconstruction_routing"] = dict(
            capacities=list(mesh.core.mesh.capacities),
            calls=mesh.core.mesh.route_calls.numpy().tolist(),
            last_capacity=int(mesh.core.mesh.selected_capacity.numpy()[0]),
        )
    if result["valid"]:
        result["quality"] = metrics(mesh.extraction)
        result["quality"]["relative_volume_error"] = float(
            mesh.volume_difference.numpy()[0] / mesh.domain_volume.numpy()[0]
        )
        count = mesh.tetrahedra_count.numpy()[0]
        worst = int(np.argmin(mesh.radius_ratio.numpy()[:count]))
        vertices = mesh.tetrahedra.numpy()[worst]
        result["quality"]["worst_tetrahedron"] = vertices.tolist()
        result["quality"]["worst_vertex_roles"] = mesh.core.boundary.roles.numpy()[
            vertices
        ].tolist()
        if args.reference:
            validate(mesh)
            result["independent_delaunay_and_topology_match"] = True
        if args.save_mesh:
            args.save_mesh.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.save_mesh,
                points=mesh.points.numpy()[: mesh.point_count.numpy()[0]],
                tetrahedra=mesh.tetrahedra.numpy()[:count],
                full_tetrahedra=mesh.full_tetrahedra.numpy()[
                    : mesh.full_tetrahedra_count.numpy()[0]
                ],
                boundary=mesh.boundary_faces.numpy()[: mesh.boundary_count.numpy()[0]],
                roles=mesh.core.boundary.roles.numpy()[: mesh.point_count.numpy()[0]],
                input_points=domain.points.numpy(),
                input_faces=domain._indices.numpy().reshape(-1, 3),
            )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if not result["valid"]:
        raise SystemExit("Mesher did not produce a valid output; inspect the saved diagnostics")


if __name__ == "__main__":
    main()
