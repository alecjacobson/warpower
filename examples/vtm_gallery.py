"""Run a complete captured meshing example and export its result to ParaView."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

from benchmarks.run import cpu_quota_delta, cpu_quota_snapshot
from examples.vtm_inputs import surface, write_off
from warpower import SizingField, SurfaceDomain, VariationalMeshingWorkspace


def write_vtu(path, points, tets, quality):
    """Portable ASCII VTK unstructured grid; serialization is outside timing."""
    with Path(path).open("w") as f:
        f.write(
            '<?xml version="1.0"?>\n<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n<UnstructuredGrid>\n'
        )
        f.write(f'<Piece NumberOfPoints="{len(points)}" NumberOfCells="{len(tets)}">\n')
        f.write('<Points><DataArray type="Float32" NumberOfComponents="3" format="ascii">\n')
        np.savetxt(f, points, fmt="%.9g")
        f.write(
            '</DataArray></Points>\n<Cells>\n<DataArray type="Int32" Name="connectivity" format="ascii">\n'
        )
        np.savetxt(f, tets, fmt="%d")
        f.write('</DataArray>\n<DataArray type="Int32" Name="offsets" format="ascii">\n')
        np.savetxt(f, 4 * np.arange(1, len(tets) + 1), fmt="%d")
        f.write('</DataArray>\n<DataArray type="UInt8" Name="types" format="ascii">\n')
        np.savetxt(f, np.full(len(tets), 10), fmt="%d")
        f.write(
            '</DataArray></Cells>\n<CellData Scalars="radius_ratio">\n<DataArray type="Float64" Name="radius_ratio" format="ascii">\n'
        )
        np.savetxt(f, quality, fmt="%.17g")
        f.write("</DataArray></CellData>\n</Piece>\n</UnstructuredGrid>\n</VTKFile>\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=[
            "torus",
            "graded-torus",
            "thin-hand",
            "counterbored-plate",
            "cavity",
            "disconnected",
            "surface",
            "graded-surface",
        ],
        default="torus",
    )
    parser.add_argument("--input", type=Path, help="Closed oriented triangular ASCII OFF surface")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument(
        "--initial",
        type=int,
        help="Minimum initial sites (default: up to 256, with room for mandatory corners)",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--subdivisions",
        type=int,
        help="Surface quadrature subdivisions (20 for thin-hand/cavity/disconnected, 16 for counterbored-plate, otherwise 4)",
    )
    parser.add_argument(
        "--crease-subdivisions",
        type=int,
        help="Crease quadrature subdivisions (20 for thin-hand/cavity/disconnected, 16 for counterbored-plate, otherwise 4)",
    )
    parser.add_argument(
        "--feature-angle", type=float, default=45.0, help="Crease detection angle in degrees"
    )
    parser.add_argument(
        "--corner-angle", type=float, default=45.0, help="Corner detection angle in degrees"
    )
    parser.add_argument("--quality-rounds", type=int, default=4)
    parser.add_argument("--local-quality-retries", type=int, default=0)
    parser.add_argument(
        "--proven-exterior-recovery",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Final whole-exterior proof recovery (default: follows exterior recovery)",
    )
    parser.add_argument("--automatic-sizing", action="store_true")
    parser.add_argument(
        "--query-backend",
        choices=["reference", "native"],
        default="reference",
        help="Surface queries: reference float64 traversal or native Warp float32 queries",
    )
    parser.add_argument(
        "--validate-geometry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate surface intersections/orientation during refit; disable for trusted input",
    )
    parser.add_argument(
        "--boundary-query-precision", choices=["float64", "float32"], default="float64"
    )
    parser.add_argument(
        "--repeats", type=int, default=0, help="Additional replays after the first complete solve"
    )
    parser.add_argument(
        "--independent-quality",
        action="store_true",
        help="Also measure two-sided fit and quality with independent CPU routines (can be slow)",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    args = parser.parse_args()
    default_subdivisions = 20 if args.case in ("thin-hand", "cavity", "disconnected") else 4
    if args.case == "counterbored-plate":
        default_subdivisions = 16
    if args.subdivisions is None:
        args.subdivisions = default_subdivisions
    if args.crease_subdivisions is None:
        args.crease_subdivisions = default_subdivisions
    if args.n < 4 or (args.initial is not None and args.initial < 4) or args.repeats < 0:
        parser.error("Require n>=4, initial>=4, repeats>=0")
    if min(args.subdivisions, args.crease_subdivisions) < 1:
        parser.error("Quadrature subdivisions must be positive")
    if not (0 < args.feature_angle < 180 and 0 < args.corner_angle < 180):
        parser.error("Feature and corner angles must lie between 0 and 180 degrees")
    if args.automatic_sizing and args.case.startswith("graded-"):
        parser.error("Use either the prescribed graded case or --automatic-sizing")
    points, faces = surface(args.case, args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    write_off(args.output / "input.off", points, faces)
    wp.init()
    quota = cpu_quota_snapshot()
    start = time.perf_counter()
    domain = SurfaceDomain(
        points,
        faces,
        surface_subdivisions=args.subdivisions,
        crease_subdivisions=args.crease_subdivisions,
        feature_angle=args.feature_angle,
        corner_angle=args.corner_angle,
        query_backend=args.query_backend,
        validate_geometry=args.validate_geometry,
    )
    sizing = None
    if args.case.startswith("graded-"):
        # Relative sizes from 0.35 on the left to at most1 on the right.
        # The min-cone extension is evaluated on device during the full solve.
        lower, upper = points.min(axis=0), points.max(axis=0)
        anchor = (lower + upper) / 2
        anchor[0] = lower[0]
        sizing = SizingField(
            wp.array(anchor[None], dtype=wp.vec3, device=domain.device),
            wp.array([0.35], dtype=float, device=domain.device),
            gradation=0.65 / float(upper[0] - lower[0]),
            minimum=0.35,
            maximum=1.0,
        )
    mesh = VariationalMeshingWorkspace(
        domain,
        args.n,
        initial_count=args.initial,
        boundary_query_precision=args.boundary_query_precision,
        seed=args.seed,
        sizing=sizing,
        quality_rounds=args.quality_rounds,
        local_quality_retries=args.local_quality_retries,
        proven_exterior_recovery=args.proven_exterior_recovery,
        automatic_sizing=args.automatic_sizing,
    )
    with wp.ScopedCapture() as captured:
        domain.refit()
        if sizing is not None:
            sizing.refit()
        mesh.run()
    wp.synchronize()
    setup_seconds = time.perf_counter() - start
    begin, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
    setup_quota = cpu_quota_delta(quota, cpu_quota_snapshot())
    first_quota = cpu_quota_snapshot()
    wall_start = time.perf_counter()
    wp.record_event(begin)
    wp.capture_launch(captured.graph)
    wp.record_event(end)
    wp.synchronize_event(end)
    first_seconds = time.perf_counter() - wall_start
    first_gpu_ms = wp.get_event_elapsed_time(begin, end)
    first_quota_delta = cpu_quota_delta(first_quota, cpu_quota_snapshot())
    replay_quota = cpu_quota_snapshot()
    samples = []
    for _ in range(args.repeats):
        wp.record_event(begin)
        wp.capture_launch(captured.graph)
        wp.record_event(end)
        wp.synchronize_event(end)
        samples.append(wp.get_event_elapsed_time(begin, end))
    replay_quota_delta = cpu_quota_delta(replay_quota, cpu_quota_snapshot())
    quota_delta = cpu_quota_delta(quota, cpu_quota_snapshot())
    result = dict(
        scope="Complete device-resident meshing, including domain/sizing refit, fresh sampling, progressive fitting, refinement, quality improvement and extraction. Setup, upload, download, validation and export are outside replay timing.",
        parameters={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        device=domain.device.name,
        warp=wp.__version__,
        input_points=len(points),
        input_faces=len(faces),
        domain_error=int(domain.error.numpy()[0]),
        detected_feature_edges=int(domain.feature_count.numpy()[0]),
        detected_corners=int(domain.corner_count.numpy()[0]),
        minimum_initial_sites=mesh.initial_count,
        effective_initial_sites=int(mesh.effective_initial_count.numpy()[0]),
        surface_quadrature_capacity=len(domain.surface_points),
        crease_quadrature_capacity=len(domain.crease_points),
        input_off_sha256=hashlib.sha256((args.output / "input.off").read_bytes()).hexdigest(),
        setup_seconds=setup_seconds,
        first_replay_wall_seconds=first_seconds,
        first_replay_gpu_ms=first_gpu_ms,
        additional_replay_gpu_ms=samples,
        median_additional_replay_ms=float(np.median(samples)) if samples else None,
        cpu_quota_delta=quota_delta,
        setup_cpu_quota_delta=setup_quota,
        first_replay_cpu_quota_delta=first_quota_delta,
        additional_replays_cpu_quota_delta=replay_quota_delta,
        valid=int(mesh.valid.numpy()[0]),
        error=int(mesh.error.numpy()[0]),
        status=int(mesh.status.numpy()[0]),
        counts=mesh.counts.numpy().tolist(),
        topology=mesh.topology.numpy().tolist(),
        rounds=mesh.rounds.numpy().tolist(),
        steps=mesh.steps.numpy().tolist(),
        moves=mesh.moves.numpy().tolist(),
        last_errors=mesh.last_errors.numpy().tolist(),
        refinement_rejections=int(mesh.refinement_rejections.numpy()[0]),
        quality_rejections=int(mesh.quality_rejections.numpy()[0]),
        last_bad_extraction=int(mesh.last_bad_extraction.numpy()[0]),
        final_extraction_error=int(mesh.extraction.error.numpy()[0]),
        final_extraction_diagnostics=int(mesh.extraction.diagnostics.numpy()[0]),
        last_quality_stage_errors={
            name: None if stage is None else int(stage.error.numpy()[0])
            for name, stage in (
                ("selection", mesh.core.selection),
                ("jitter", mesh.jitter),
                ("interior", mesh.interior),
                ("bounded_dual", None if mesh.interior is None else mesh.interior.cells),
            )
        },
        quality_stage_diagnostic_scope="Child scratch can describe a rejected or skipped trial; published mesh validity is determined by the parent error/valid/counts.",
    )
    if result["error"] & 1 and result["detected_corners"] > args.n:
        result["configuration_failure"] = (
            f"The surface has {result['detected_corners']} mandatory corners but the target is "
            f"{args.n} sites. Increase --n to at least {result['detected_corners']} "
            "or explicitly choose a different feature policy."
        )
    if mesh.containment_recovery:
        result["material_recovery"] = dict(
            activated=int(mesh._material_recovery.numpy()[0]),
            last_added_cells=int(mesh.extraction.repaired_count.numpy()[0]),
            last_passes=int(mesh.extraction.repair_rounds.numpy()[0]),
        )
    if mesh.exterior_recovery:
        result["exterior_band_recovery"] = dict(
            material_quality_guard_active=int(mesh._material_recovery.numpy()[0]),
            enabled_after_fitting=int(mesh._exterior_enabled.numpy()[0]),
            last_removed_cells=int(mesh.extraction.removed_count.numpy()[0]),
            last_passes=int(mesh.extraction.exterior_rounds.numpy()[0]),
            last_rejected=int(mesh.extraction.exterior_rejected.numpy()[0]),
        )
    if mesh.proven_exterior_recovery:
        result["proven_exterior_recovery"] = dict(
            enabled_after_fitting=int(mesh._proven_exterior_enabled.numpy()[0]),
            last_removed_cells=int(mesh.extraction.proven_removed_count.numpy()[0]),
            last_rejected=int(mesh.extraction.proven_exterior_rejected.numpy()[0]),
            midpoint_statistics=mesh._boundary_rounding.statistics.numpy().tolist(),
            midpoint_statistics_columns=[
                "proposed_axes",
                "proposed_sites",
                "accepted_axes",
                "accepted_sites",
            ],
        )
    if result["valid"]:
        n, full_count, m, boundary_count = result["counts"]
        p = mesh.points.numpy()[:n]
        t = mesh.tetrahedra.numpy()[:m]
        b = mesh.boundary_faces.numpy()[:boundary_count]
        q = mesh.radius_ratio.numpy()[:m]
        result["quality"] = dict(
            radius_ratio_percentiles=dict(
                zip(
                    map(str, [0, 1, 5, 50, 95, 100]),
                    np.percentile(q, [0, 1, 5, 50, 95, 100]).tolist(),
                )
            ),
            volume=float(mesh.metrics.numpy()[0]),
            relative_volume_error=float(mesh.metrics.numpy()[2] / mesh.metrics.numpy()[1]),
            directed_surface_rms=float(mesh.metrics.numpy()[4]),
        )
        np.savez_compressed(
            args.output / "mesh.npz",
            points=p,
            tetrahedra=t,
            boundary=b,
            full_tetrahedra=mesh.full_tetrahedra.numpy()[:full_count],
            radius_ratio=q,
            roles=mesh.core.boundary.roles.numpy()[:n],
            input_points=points,
            input_faces=faces,
        )
        write_vtu(args.output / "mesh.vtu", p, t, q)
        write_off(args.output / "boundary.off", p, b)
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        if args.independent_quality:
            from benchmarks.mesh_quality import measure

            result["independent_quality"] = measure(p, t, points, faces)
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if not result["valid"]:
        raise SystemExit("Meshing failed; the saved result.json contains diagnostics")


if __name__ == "__main__":
    main()
