"""Captured fitted-torus extraction, optionally including the VTM optimization."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

import warpower
from benchmarks.run import cpu_quota_delta, cpu_quota_snapshot
from benchmarks.vtm_core import advance
from tests.test_domain import brute_distances, solid_angle_inside, torus
from tests.test_extraction import consume, independent_topology
from tests.test_vtm_core import produce, torus_sites
from warpower._vtm import VTMWorkspace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--combined", action="store_true")
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--dot", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--save", type=Path)
    args = parser.parse_args()
    if args.n < 4 or args.iterations < 0 or args.repeats < 1:
        parser.error("n>=4, iterations>=0, repeats>=1 required")
    start = time.perf_counter()
    vertices, triangles = torus(48, 24)
    domain = warpower.SurfaceDomain(vertices, triangles, surface_subdivisions=4)
    source = wp.array(torus_sites(args.n, args.seed), dtype=wp.vec3, device="cuda:0")
    points = wp.clone(source)
    core = VTMWorkspace(domain, points, max_iterations=args.iterations)
    core.run()
    assert core.error.numpy()[0] == 0
    if not args.combined:
        wp.copy(source, core.points)
    extract = warpower.ExtractionWorkspace(
        domain, points, core.full_tetrahedra, core.full_tetrahedra_count, source_error=core.error
    )
    phase = wp.zeros(1, dtype=int, device="cuda:0")
    volume = wp.zeros(1, dtype=wp.float64, device="cuda:0")
    for module in ("tests.test_vtm_core", "tests.test_extraction", "benchmarks.vtm_core"):
        wp.load_module(module=module, device="cuda:0")
    with wp.ScopedCapture() as capture:
        wp.launch(produce, args.n, inputs=[source, points, phase])
        if args.combined:
            core.run()
        extract.run()
        volume.zero_()
        wp.launch(
            consume,
            len(extract.tetrahedra),
            inputs=[points, extract.tetrahedra, extract.tetrahedra_count, extract.valid, volume],
        )
        wp.launch(advance, 1, inputs=[phase])
    wp.synchronize()
    setup = time.perf_counter() - start
    for _ in range(3):
        wp.capture_launch(capture.graph)
    wp.synchronize()
    assert extract.valid.numpy()[0] == 1
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
    assert extract.error.numpy()[0] == 0
    assert extract.diagnostics.numpy()[0] == 0
    np.testing.assert_allclose(volume.numpy(), extract.volume.numpy(), rtol=1e-12)
    n_tets = int(extract.tetrahedra_count.numpy()[0])
    n_faces = int(extract.boundary_count.numpy()[0])
    n_full = int(core.full_tetrahedra_count.numpy()[0])
    shells = np.unique(extract.shell_labels.numpy()[:n_faces])
    quality = extract.radius_ratio.numpy()[:n_tets]
    result = dict(
        scope=(
            "Changing-input producer, uniform boundary/interior optimization, extraction/diagnostics, volume consumer"
            if args.combined
            else "Changing-input producer, extraction/diagnostics, volume consumer; initial optimization excluded"
        ),
        case="torus",
        seed=args.seed,
        n=args.n,
        device=points.device.name,
        warp=wp.__version__,
        iterations=args.iterations,
        tolerance=extract.tolerance,
        full_hull_tetrahedra=n_full,
        tetrahedra=n_tets,
        boundary_faces=n_faces,
        error=int(extract.error.numpy()[0]),
        diagnostics=int(extract.diagnostics.numpy()[0]),
        components=int(extract.component_count.numpy()[0]),
        shells=len(shells),
        cavities=int(extract.cavity_count.numpy()[0]),
        shell_euler=extract.shell_euler.numpy()[shells].tolist(),
        shell_volumes=extract.shell_volume.numpy()[shells].tolist(),
        volume=float(extract.volume.numpy()[0]),
        domain_volume=float(extract.domain_volume.numpy()[0]),
        volume_difference=float(extract.volume_difference.numpy()[0]),
        surface_rms_distance=float(extract.surface_rms_distance.numpy()[0]),
        surface_max_sample_distance=float(extract.surface_max_sample_distance.numpy()[0]),
        surface_metric="Three degree-two samples per extracted face to input triangles; directed, not Hausdorff",
        radius_ratio_percentiles=dict(
            zip(
                ["0", "1", "5", "50", "95", "100"],
                np.percentile(quality, [0, 1, 5, 50, 95, 100]).tolist(),
            )
        ),
        gpu_milliseconds=samples,
        median_gpu_milliseconds=float(np.median(samples)),
        setup_seconds=setup,
        cpu_quota_delta=cpu,
        cpu_throttled=any(v["throttled_periods"] for v in cpu.values()),
    )
    if args.reference or args.save:
        p = points.numpy()
        t = extract.tetrahedra.numpy()[:n_tets]
        f = extract.boundary_faces.numpy()[:n_faces]
        if args.reference:
            full = core.full_tetrahedra.numpy()[:n_full]
            cells = p[full].astype(np.float64)
            a = cells[:, 1:] - cells[:, :1]
            offset = np.linalg.solve(2 * a, np.sum(a * a, axis=2)[..., None])[..., 0]
            centers = cells[:, 0] + offset
            inside = solid_angle_inside(vertices, triangles, centers)
            distance = brute_distances(vertices, triangles, centers)
            selected = inside | (distance / np.linalg.norm(offset, axis=1) < extract.tolerance)
            np.testing.assert_array_equal(t, full[selected])
            reference_faces, components, chi = independent_topology(p, t)
            assert {tuple(sorted(x)) for x in f} == {tuple(sorted(x)) for x in reference_faces}
            assert components == result["components"] and chi == sum(result["shell_euler"])
            result["independent_classification_and_topology_match"] = True
        if args.save:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.save, points=p, tetrahedra=t, boundary_faces=f)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
