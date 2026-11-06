"""Automatic pole sizing and material sampling, separately and in a captured loop."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

import warpower
from benchmarks.run import cpu_quota_delta, cpu_quota_snapshot
from tests.test_automatic_sizing import sphere
from tests.test_domain import cube, torus


@wp.kernel
def deform(source: wp.array[wp.vec3], points: wp.array[wp.vec3], phase: wp.array[int]):
    i = wp.tid()
    points[i] = source[i] * (1.0 + 0.01 * float(phase[0] % 5))


@wp.kernel
def consume(points: wp.array[wp.vec3], count: wp.array[int], total: wp.array[wp.vec3d]):
    i = wp.tid()
    if i < count[0]:
        wp.atomic_add(total, 0, wp.vec3d(points[i]))


@wp.kernel
def advance(phase: wp.array[int], seed: wp.array[int]):
    phase[0] += 1
    seed[0] += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("thin-plate", "sphere", "torus"), default="thin-plate")
    parser.add_argument("--n", type=int, default=100000)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dot", type=Path)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if min(args.n, args.repeats) < 1:
        parser.error("Counts and repetitions must be positive")
    if args.case == "thin-plate":
        points, faces = cube()
        points[:, 2] *= 0.25
        subdivisions = 1
    elif args.case == "sphere":
        points, faces = sphere(64, radius=2.0)
        subdivisions = 0
    else:
        points, faces = torus(16, 8)
        subdivisions = 0
    start = time.perf_counter()
    domain = warpower.SurfaceDomain(points, faces)
    auto = warpower.AutomaticSizingWorkspace(
        domain, control_subdivisions=subdivisions, gradation=0.5
    )
    sampler = warpower.InitialSamplingWorkspace(
        domain, args.n, sizing=auto.field, resolution=args.resolution
    )
    source = wp.array(points, dtype=wp.vec3, device=domain.device)
    phase = wp.zeros(1, dtype=int, device=domain.device)
    total = wp.zeros(1, dtype=wp.vec3d, device=domain.device)
    wp.load_module(module=__name__, device=domain.device)
    graphs = {}
    with wp.ScopedCapture() as capture:
        auto.refit()
    graphs["automatic_refit"] = capture.graph
    with wp.ScopedCapture() as capture:
        sampler.run()
    graphs["sampling"] = capture.graph
    with wp.ScopedCapture() as capture:
        wp.launch(deform, len(points), inputs=[source, domain.points, phase])
        domain.refit()
        auto.refit()
        sampler.run()
        total.zero_()
        wp.launch(consume, args.n, inputs=[sampler.points, sampler.count, total])
        wp.launch(advance, 1, inputs=[phase, sampler.seed])
    graphs["producer_domain_auto_sample_consumer"] = capture.graph
    wp.synchronize()
    setup = time.perf_counter() - start
    timings = {}
    quota_start = cpu_quota_snapshot()
    for name, graph in graphs.items():
        for _ in range(2):
            wp.capture_launch(graph)
        wp.synchronize()
        first, last = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
        durations = []
        for _ in range(args.repeats):
            wp.record_event(first)
            wp.capture_launch(graph)
            wp.record_event(last)
            wp.synchronize_event(last)
            durations.append(wp.get_event_elapsed_time(first, last))
        timings[name] = dict(samples=durations, median=float(np.median(durations)))
        errors = [int(a.numpy()[0]) for a in (domain.error, auto.error, sampler.error)]
        assert errors == [0, 0, 0], (name, errors, int(sampler.attempts.numpy().max()))
    quota = cpu_quota_delta(quota_start, cpu_quota_snapshot())
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
    cloud = sampler.points.numpy()
    query = warpower.DomainQueryWorkspace(domain, sampler.points).run()
    assert query.error.numpy()[0] == sampler.error.numpy()[0] == auto.error.numpy()[0] == 0
    assert np.all(query.inside.numpy() == 1)
    assert sampler.count.numpy()[0] == args.n
    assert len(np.unique(cloud, axis=0)) == args.n
    np.testing.assert_allclose(total.numpy()[0], cloud.astype(float).sum(axis=0), atol=1e-8)
    result = dict(
        device=domain.device.name,
        warp=wp.__version__,
        case=args.case,
        sites=args.n,
        control_sites=auto.n,
        control_subdivisions=subdivisions,
        resolution=args.resolution,
        setup_seconds=setup,
        timings_ms=timings,
        cpu_quota_delta=quota,
        cpu_throttled=any(v["throttled_periods"] for v in quota.values()),
        errors=dict(
            domain=int(domain.error.numpy()[0]),
            automatic=int(auto.error.numpy()[0]),
            sampling=int(sampler.error.numpy()[0]),
        ),
        valid_unique_interior_sites=args.n,
        finite_poles=int(auto.pole_count.numpy()[0]),
        feature_size_quantiles=np.quantile(auto.local_feature_size.numpy(), [0, 0.5, 1]).tolist(),
        attempts_quantiles=np.quantile(sampler.attempts.numpy(), [0, 0.5, 0.99, 1]).tolist(),
        scope="Resident boundary; control Delaunay/poles/sizing and density sampling. Setup separate. No volume-site tetrahedralization, VTM optimization, refinement or extraction.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
