"""CUDA-event stage timings for optional exact-certified Delaunay construction."""

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import warp as wp

from benchmarks.run import cpu_quota_delta, cpu_quota_snapshot
from warpower import BoundaryWorkspace, DelaunayWorkspace, SurfaceDomain
from warpower._delaunay_recovery import normalize
from warpower._vtm import anchor_corners


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=["cube", "grid", "corner-cloud", "random"], default="random"
    )
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dot", type=Path)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    rng = np.random.default_rng(2026)
    if args.case in ("cube", "grid"):
        values = [-1, 1] if args.case == "cube" else range(4)
        points = np.array(list(product(values, repeat=3)), np.float32)
    elif args.case == "corner-cloud":
        from tests.test_domain import cube

        points = rng.uniform(-0.9, 0.9, (200, 3)).astype(np.float32)
        domain = SurfaceDomain(*cube(), surface_subdivisions=16)
        boundary = BoundaryWorkspace(domain, points).run()
        wp.launch(
            anchor_corners,
            len(points),
            inputs=[boundary.points, boundary.targets, boundary.roles, boundary.error],
        )
        points = boundary.points.numpy()
    else:
        points = rng.uniform(-1, 1, (args.n, 3)).astype(np.float32)
    workspace = DelaunayWorkspace(points).run()
    wp.synchronize()
    assert workspace.error.numpy()[0] == 0
    begin, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)

    def measure(graph):
        wp.capture_launch(graph)
        wp.synchronize()
        before = cpu_quota_snapshot()
        samples = []
        for _ in range(args.repeats):
            wp.record_event(begin)
            wp.capture_launch(graph)
            wp.record_event(end)
            wp.synchronize_event(end)
            samples.append(wp.get_event_elapsed_time(begin, end))
        return dict(
            median_ms=float(np.median(samples)),
            samples_ms=samples,
            cpu=cpu_quota_delta(before, cpu_quota_snapshot()),
        )

    full = measure(workspace.graph)
    base = measure(workspace._base.graph)
    attempts = wp.full(
        1, max(0, int(workspace.attempts.numpy()[0]) - 1), dtype=int, device=workspace.device
    )
    wp.launch(
        normalize,
        len(points),
        inputs=[
            workspace.points,
            workspace._base._bounds,
            workspace._base._hierarchy.owners,
            attempts,
            workspace._normalized,
            workspace._weights,
        ],
    )
    auxiliary = measure(workspace._auxiliary.graph)
    with wp.ScopedCapture() as capture:
        workspace._certificate.run()
    certificate = measure(capture.graph)
    workspace.run()
    wp.synchronize()
    assert workspace.error.numpy()[0] == 0
    if args.dot:
        args.dot.parent.mkdir(parents=True, exist_ok=True)
        wp.capture_debug_dot_print(workspace.graph, str(args.dot), verbose=True)
    if args.profile:
        import torch

        torch.cuda.cudart().cudaProfilerStart()
        for _ in range(3):
            workspace.run()
        wp.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    result = dict(
        case=args.case,
        n=len(points),
        seed=2026,
        device=workspace.device.name,
        warp=wp.__version__,
        full=full,
        base=base,
        auxiliary=auxiliary,
        certificate=certificate,
        tetrahedra=int(workspace.tetrahedra_count.numpy()[0]),
        error=int(workspace.error.numpy()[0]),
        base_error=int(workspace.base_error.numpy()[0]),
        candidate_error=int(workspace.candidate_error.numpy()[0]),
        certificate_error=int(workspace.certificate_error.numpy()[0]),
        attempts=int(workspace.attempts.numpy()[0]),
        certificate_point_tests=int(workspace._certificate.statistics.numpy()[0]),
        certificate_node_visits=int(workspace._certificate.statistics.numpy()[1]),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
