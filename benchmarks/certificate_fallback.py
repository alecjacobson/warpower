"""Compare certificate paths on identical corner-cloud buffers and shuffled cell orders."""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

from benchmarks.run import cpu_quota_delta, cpu_quota_snapshot
from tests.test_domain import cube
from warpower import BoundaryWorkspace, DelaunayWorkspace, SurfaceDomain
from warpower._delaunay_certificate import DelaunayCertificate
from warpower._site_bvh import SiteHierarchy
from warpower._vtm import anchor_corners


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--profile", choices=["old", "new"])
    a = p.parse_args()
    points = np.random.default_rng(2026).uniform(-0.9, 0.9, (200, 3)).astype(np.float32)
    domain = SurfaceDomain(*cube(), surface_subdivisions=16)
    boundary = BoundaryWorkspace(domain, points).run()
    wp.launch(
        anchor_corners,
        len(points),
        inputs=[boundary.points, boundary.targets, boundary.roles, boundary.error],
    )
    points = boundary.points.numpy()
    w = DelaunayWorkspace(points).run()
    assert w.error.numpy()[0] == 0
    n = int(w.tetrahedra_count.numpy()[0])
    original = w.tetrahedra.numpy()
    source = wp.clone(w.tetrahedra)
    count = wp.full(1, n, dtype=int, device=w.device)
    h = SiteHierarchy(w.points)
    h.tree.owners = wp.array(np.arange(len(points), dtype=np.int32), device=w.device)
    certificates = []
    graphs = []
    for old in [True, False]:
        c = DelaunayCertificate(w.points, source, count, h.tree)
        if old:
            c._anchors = None
        with wp.ScopedCapture() as cap:
            c.run()
        certificates.append(c)
        graphs.append(cap.graph)
    start, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
    results = []
    for seed in [0, 1, 2]:
        order = np.random.default_rng(seed).permutation(n)
        t = original.copy()
        t[:n] = original[order]
        wp.copy(source, wp.array(t, dtype=int, device=w.device))
        wp.synchronize()
        for g in graphs:
            wp.capture_launch(g)
        wp.synchronize()
        before = cpu_quota_snapshot()
        samples = [[], []]
        for iteration in range(30):
            for index in [0, 1] if iteration % 2 == 0 else [1, 0]:
                wp.record_event(start)
                wp.capture_launch(graphs[index])
                wp.record_event(end)
                wp.synchronize_event(end)
                samples[index].append(wp.get_event_elapsed_time(start, end))
        quota = cpu_quota_delta(before, cpu_quota_snapshot())
        for c in certificates:
            assert c.error.numpy()[0] == 0
            assert c.tetrahedra_count.numpy()[0] == n
        assert certificates[1]._local.numpy()[0] == 0
        np.testing.assert_array_equal(
            certificates[0].tetrahedra.numpy()[:n], certificates[1].tetrahedra.numpy()[:n]
        )
        results.append(
            dict(
                seed=seed,
                old_ms=samples[0],
                new_ms=samples[1],
                old_median_ms=float(np.median(samples[0])),
                new_median_ms=float(np.median(samples[1])),
                cpu=quota,
            )
        )
    if a.profile:
        import torch

        index = 0 if a.profile == "old" else 1
        torch.cuda.cudart().cudaProfilerStart()
        for _ in range(3):
            wp.capture_launch(graphs[index])
        wp.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    r = dict(
        scope="Paired old/global-only and proposed local-with-global-fallback certificate graphs, identical fixed corner-cloud candidate buffers and original-site hierarchy, three shuffled candidate orderings, alternating execution order, 30 samples each.",
        tetrahedra=n,
        results=results,
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
