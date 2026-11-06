"""Compare global sphere kernels on identical positive-cell buffers and site order."""

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import warp as wp
from scipy.spatial import Delaunay

from benchmarks.run import cpu_quota_delta, cpu_quota_snapshot
from warpower._canonical import canonicalize
from warpower._delaunay_certificate import check_spheres, check_spheres_points
from warpower._site_bvh import SiteHierarchy


def measure(points, repeats):
    cells = Delaunay(points).simplices.astype(np.int32)
    p64 = points[cells].astype(np.float64)
    det = np.linalg.det(p64[:, 1:] - p64[:, :1])
    cells = cells[det != 0].copy()
    for i in np.flatnonzero(det[det != 0] < 0):
        cells[i, [0, 1]] = cells[i, [1, 0]]
    p = wp.array(points, dtype=wp.vec3, device="cuda:0")
    t = wp.array(cells, dtype=int, device=p.device)
    count = wp.full(1, len(t), dtype=int, device=p.device)
    hierarchy = SiteHierarchy(p)
    weights = wp.zeros(len(p), dtype=float, device=p.device)
    owners, retained = canonicalize(p, weights)
    hierarchy.tree.owners = owners
    error = wp.zeros(1, dtype=int, device=p.device)
    statistics = wp.zeros(2, dtype=int, device=p.device)
    wp.load_module(module="warpower._delaunay_certificate", device=p.device, block_dim=32)
    graphs = {}
    for name, kernel, stride in [("tree", check_spheres, 1), ("points", check_spheres_points, 32)]:
        with wp.ScopedCapture() as capture:
            error.zero_()
            statistics.zero_()
            wp.launch(
                kernel,
                stride * len(t),
                inputs=[p, t, count, hierarchy.tree, error, statistics],
                block_dim=32,
            )
        graphs[name] = capture.graph
        wp.capture_launch(capture.graph)
    wp.synchronize()
    begin, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
    samples = {k: [] for k in graphs}
    errors = {k: [] for k in graphs}
    quota = cpu_quota_snapshot()
    for i in range(repeats):
        order = ["tree", "points"] if i % 2 == 0 else ["points", "tree"]
        for key in order:
            wp.record_event(begin)
            wp.capture_launch(graphs[key])
            wp.record_event(end)
            wp.synchronize_event(end)
            samples[key].append(wp.get_event_elapsed_time(begin, end))
            errors[key].append(int(error.numpy()[0]))
    assert errors["tree"] == errors["points"]
    return dict(
        n=len(p),
        tetrahedra=len(t),
        samples_ms=samples,
        median_ms={k: float(np.median(v)) for k, v in samples.items()},
        errors=errors,
        cpu=cpu_quota_delta(quota, cpu_quota_snapshot()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    rng = np.random.default_rng(2026)
    cases = {
        f"random{n}": rng.uniform(-1, 1, (n, 3)).astype(np.float32) for n in [128, 512, 1024, 2048]
    }
    cases.update(
        {
            f"grid{side**3}": np.array(list(product(range(side), repeat=3)), np.float32)
            for side in [6, 8, 10]
        }
    )
    results = {}
    for name, points in cases.items():
        results[name] = measure(points, args.repeats)
        print(name, results[name]["median_ms"], flush=True)
    report = dict(
        scope="Matched fixed-buffer global sphere-kernel comparison only, with alternating order. Positive SciPy candidate cells are used as input; lattice tie choices need not agree with Warpower, and matching nonzero rejection is expected there. No local certificate, construction, topology checks, setup or readback in timed kernel intervals. This is not a full geometry or CPU performance comparison.",
        cases=results,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
