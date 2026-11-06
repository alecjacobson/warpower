"""Paired complete torus meshing comparison for generation cache reset."""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

import warpower
from benchmarks.mesher import surface
from benchmarks.recovery_policy import HistoricalMesher
from benchmarks.run import cpu_quota_delta, cpu_quota_snapshot
from tests.test_mesher import validate
from tests.test_tetrahedra import tet_set


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    domain = warpower.SurfaceDomain(*surface("torus"), surface_subdivisions=4)
    meshes = [
        cls(domain, 1000, initial_count=256)
        for cls in (HistoricalMesher, warpower.VariationalMeshingWorkspace)
    ]
    graphs = []
    for mesh in meshes:
        with wp.ScopedCapture() as cap:
            domain.refit()
            mesh.run()
        graphs.append(cap.graph)
        for _ in range(2):
            wp.capture_launch(cap.graph)
    wp.synchronize()
    begin, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
    samples = [[], []]
    quota = cpu_quota_snapshot()
    for repeat in range(args.repeats):
        for i in [0, 1] if repeat % 2 == 0 else [1, 0]:
            wp.record_event(begin)
            wp.capture_launch(graphs[i])
            wp.record_event(end)
            wp.synchronize_event(end)
            samples[i].append(wp.get_event_elapsed_time(begin, end))
    cpu = cpu_quota_delta(quota, cpu_quota_snapshot())
    for mesh in meshes:
        validate(mesh)
    np.testing.assert_array_equal(meshes[0].counts.numpy(), meshes[1].counts.numpy())
    np.testing.assert_array_equal(meshes[0].points.numpy(), meshes[1].points.numpy())
    for cells, count in [
        ("tetrahedra", "tetrahedra_count"),
        ("full_tetrahedra", "full_tetrahedra_count"),
    ]:
        n = int(getattr(meshes[0], count).numpy()[0])
        assert tet_set(getattr(meshes[0], cells).numpy()[:n]) == tet_set(
            getattr(meshes[1], cells).numpy()[:n]
        )
    report = dict(
        scope="Paired complete resident 1000-site torus generation, including domain refit. Alternate historical cached and generation-reset graphs, separately warmed; no setup, upload, or validation in timing.",
        device=domain.device.name,
        warp=wp.__version__,
        repeats=args.repeats,
        historical_samples_ms=samples[0],
        reset_samples_ms=samples[1],
        historical_median_ms=float(np.median(samples[0])),
        reset_median_ms=float(np.median(samples[1])),
        cpu_quota_delta=cpu,
        cpu_throttled=any(v["throttled_periods"] for v in cpu.values()),
        counts=meshes[1].counts.numpy().tolist(),
        metrics=[mesh.metrics.numpy().tolist() for mesh in meshes],
        independent_validation=True,
        identical_points_and_canonical_full_and_retained_tetrahedra=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
