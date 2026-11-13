"""Alternating complete VTM solves with and without same-input hull proofs."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

from benchmarks.replay.backends import canonical_rows
from benchmarks.run import cpu_quota_delta, cpu_quota_snapshot
from examples.vtm_inputs import surface
from warpower import SizingField, SurfaceDomain, VariationalMeshingWorkspace
from warpower._reuse_delaunay import ReuseDelaunayWorkspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["torus", "graded-torus"], required=True)
    parser.add_argument("--count", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    points, faces = surface(args.case)
    workspaces, graphs, setup = [], [], []
    for enabled in [False, True]:
        started = time.perf_counter()
        ReuseDelaunayWorkspace._cache_hull_support = enabled
        domain = SurfaceDomain(
            points,
            faces,
            feature_angle=120,
            surface_subdivisions=1,
            crease_subdivisions=1,
            query_backend="native",
            validate_geometry=False,
        )
        sizing = None
        if args.case == "graded-torus":
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
            domain, args.count, initial_count=256, seed=2026, sizing=sizing
        )
        with wp.ScopedCapture() as capture:
            domain.refit()
            if sizing is not None:
                sizing.refit()
            mesh.run()
        wp.synchronize()
        setup.append(time.perf_counter() - started)
        workspaces.append(mesh)
        graphs.append(capture.graph)
    rows = []
    for repeat in range(-1, args.repeats):
        for variant in [0, 1] if repeat % 2 else [1, 0]:
            before = cpu_quota_snapshot()
            start = time.perf_counter()
            wp.capture_launch(graphs[variant])
            wp.synchronize()
            elapsed = time.perf_counter() - start
            quota = cpu_quota_delta(before, cpu_quota_snapshot())
            mesh = workspaces[variant]
            row = dict(
                repeat=repeat,
                variant=variant,
                seconds=elapsed,
                cpu_quota=quota,
                error=int(mesh.error.numpy()[0]),
                valid=int(mesh.valid.numpy()[0]),
                counts=mesh.counts.numpy().tolist(),
                steps=mesh.steps.numpy().tolist(),
                rounds=mesh.rounds.numpy().tolist(),
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
            if row["error"] or not row["valid"]:
                raise RuntimeError("Complete mesher failed")
    saved = []
    for variant, mesh in enumerate(workspaces):
        data = dict(
            points=mesh.core.points.numpy(),
            full_tetrahedra=mesh.core.full_tetrahedra.numpy()[
                : int(mesh.core.full_tetrahedra_count.numpy()[0])
            ],
            roles=mesh.core.boundary.roles.numpy(),
        )
        np.savez_compressed(args.output / f"variant-{variant}.npz", **data)
        saved.append(data)
    a, b = saved
    report = dict(
        case=args.case,
        count=args.count,
        setup_seconds=setup,
        records=rows,
        median_ms=[
            float(np.median([r["seconds"] for r in rows if r["repeat"] >= 0 and r["variant"] == v]))
            * 1000
            for v in [0, 1]
        ],
        points_bitwise_equal=bool(np.array_equal(a["points"], b["points"])),
        maximum_coordinate_difference=float(
            np.max(np.abs(a["points"].astype(float) - b["points"]))
        ),
        tetrahedra_equal=bool(
            np.array_equal(
                canonical_rows(a["full_tetrahedra"]), canonical_rows(b["full_tetrahedra"])
            )
        ),
        roles_equal=bool(np.array_equal(a["roles"], b["roles"])),
        scope="Complete captured fresh sampling, fitting, growth, quality, extraction and diagnostics. Allocation/JIT/setup and result downloads excluded; synchronization included.",
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["tetrahedra_equal"] or not report["roles_equal"]:
        raise RuntimeError("Integration outputs differ; investigate before adoption")


if __name__ == "__main__":
    main()
