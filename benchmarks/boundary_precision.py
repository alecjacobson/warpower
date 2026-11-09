"""Paired boundary precision timings and offline numerical-error diagnostics."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import warp as wp
from scipy.spatial import cKDTree

from benchmarks.run import cpu_quota_delta, cpu_quota_snapshot
from examples.vtm_inputs import read_off
from warpower import BoundaryWorkspace, SizingField, SizingQueryWorkspace, SurfaceDomain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Triangular surface OFF")
    parser.add_argument("--sites", type=Path, required=True, help="NPZ mesh or NPY trajectory")
    parser.add_argument("--graded", action="store_true")
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    surface, faces = read_off(args.input)
    domain = SurfaceDomain(
        surface,
        faces,
        feature_angle=120,
        surface_subdivisions=1,
        crease_subdivisions=1,
        query_backend="native",
        validate_geometry=False,
    )
    source = np.load(args.sites)
    if isinstance(source, np.lib.npyio.NpzFile):
        frames = source["points"][None]
        source.close()
    else:
        frames = source
    indices = sorted(set([0, min(2, len(frames) - 1), len(frames) // 2, len(frames) - 1]))
    points = wp.array(frames[0], dtype=wp.vec3, device=domain.device)
    options = {}
    if args.graded:
        lower, upper = surface.min(axis=0), surface.max(axis=0)
        anchor = (lower + upper) / 2
        anchor[0] = lower[0]
        field = SizingField(
            wp.array(anchor[None], dtype=wp.vec3, device=domain.device),
            wp.array([0.35], dtype=float, device=domain.device),
            gradation=0.65 / float(upper[0] - lower[0]),
            minimum=0.35,
            maximum=1.0,
        )
        sq = SizingQueryWorkspace(field, domain.surface_points).run()
        cq = SizingQueryWorkspace(field, domain.crease_points).run()
        options = dict(surface_sizing=sq.values, crease_sizing=cq.values)
    modes = [
        ("float64", "float64"),
        ("float32", "float64"),
        ("float64", "float32"),
        ("float32", "float32"),
    ]
    fits = [
        BoundaryWorkspace(domain, points, query_precision=q, accumulation_precision=a, **options)
        for q, a in modes
    ]
    first, last = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
    report = dict(
        input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        sites_sha256=hashlib.sha256(args.sites.read_bytes()).hexdigest(),
        graded=args.graded,
        runtime_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path("warpower").glob("*.py"))
        },
        scope="Paired complete BoundaryWorkspace.run, excluding setup/transfers/error audit; offline reference diagnostics have no production cost.",
        records=[],
    )
    samples = domain.surface_points.numpy().astype(float)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index in indices:
        positions = np.ascontiguousarray(frames[index], dtype=np.float32)
        points.assign(positions)
        times = [[] for _ in fits]
        for fit in fits:
            fit.run()
        wp.synchronize()
        quota = cpu_quota_snapshot()
        for repeat in range(args.repeats):
            for k in range(4) if repeat % 2 == 0 else reversed(range(4)):
                wp.record_event(first)
                fits[k].run()
                wp.record_event(last)
                wp.synchronize_event(last)
                times[k].append(wp.get_event_elapsed_time(first, last))
        cpu = cpu_quota_delta(quota, cpu_quota_snapshot())
        reference = fits[0]
        owners = reference.surface_owners.numpy()
        targets = reference.targets.numpy().astype(float)
        masses = reference.surface_mass.numpy()
        spacing = cKDTree(positions).query(positions, k=2)[0][:, 1]
        rows = []
        for k, (fit, (query, accumulation)) in enumerate(zip(fits, modes)):
            current = fit.surface_owners.numpy()
            changed = np.flatnonzero(current != owners)
            assert fit.error.numpy()[0] == 0
            excess = np.zeros(0)
            if len(changed):
                exact = np.sum(
                    (samples[changed] - positions[owners[changed]].astype(float)) ** 2, axis=1
                )
                chosen = np.sum(
                    (samples[changed] - positions[current[changed]].astype(float)) ** 2, axis=1
                )
                excess = chosen - exact
            delta = np.linalg.norm(fit.targets.numpy().astype(float) - targets, axis=1)
            mass_delta = np.abs(fit.surface_mass.numpy() - masses)
            rows.append(
                dict(
                    query=query,
                    accumulation=accumulation,
                    samples_ms=times[k],
                    median_ms=float(np.median(times[k])),
                    changed_surface_owners=len(changed),
                    maximum_squared_distance_excess=float(np.max(excess, initial=0)),
                    maximum_target_distance=float(delta.max()),
                    maximum_target_distance_over_spacing=float(
                        np.max(delta[spacing > 0] / spacing[spacing > 0], initial=0)
                    ),
                    maximum_relative_mass_error=float(
                        np.max(mass_delta[masses > 0] / masses[masses > 0], initial=0)
                    ),
                    relative_surface_energy_error=float(
                        fit.energy.numpy()[0] / reference.energy.numpy()[0] - 1
                    ),
                    added_accumulation_bytes=32 * len(points) if accumulation == "float32" else 0,
                )
            )
        row = dict(frame=index, n=len(positions), cpu=cpu, modes=rows)
        report["records"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
