"""Record actual 8K Armadillo VTM inputs and compare reconstruction warm starts."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import warp as wp
from scipy.spatial import Delaunay

from benchmarks._neighbor_clipping import install
from benchmarks.run import cpu_quota_delta, cpu_quota_snapshot
from benchmarks.tetrahedra import canonical
from examples.vtm_inputs import surface
from warpower import DelaunayWorkspace, SurfaceDomain, VariationalMeshingWorkspace
from warpower._wall_reuse import WallReuseDelaunayWorkspace


@wp.kernel
def save(
    points: wp.array[wp.vec3],
    active: wp.array[int],
    frames: wp.array2d[wp.vec3],
    count: wp.array[int],
):
    i = wp.tid()
    if active[0] == points.shape[0] and count[0] < frames.shape[0]:
        frames[count[0], i] = points[i]


@wp.kernel
def advance(active: wp.array[int], n: int, count: wp.array[int]):
    if active[0] == n:
        count[0] += 1


def record_trials(input_path, out):
    p, f = surface("surface", input_path)
    domain = SurfaceDomain(
        p,
        f,
        feature_angle=120,
        surface_subdivisions=1,
        crease_subdivisions=1,
        query_backend="native",
        validate_geometry=False,
    )
    mesh = VariationalMeshingWorkspace(domain, 8192, initial_count=256, seed=2026)
    frames = wp.empty((512, 8192), dtype=wp.vec3, device=mesh.device)
    count = wp.zeros(1, dtype=int, device=mesh.device)
    core = mesh.core
    original = core._reconstruct

    def reconstruct():
        wp.launch(save, 8192, inputs=[core.points, core.active_count, frames, count])
        wp.launch(advance, 1, inputs=[core.active_count, 8192, count])
        original()

    core._reconstruct = reconstruct
    with wp.ScopedCapture() as capture:
        domain.refit()
        mesh.run()
    wp.capture_launch(capture.graph)
    wp.synchronize()
    n = int(count.numpy()[0])
    assert 1 < n <= len(frames), n
    assert mesh.error.numpy()[0] == 0 and mesh.valid.numpy()[0] == 1
    values = frames.numpy()[:n]
    np.save(out / "frames.npy", values)
    metadata = dict(
        source_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
        frames=n,
        shape=values.shape,
        frame_bytes_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
        mesher_counts=mesh.counts.numpy().tolist(),
        error=mesh.error.numpy().tolist(),
        steps=mesh.steps.numpy().tolist(),
        rounds=mesh.rounds.numpy().tolist(),
        scope="Actual full-budget VTM reconstruction inputs, including rejected line-search trials; one fresh native-query solve. Instrumented capture is not a performance measurement.",
    )
    (out / "record.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)


def measure_trials(out):
    frames = np.load(out / "frames.npy")
    motion = np.max(np.linalg.norm(np.diff(frames, axis=0).astype(np.float64), axis=2), axis=1)
    order = np.argsort(motion, kind="stable")
    indices = sorted(
        set(
            [
                1,
                2,
                len(frames) - 1,
                *[int(order[k]) + 1 for k in [0, len(order) // 2, len(order) - 1]],
            ]
        )
    )
    source = wp.array(frames, dtype=wp.vec3, device="cuda:0")
    workspaces = [
        DelaunayWorkspace(frames[0]),
        DelaunayWorkspace(frames[0]),
        WallReuseDelaunayWorkspace(frames[0]),
    ]
    from benchmarks._cell_continuation import original_schedule

    for mesh in workspaces:
        original_schedule(mesh)
    cache = install(workspaces[1], starting_capacity=256)
    names = ["fresh", "neighbors-larger", "fragment"]
    begin, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
    report = dict(
        frame_selection="First two transitions, last transition, and minimum/median/maximum maximum-site displacement transitions.",
        frame_indices=indices,
        source_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),
        records=[],
        runtime_source_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                Path(__file__).with_name("_neighbor_clipping.py"),
                *sorted(Path("warpower").glob("*.py")),
            ]
        },
    )
    for index in indices:
        samples = [[], [], []]
        quota = cpu_quota_snapshot()
        for repeat in range(6):
            for which in [0, 1, 2] if repeat % 2 == 0 else [2, 1, 0]:
                mesh = workspaces[which]
                wp.copy(mesh.points, source[index - 1])
                mesh._preferred.zero_()
                if which == 1:
                    cache.clear()
                if which == 2:
                    mesh.clear_reuse()
                mesh.run()
                wp.copy(mesh.points, source[index])
                wp.record_event(begin)
                mesh.run()
                wp.record_event(end)
                wp.synchronize_event(end)
                if repeat:
                    samples[which].append(wp.get_event_elapsed_time(begin, end))
        cpu = cpu_quota_delta(quota, cpu_quota_snapshot())
        expected = canonical(Delaunay(frames[index].astype(np.float64)).simplices)
        values = []
        for which, mesh in enumerate(workspaces):
            count = int(mesh.tetrahedra_count.numpy()[0])
            values.append(
                dict(
                    strategy=names[which],
                    samples_ms=samples[which],
                    median_ms=float(np.median(samples[which])),
                    error=mesh.error.numpy().tolist(),
                    base_error=mesh.base_error.numpy().tolist(),
                    tetrahedra=count,
                    scipy_equal=bool(
                        np.array_equal(canonical(mesh.tetrahedra.numpy()[:count]), expected)
                    ),
                )
            )
        values[-1].update(
            reuse_accepted=workspaces[-1].reuse_accepted.numpy().tolist(),
            reuse_seeds=workspaces[-1].reuse_seeds.numpy().tolist(),
            reuse_rounds=workspaces[-1].reuse_rounds.numpy().tolist(),
        )
        row = dict(
            frame=index, maximum_displacement=float(motion[index - 1]), cpu=cpu, values=values
        )
        report["records"].append(row)
        (out / "paired.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--record-input",
        type=Path,
        help="Record a fresh uniform 8K solve of this OFF mesh before measuring",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.record_input:
        record_trials(args.record_input, args.output)
    measure_trials(args.output)


if __name__ == "__main__":
    main()
