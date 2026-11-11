"""Record actual full-budget VTM reconstruction calls, outside timing runs.

The captured CUDA graph writes coordinates and producer state on device. There
is no host synchronization inside the solve and no invented displacement field.
"""

import argparse
import hashlib
import inspect
import json
import subprocess
from pathlib import Path

import numpy as np
import warp as wp

from benchmarks.replay.fixture import Fixture, Writer, digest
from examples.vtm_inputs import surface, write_off
from warpower import SizingField, SurfaceDomain, VariationalMeshingWorkspace


@wp.kernel
def snapshot(
    points: wp.array[wp.vec3],
    active: wp.array[int],
    mode: wp.array[int],
    alpha: wp.array[wp.float64],
    source: int,
    frames: wp.array2d[wp.vec3],
    metadata: wp.array2d[int],
    alphas: wp.array[wp.float64],
    cursor: wp.array[int],
):
    i = wp.tid()
    at = cursor[0]
    if active[0] == points.shape[0] and at < frames.shape[0]:
        frames[at, i] = points[i]
        if i == 0:
            metadata[at, 0] = mode[0]
            metadata[at, 1] = source
            alphas[at] = alpha[0]


@wp.kernel
def result(
    active: wp.array[int],
    n: int,
    tets: wp.array2d[int],
    tet_count: wp.array[int],
    error: wp.array[int],
    initial_tets: wp.array2d[int],
    metadata: wp.array2d[int],
    cursor: wp.array[int],
):
    i = wp.tid()
    at = cursor[0]
    if active[0] == n and at < metadata.shape[0]:
        if i == 0:
            metadata[at, 2] = error[0]
            metadata[at, 3] = tet_count[0]
        if at == 0 and i < tet_count[0]:
            for k in range(4):
                initial_tets[i, k] = tets[i, k]


@wp.kernel
def advance(active: wp.array[int], n: int, cursor: wp.array[int]):
    if active[0] == n:
        cursor[0] += 1


def select_transitions(frames, stages):
    """Select before measuring: early/late, motion quantiles and phase changes."""
    motion = np.max(np.linalg.norm(np.diff(frames.astype(np.float64), axis=0), axis=2), axis=1)
    order = np.argsort(motion, kind="stable")
    chosen = {1, min(2, len(frames) - 1), len(frames) - 1}
    chosen.update(int(order[k]) + 1 for k in [0, len(order) // 2, len(order) - 1])
    chosen.update(i for i in range(1, len(stages)) if stages[i] != stages[i - 1])
    return sorted(chosen), motion


def capture(args):
    points, faces = surface(args.case, args.input)
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
    if args.case.startswith("graded-"):
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
        args.count,
        initial_count=256,
        seed=args.seed,
        sizing=sizing,
    )
    core = mesh.core
    frames = wp.empty((args.max_frames, args.count), dtype=wp.vec3, device=mesh.device)
    metadata = wp.zeros((args.max_frames, 4), dtype=int, device=mesh.device)
    alphas = wp.zeros(args.max_frames, dtype=wp.float64, device=mesh.device)
    cursor = wp.zeros(1, dtype=int, device=mesh.device)
    initial_tets = wp.empty_like(core.full_tetrahedra)
    original = core._reconstruct
    sources = []

    def reconstruct():
        caller = inspect.currentframe().f_back.f_code
        label = f"{Path(caller.co_filename).name}:{caller.co_name}"
        if label not in sources:
            sources.append(label)
        wp.launch(
            snapshot,
            args.count,
            inputs=[
                core.points,
                core.active_count,
                mesh._mode,
                core._alpha,
                sources.index(label),
                frames,
                metadata,
                alphas,
                cursor,
            ],
        )
        original()
        wp.launch(
            result,
            len(initial_tets),
            inputs=[
                core.active_count,
                args.count,
                core.full_tetrahedra,
                core.full_tetrahedra_count,
                core.mesh.error,
                initial_tets,
                metadata,
                cursor,
            ],
        )
        wp.launch(advance, 1, inputs=[core.active_count, args.count, cursor])

    core._reconstruct = reconstruct
    try:
        with wp.ScopedCapture() as captured:
            domain.refit()
            if sizing is not None:
                sizing.refit()
            mesh.run()
        wp.capture_launch(captured.graph)
        wp.synchronize()
    finally:
        core._reconstruct = original
    count = int(cursor.numpy()[0])
    if not 1 < count <= args.max_frames:
        raise RuntimeError(f"Need 2..{args.max_frames} full-budget calls, observed {count}")
    values, meta, alpha = frames.numpy()[:count], metadata.numpy()[:count], alphas.numpy()[:count]
    stages = ["quality" if row[0] else "fitting" for row in meta]
    selected, motion = select_transitions(values, stages)
    writer = Writer(
        args.output,
        name=f"vtm-{args.case}-{args.count}-seed{args.seed}",
        provenance=dict(
            producer="warpower.VariationalMeshingWorkspace",
            case=args.case,
            count=args.count,
            seed=args.seed,
            initial_count=256,
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            source_sha256={
                str(p): digest(p) for p in [Path(__file__), *sorted(Path("warpower").glob("*.py"))]
            },
            input_arrays_sha256=hashlib.sha256(points.tobytes() + faces.tobytes()).hexdigest(),
            final_error=int(mesh.error.numpy()[0]),
            final_valid=int(mesh.valid.numpy()[0]),
            final_counts=mesh.counts.numpy().tolist(),
            final_steps=mesh.steps.numpy().tolist(),
        ),
        limitations=[
            "Actual consecutive full-budget reconstruction inputs; earlier growth is excluded.",
            "Site IDs are fixed full-budget slots; accepted/rejected optimization outcomes are not inferred.",
            "Sequence replay carries each backend's own retained state between these input calls; it does not inject producer rollbacks.",
            "Instrumentation is outside performance measurement. Procedural examples are not paper assets.",
        ],
    )
    ids = np.arange(args.count, dtype=np.int64)
    for i, frame in enumerate(values):
        writer.append(
            frame,
            ids,
            event="initial" if i == 0 else "reconstruction-call",
            stage=stages[i],
            reset=i == 0,
            source_step=i,
            metadata=dict(
                callsite=sources[meta[i, 1]],
                line_search_alpha=float(alpha[i]),
                producer_geometry_error=int(meta[i, 2]),
                producer_tetrahedra=int(meta[i, 3]),
            ),
        )
    write_off(args.output / "input.off", points, faces)
    np.savez_compressed(
        args.output / "initial_mesh.npz",
        points=values[0],
        site_ids=ids,
        tetrahedra=initial_tets.numpy()[: meta[0, 3]],
        error=meta[0, 2],
    )
    writer.document["provenance"]["initial_mesh"] = dict(
        path="initial_mesh.npz",
        sha256=digest(args.output / "initial_mesh.npz"),
        meaning="Producer full-hull output for frame 0, local indices into its points/site_ids",
    )
    writer.finalize()
    summary = dict(
        frames=count,
        transitions=selected,
        maximum_displacements=motion.tolist(),
        selection="Before timing: first two, last, minimum/median/maximum motion, and phase boundaries",
        validation=Fixture(args.output).validate(),
        final_error=int(mesh.error.numpy()[0]),
        final_valid=int(mesh.valid.numpy()[0]),
    )
    (args.output / "selection.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=["torus", "graded-torus", "thin-hand", "surface"], required=True
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-frames", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.count < 256 or args.max_frames < 2:
        parser.error("count must be >=256 and max-frames >=2")
    capture(args)


if __name__ == "__main__":
    main()
