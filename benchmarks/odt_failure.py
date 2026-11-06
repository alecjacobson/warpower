"""Save the last rejected construction input for independent diagnosis."""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

import warpower


@wp.kernel
def save_failed_input(
    points: wp.array[wp.vec3],
    status: wp.array[int],
    error: wp.array[int],
    saved: wp.array[wp.vec3],
    saved_status: wp.array[int],
    seen: wp.array[int],
):
    i = wp.tid()
    if error[0] != 0:
        saved[i] = points[i]
        saved_status[i] = status[i]
        if i == 0:
            seen[0] = error[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = np.random.default_rng(args.seed).uniform(-1, 1, (args.n, 3)).astype(np.float32)
    workspace = warpower.ODTWorkspace(source, max_iterations=args.iterations, max_backtracks=0)
    saved = wp.zeros_like(workspace.points)
    status = wp.zeros(args.n, dtype=int, device=workspace.device)
    seen = wp.zeros(1, dtype=int, device=workspace.device)
    original_run = workspace.mesh.run

    def instrumented_run(stream=None):
        result = original_run(stream)
        wp.launch(
            save_failed_input,
            args.n,
            inputs=[
                workspace.points,
                workspace.mesh._tet_status,
                workspace.mesh.error,
                saved,
                status,
                seen,
            ],
            device=workspace.device,
        )
        return result

    workspace.mesh.run = instrumented_run
    with wp.ScopedCapture() as capture:
        workspace.run()
    wp.capture_launch(capture.graph)
    failure = int(seen.numpy()[0])
    record = {
        "n": args.n,
        "seed": args.seed,
        "requested_iterations": args.iterations,
        "accepted_iterations": int(workspace.iterations.numpy()[0]),
        "error": int(workspace.error.numpy()[0]),
        "captured_geometry_error": failure,
        "last_geometry_error": int(workspace.last_geometry_error.numpy()[0]),
        "failed_sites": np.flatnonzero(status.numpy()).tolist() if failure else [],
    }
    arrays = {"last_valid_points": workspace.points.numpy()}
    if failure:
        arrays.update(points=saved.numpy(), status=status.numpy())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    args.output.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
