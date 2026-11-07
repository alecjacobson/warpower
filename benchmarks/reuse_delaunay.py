"""Audit fragment reuse and full fallback in a resident producer/consumer loop."""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

from warpower._reuse_delaunay import ReuseDelaunayWorkspace

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.kernel
def produce(frames: wp.array2d[wp.vec3], phase: wp.array[int], points: wp.array[wp.vec3]):
    i = wp.tid()
    points[i] = frames[phase[0], i]


@wp.kernel
def consume(
    phase: wp.array[int],
    tets: wp.array2d[int],
    count: wp.array[int],
    error: wp.array[int],
    attempted: wp.array[int],
    accepted: wp.array[int],
    rounds: wp.array[int],
    base_attempted: wp.array[int],
    seed_count: wp.array[int],
    proofs: wp.array[int],
    face_proofs: wp.array[int],
    records: wp.array2d[wp.int64],
):
    row = phase[0]
    records[row, 0] = wp.int64(count[0])
    records[row, 1] = wp.int64(error[0])
    records[row, 2] = wp.int64(attempted[0])
    records[row, 3] = wp.int64(accepted[0])
    records[row, 4] = wp.int64(rounds[0])
    records[row, 5] = wp.int64(base_attempted[0])
    checksum = wp.int64(0)
    for i in range(count[0]):
        for k in range(4):
            checksum += wp.int64(tets[i, k])
    records[row, 6] = checksum
    records[row, 7] = wp.int64(0)
    records[row, 8] = wp.int64(0)
    records[row, 9] = wp.int64(0)
    records[row, 10] = wp.int64(0)
    if seed_count[0] > 0:
        records[row, 7] = wp.int64(proofs[0])
        records[row, 8] = wp.int64(proofs[1])
        records[row, 9] = wp.int64(face_proofs[0])
        records[row, 10] = wp.int64(face_proofs[1])
    phase[0] = (row + 1) % records.shape[0]


def audit(profile=False, dot=None):
    rng = np.random.default_rng(930)
    initial = rng.normal(size=(64, 3)).astype(np.float32)
    nearby = initial.copy()
    nearby[5] += np.float32(1e-5)
    moved = initial + rng.normal(0, 0.05, initial.shape).astype(np.float32)
    duplicate = moved.copy()
    duplicate[32:] = duplicate[:32]
    invalid = initial.copy()
    invalid[4, 0] = np.nan
    frames = np.stack([initial, nearby, moved, duplicate, invalid, initial])
    mesh = ReuseDelaunayWorkspace(initial)
    source = wp.array(frames, dtype=wp.vec3, device=mesh.device)
    phase = wp.zeros(1, dtype=int, device=mesh.device)
    records = wp.zeros((len(frames), 11), dtype=wp.int64, device=mesh.device)
    wp.load_module(module=__name__, device=mesh.device)
    with wp.ScopedCapture() as cap:
        wp.launch(produce, mesh.n, inputs=[source, phase, mesh.points])
        mesh.run()
        wp.launch(
            consume,
            1,
            inputs=[
                phase,
                mesh.tetrahedra,
                mesh.tetrahedra_count,
                mesh.error,
                mesh.reuse_attempted,
                mesh.reuse_accepted,
                mesh.reuse_rounds,
                mesh.base_attempted,
                mesh.reuse_seeds,
                mesh._reuse_completion._certificate.proof_statistics,
                mesh._reuse_completion._certificate.face_statistics,
                records,
            ],
        )
    if dot:
        wp.capture_debug_dot_print(cap.graph, str(dot), verbose=True)
    for _ in frames:
        wp.capture_launch(cap.graph)
    wp.synchronize()
    mesh.clear_reuse()
    wp.synchronize()
    if profile:
        import torch

        torch.cuda.cudart().cudaProfilerStart()
    for _ in frames:
        wp.capture_launch(cap.graph)
    wp.synchronize()
    if profile:
        torch.cuda.cudart().cudaProfilerStop()
    values = records.numpy()
    np.testing.assert_array_equal(values[:, 1], [0, 0, 0, 0, 1, 0])
    assert values[1, 3] == 1 and values[1, 5] == 0
    assert values[2, 3] == 1 and values[2, 4] > 0
    assert values[1, 7] == values[1, 0] and values[1, 8] == 0
    assert values[2, 7] > 0 and values[2, 8] > 0
    assert values[2, 9] > 0 and values[2, 10] > 0
    np.testing.assert_array_equal(values[1, 9:], 0)
    np.testing.assert_array_equal(values[[0, 4, 5], 7:], 0)
    assert values[4, 0] == values[4, 6] == 0
    assert values[5, 2] == 0 and values[5, 5] == 1
    np.testing.assert_array_equal(values[0, [0, 6]], values[5, [0, 6]])
    assert phase.numpy()[0] == 0
    return dict(
        scope="Six warmed resident producer/reconstruction/consumer phases: fresh random64, nearby unchanged topology, moved sites requiring facet completion, duplicate sites, nonfinite input and restored random64. Device phase advance and one final synchronization; setup, warmup and readback excluded from native trace. Internal VTM strategy audit, not a throughput measurement.",
        columns=[
            "tetrahedra",
            "error",
            "reuse_attempted",
            "reuse_accepted",
            "completion_rounds",
            "base_attempted",
            "checksum",
            "cell_proofs_reused_within_completion",
            "cells_globally_checked_within_completion",
            "supported_faces_skipping_candidate_search",
            "unsupported_faces_requiring_candidate_search",
        ],
        records=values.tolist(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dot", type=Path)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.dot:
        args.dot.parent.mkdir(parents=True, exist_ok=True)
    result = audit(args.profile, args.dot)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
