"""Audit changing-input producer/recovery/consumer composition on the fitted hand."""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

from warpower import DelaunayWorkspace

NAN = wp.constant(float("nan"))
wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.kernel
def produce(source: wp.array[wp.vec3], points: wp.array[wp.vec3], phase: wp.array[int]):
    i = wp.tid()
    points[i] = source[i] * wp.where(phase[0] == 1, 2.0, 1.0)
    if phase[0] == 2 and i == 0:
        points[i] = wp.vec3(NAN, 0.0, 0.0)


@wp.kernel
def consume(
    phase: wp.array[int],
    tets: wp.array2d[int],
    count: wp.array[int],
    error: wp.array[int],
    attempts: wp.array[int],
    level: wp.array[int],
    records: wp.array2d[wp.int64],
):
    row = phase[0]
    records[row, 0] = wp.int64(count[0])
    records[row, 1] = wp.int64(error[0])
    records[row, 2] = wp.int64(attempts[0])
    records[row, 3] = wp.int64(level[0])
    total = wp.int64(0)
    for i in range(count[0]):
        for k in range(4):
            total += wp.int64(tets[i, k])
    records[row, 4] = total
    phase[0] = (row + 1) % 4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dot", type=Path)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    points = np.load(Path(__file__).parents[1] / "tests/data/hand-fit256.npz")["points"]
    mesh = DelaunayWorkspace(points)
    source = wp.array(points, dtype=wp.vec3, device=mesh.device)
    phase = wp.zeros(1, dtype=int, device=mesh.device)
    records = wp.zeros((4, 5), dtype=wp.int64, device=mesh.device)
    wp.load_module(module=__name__, device=mesh.device)
    with wp.ScopedCapture() as cap:
        wp.launch(produce, len(points), inputs=[source, mesh.points, phase])
        mesh.run()
        wp.launch(
            consume,
            1,
            inputs=[
                phase,
                mesh.tetrahedra,
                mesh.tetrahedra_count,
                mesh.error,
                mesh.attempts,
                mesh.auxiliary_level,
                records,
            ],
        )
    if args.dot:
        wp.capture_debug_dot_print(cap.graph, str(args.dot), verbose=True)
    if args.profile:
        import torch

        torch.cuda.cudart().cudaProfilerStart()
    for _ in range(4):
        wp.capture_launch(cap.graph)
    wp.synchronize()
    if args.profile:
        torch.cuda.cudart().cudaProfilerStop()
    values = records.numpy()
    np.testing.assert_array_equal(values[:, 1], [0, 0, 1, 0])
    np.testing.assert_array_equal(values[:, 2], [5, 1, 0, 5])
    np.testing.assert_array_equal(values[:, 3], [4, 4, 0, 4])
    np.testing.assert_array_equal(values[[0, 1, 3], 0], 1430)
    np.testing.assert_array_equal(values[[0, 1, 3], 4], values[0, 4])
    assert values[2, 0] == values[2, 4] == 0
    result = dict(
        scope="Four captured producer/reconstruction/consumer replays: original fitted hand, exact scale by two, invalid input, restored original. No host readback between replays; not a throughput measurement.",
        columns=["tetrahedra", "error", "attempts", "auxiliary_level", "checksum"],
        records=values.tolist(),
        next_phase=int(phase.numpy()[0]),
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
