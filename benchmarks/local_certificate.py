"""Audit local/global certificate changes inside one resident producer/consumer graph."""

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import warp as wp

from warpower import DelaunayWorkspace

NAN = wp.constant(float("nan"))
wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.kernel
def produce(
    random: wp.array[wp.vec3],
    cube: wp.array[wp.vec3],
    points: wp.array[wp.vec3],
    phase: wp.array[int],
):
    i = wp.tid()
    points[i] = random[i]
    if phase[0] == 1:
        points[i] = cube[i % 8]
    elif phase[0] == 2 and i == 0:
        points[i] = wp.vec3(NAN, 0.0, 0.0)


@wp.kernel
def consume(
    phase: wp.array[int],
    tets: wp.array2d[int],
    count: wp.array[int],
    error: wp.array[int],
    local: wp.array[int],
    records: wp.array2d[wp.int64],
):
    row = phase[0]
    records[row, 0] = wp.int64(count[0])
    records[row, 1] = wp.int64(error[0])
    records[row, 2] = wp.int64(wp.where(error[0] == 0, local[0], -1))
    total = wp.int64(0)
    for i in range(count[0]):
        for k in range(4):
            total += wp.int64(tets[i, k])
    records[row, 3] = total
    phase[0] = (row + 1) % 4


def audit(profile=False, dot=None):
    points = np.random.default_rng(541).normal(size=(200, 3)).astype(np.float32)
    mesh = DelaunayWorkspace(points)
    source = wp.array(points, dtype=wp.vec3, device=mesh.device)
    cube = wp.array(list(product([-1.0, 1.0], repeat=3)), dtype=wp.vec3, device=mesh.device)
    phase = wp.zeros(1, dtype=int, device=mesh.device)
    records = wp.zeros((4, 4), dtype=wp.int64, device=mesh.device)
    wp.load_module(module=__name__, device=mesh.device)
    with wp.ScopedCapture() as cap:
        wp.launch(produce, len(points), inputs=[source, cube, mesh.points, phase])
        mesh.run()
        wp.launch(
            consume,
            1,
            inputs=[
                phase,
                mesh.tetrahedra,
                mesh.tetrahedra_count,
                mesh.error,
                mesh._certificate._local,
                records,
            ],
        )
    if dot:
        wp.capture_debug_dot_print(cap.graph, str(dot), verbose=True)
    # Instantiate/upload and warm every branch before the measured native trace.
    for _ in range(4):
        wp.capture_launch(cap.graph)
    wp.synchronize()
    if profile:
        import torch

        torch.cuda.cudart().cudaProfilerStart()
    for _ in range(4):
        wp.capture_launch(cap.graph)
    wp.synchronize()
    if profile:
        torch.cuda.cudart().cudaProfilerStop()
    values = records.numpy()
    np.testing.assert_array_equal(values[:, 1], [0, 0, 1, 0])
    np.testing.assert_array_equal(values[:, 2], [1, 0, -1, 1])
    assert values[0, 0] > 1000
    assert values[1, 0] == 6
    np.testing.assert_array_equal(values[0], values[3])
    assert values[2, 0] == values[2, 3] == 0
    assert phase.numpy()[0] == 0
    return dict(
        scope="Four warmed captured producer/reconstruction/consumer replays: random200 (local), padded cube (global), invalid input, restored random200. Device phase advance and final synchronization only. Setup/warmup and readback excluded from native trace; not a throughput measurement.",
        columns=["tetrahedra", "error", "local_certificate_or_minus_one_on_error", "checksum"],
        records=values.tolist(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dot", type=Path)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    result = audit(args.profile, args.dot)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
