"""Profile a device-only changing-input certified-meshing producer/consumer loop."""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

from tests.test_fast_candidates import cloud, producer
from warpower import DelaunayWorkspace

wp.set_module_options({"enable_backward": False})


@wp.kernel
def consume(
    tets: wp.array2d[int], count: wp.array[int], error: wp.array[int], total: wp.array[wp.int64]
):
    i = wp.tid()
    if error[0] == 0 and i < count[0]:
        value = wp.int64(0)
        for k in range(4):
            value += wp.int64(tets[i, k])
        wp.atomic_add(total, 0, value)


@wp.kernel
def advance(
    phase: wp.array[int],
    count: wp.array[int],
    error: wp.array[int],
    attempted: wp.array[int],
    accepted: wp.array[int],
    fast_error: wp.array[int],
    repaired: wp.array[int],
    total: wp.array[wp.int64],
    log: wp.array2d[wp.int64],
):
    row = phase[0]
    log[row, 0] = wp.int64(count[0])
    log[row, 1] = wp.int64(error[0])
    log[row, 2] = wp.int64(attempted[0])
    log[row, 3] = wp.int64(accepted[0])
    log[row, 4] = wp.int64(fast_error[0])
    log[row, 5] = wp.int64(repaired[0])
    log[row, 6] = total[0]
    phase[0] = (row + 1) % 4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=32768)
    parser.add_argument("--no-fast", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--dot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.n < 32768:
        parser.error("n must be at least 32768 to exercise fast construction")
    points = cloud(args.n)
    mesh = DelaunayWorkspace(points, fast_candidates=not args.no_fast)
    source = wp.array(points, dtype=wp.vec3, device=mesh.device)
    invalid = wp.array([np.nan], dtype=float, device=mesh.device)
    phase = wp.zeros(1, dtype=int, device=mesh.device)
    total = wp.zeros(1, dtype=wp.int64, device=mesh.device)
    log = wp.zeros((4, 7), dtype=wp.int64, device=mesh.device)
    wp.load_module(module=__name__, device=mesh.device)
    wp.load_module(module="tests.test_fast_candidates", device=mesh.device)
    with wp.ScopedCapture() as cap:
        wp.launch(producer, len(points), inputs=[source, invalid, mesh.points, phase])
        mesh.run()
        total.zero_()
        wp.launch(
            consume,
            len(mesh.tetrahedra),
            inputs=[mesh.tetrahedra, mesh.tetrahedra_count, mesh.error, total],
        )
        wp.launch(
            advance,
            1,
            inputs=[
                phase,
                mesh.tetrahedra_count,
                mesh.error,
                mesh.fast_attempted,
                mesh.fast_accepted,
                mesh.fast_error,
                mesh.fast_repaired_cells,
                total,
                log,
            ],
        )
    for _ in range(4):
        wp.capture_launch(cap.graph)
    wp.synchronize()
    if args.dot:
        args.dot.parent.mkdir(parents=True, exist_ok=True)
        wp.capture_debug_dot_print(cap.graph, str(args.dot), verbose=True)
    if args.profile:
        import torch

        torch.cuda.cudart().cudaProfilerStart()
    for _ in range(4):
        wp.capture_launch(cap.graph)
    wp.synchronize()
    if args.profile:
        torch.cuda.cudart().cudaProfilerStop()
    values = log.numpy()
    np.testing.assert_array_equal(values[:, 1], [0, 0, 0, 1])
    assert np.all(values[:3, 0] > 0) and values[3, 0] == 0
    assert np.all(values[:3, 6] > 0) and values[3, 6] == 0
    np.testing.assert_array_equal(values[:, 2], [1, 1, 0, 0] if not args.no_fast else [0] * 4)
    result = dict(
        scope="Four warmed captured replays; device producer changes one point, collapses to128 distinct sites, introduces invalid input, and advances phase. Meshing and consumer counts/checksums stay on device; one final synchronization outside geometry. Not a throughput benchmark.",
        n=args.n,
        fast_candidates=mesh.fast_candidates,
        device=mesh.device.name,
        warp=wp.__version__,
        columns=[
            "tetrahedra",
            "error",
            "fast_attempted",
            "fast_accepted",
            "fast_error",
            "fast_repaired_cells",
            "consumer_checksum",
        ],
        phases=values.tolist(),
        next_phase=int(phase.numpy()[0]),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
