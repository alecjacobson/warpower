"""Audit uncertain extraction geometry with resident valid/invalid/restored inputs."""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

from examples.vtm_inputs import surface
from tests.test_wide_center import bridge
from warpower import ExtractionWorkspace, SurfaceDomain


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
    valid: wp.array[int],
    boundary_count: wp.array[int],
    records: wp.array2d[int],
):
    row = phase[0]
    records[row, 0] = error[0]
    records[row, 1] = valid[0]
    records[row, 2] = count[0]
    records[row, 3] = boundary_count[0]
    checksum = int(0)
    for i in range(count[0]):
        for k in range(4):
            checksum += tets[i, k]
    records[row, 4] = checksum
    phase[0] = (row + 1) % records.shape[0]


def audit(profile=False, dot=None):
    healthy = np.array([[-2, -0.1, 0], [-1.8, -0.1, 0], [-2, 0.1, 0], [-2, -0.1, 0.2]], np.float32)
    original = np.vstack([bridge(), healthy])
    flat = original.copy()
    flat[3] = flat[0]
    invalid = original.copy()
    invalid[0, 0] = np.nan
    frames = np.stack([original, flat, invalid, original])
    domain = SurfaceDomain(*surface("disconnected"), surface_subdivisions=1)
    points = wp.array(original, dtype=wp.vec3, device=domain.device)
    extraction = ExtractionWorkspace(domain, points, np.arange(8, dtype=np.int32).reshape(2, 4))
    source = wp.array(frames, dtype=wp.vec3, device=domain.device)
    phase = wp.zeros(1, dtype=int, device=domain.device)
    records = wp.zeros((4, 5), dtype=int, device=domain.device)
    wp.load_module(module=__name__, device=domain.device)
    with wp.ScopedCapture() as cap:
        wp.launch(produce, len(points), inputs=[source, phase, points])
        extraction.run()
        wp.launch(
            consume,
            1,
            inputs=[
                phase,
                extraction.tetrahedra,
                extraction.tetrahedra_count,
                extraction.error,
                extraction.valid,
                extraction.boundary_count,
                records,
            ],
        )
    if dot:
        wp.capture_debug_dot_print(cap.graph, str(dot), verbose=True)
    for _ in frames:
        wp.capture_launch(cap.graph)
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
    np.testing.assert_array_equal(
        values, [[0, 1, 1, 4, 22], [8, 0, 0, 0, 0], [4, 0, 0, 0, 0], [0, 1, 1, 4, 22]]
    )
    np.testing.assert_array_equal(points.numpy(), original)
    assert phase.numpy()[0] == 0
    return dict(
        scope="Four warmed resident producer/extraction/consumer phases: an exact-positive bridge whose ordinary determinant is zero plus healthy material; a physically flat input; a nonfinite input; original input restored. Device phase advance and one final synchronization. Setup, warmup and validation excluded from native trace; not a throughput measurement.",
        columns=["error", "valid", "tetrahedra", "boundary_faces", "index_sum"],
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
