"""Audit integrated facet recovery with changing resident input and a consumer."""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

from tests.test_domain import torus
from warpower import DelaunayWorkspace

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})
NAN = wp.constant(float("nan"))


@wp.kernel
def produce(
    structured: wp.array[wp.vec3],
    random: wp.array[wp.vec3],
    points: wp.array[wp.vec3],
    phase: wp.array[int],
):
    i = wp.tid()
    points[i] = structured[i]
    if phase[0] == 1:
        points[i] = random[i]
    elif phase[0] == 2 and i == 0:
        points[i] = wp.vec3(NAN, 0.0, 0.0)


@wp.kernel
def consume(
    tets: wp.array2d[int],
    count: wp.array[int],
    error: wp.array[int],
    recovered: wp.array[int],
    seeds: wp.array[int],
    rounds: wp.array[int],
    phase: wp.array[int],
    records: wp.array2d[wp.int64],
):
    row = phase[0]
    records[row, 0] = wp.int64(count[0])
    records[row, 1] = wp.int64(error[0])
    records[row, 2] = wp.int64(recovered[0])
    records[row, 3] = wp.int64(seeds[0])
    records[row, 4] = wp.int64(rounds[0])
    total = wp.int64(0)
    for i in range(count[0]):
        for k in range(4):
            total += wp.int64(tets[i, k])
    records[row, 5] = total
    phase[0] = (row + 1) % 4


def audit(profile=False, dot=None):
    structured, _ = torus(16, 8)
    random = np.random.default_rng(2026).normal(size=structured.shape).astype(np.float32)
    expected = []
    for points in [structured, random]:
        baseline = DelaunayWorkspace(points, facet_recovery=False).run()
        assert baseline.error.numpy()[0] == 0
        cells = baseline.tetrahedra.numpy()[: baseline.tetrahedra_count.numpy()[0]]
        expected.append((len(cells), int(cells.sum())))
    w = DelaunayWorkspace(structured, facet_recovery=True, symbolic_recovery=False, max_attempts=1)
    source = wp.array(structured, dtype=wp.vec3, device=w.device)
    other = wp.array(random, dtype=wp.vec3, device=w.device)
    phase = wp.zeros(1, dtype=int, device=w.device)
    records = wp.zeros((4, 6), dtype=wp.int64, device=w.device)
    wp.load_module(module=__name__, device=w.device)
    with wp.ScopedCapture() as capture:
        wp.launch(produce, len(source), inputs=[source, other, w.points, phase])
        w.run()
        wp.launch(
            consume,
            1,
            inputs=[
                w.tetrahedra,
                w.tetrahedra_count,
                w.error,
                w.facet_recovered,
                w.facet_seeds,
                w.facet_rounds,
                phase,
                records,
            ],
        )
    if dot:
        wp.capture_debug_dot_print(capture.graph, str(dot), verbose=True)
    # Native timing traces need warmed replay. Memory/lifetime checks should
    # exercise the first four input phases without duplicating that full cycle.
    if profile:
        for _ in range(4):
            wp.capture_launch(capture.graph)
        wp.synchronize()
    if profile:
        import torch

        torch.cuda.cudart().cudaProfilerStart()
    for _ in range(4):
        wp.capture_launch(capture.graph)
    wp.synchronize()
    if profile:
        torch.cuda.cudart().cudaProfilerStop()
    values = records.numpy()
    np.testing.assert_array_equal(values[:, 0], [expected[0][0], expected[1][0], 0, expected[0][0]])
    np.testing.assert_array_equal(values[:, 1], [0, 0, 1, 0])
    np.testing.assert_array_equal(values[[0, 3], 2], 1)
    np.testing.assert_array_equal(values[:, 5], [expected[0][1], expected[1][1], 0, expected[0][1]])
    np.testing.assert_array_equal(values[2, 2:], 0)
    assert phase.numpy()[0] == 0
    assert w.symbolic_attempted.numpy()[0] == 0
    return dict(
        scope="Four integrated producer/Delaunay/consumer replays with explicit facet recovery, one auxiliary attempt and symbolic cells disabled: structured torus, random sites, nonfinite input, restored torus. Device phase advance; final synchronization/readback only. Native profiling additionally warms four replays outside the trace. Not a throughput or full VTM comparison.",
        warmup_replays=4 if profile else 0,
        columns=[
            "tetrahedra",
            "error",
            "facet_recovered",
            "seed_cells",
            "completion_rounds",
            "checksum",
        ],
        records=values.tolist(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--dot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.dot:
        args.dot.parent.mkdir(parents=True, exist_ok=True)
    result = audit(args.profile, args.dot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
