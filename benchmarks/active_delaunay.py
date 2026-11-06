"""Audit device-selected reconstruction capacities in a resident producer/consumer loop."""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

from tests.test_active_delaunay import advance, consume, produce
from tests.test_tetrahedra import check_mesh, reference, tet_set
from warpower._active_delaunay import ActiveDelaunayWorkspace
from warpower._active_sites import prepare_mesh


def audit(profile=False, dot=None):
    counts = [16, 49, 32, 97, 16, 96, 16]
    frames = np.random.default_rng(1213).normal(size=(len(counts), 96, 3)).astype(np.float32)
    frames[-1] = frames[0]
    frames[2, 16:32] = frames[2, :16]
    frames[4, 4, 0] = np.nan
    for frame, count in zip(frames, counts):
        frame[count:] = np.nan
    device = "cuda:0"
    source = wp.array(frames, dtype=wp.vec3, device=device)
    sizes = wp.array(counts, dtype=int, device=device)
    phase = wp.zeros(1, dtype=int, device=device)
    points = wp.zeros(96, dtype=wp.vec3, device=device)
    prepared = wp.zeros_like(points)
    active = wp.zeros(1, dtype=int, device=device)
    mesh = ActiveDelaunayWorkspace(prepared, active_count=active, prefix_capacities=(16, 48))
    history = wp.empty((len(counts), len(mesh.tetrahedra), 4), dtype=int, device=device)
    records = wp.empty((len(counts), 3), dtype=int, device=device)
    wp.load_module(module="tests.test_active_delaunay", device=device)
    wp.load_module(module="warpower._active_sites", device=device)
    with wp.ScopedCapture() as cap:
        wp.launch(produce, mesh.n, inputs=[source, sizes, phase, points, active])
        wp.launch(prepare_mesh, mesh.n, inputs=[points, active, prepared])
        mesh.run()
        wp.launch(
            consume,
            len(mesh.tetrahedra),
            inputs=[
                phase,
                mesh.tetrahedra,
                mesh.tetrahedra_count,
                mesh.error,
                mesh.selected_capacity,
                history,
                records,
            ],
        )
        wp.launch(advance, 1, inputs=[phase])
    if dot:
        wp.capture_debug_dot_print(cap.graph, str(dot), verbose=True)
    for _ in counts:
        wp.capture_launch(cap.graph)
    phase.zero_()
    mesh.clear_reuse()
    wp.synchronize()
    if profile:
        import torch

        torch.cuda.cudart().cudaProfilerStart()
    for _ in counts:
        wp.capture_launch(cap.graph)
    wp.synchronize()
    if profile:
        torch.cuda.cudart().cudaProfilerStop()
    rows, cells = records.numpy(), history.numpy()
    for i, count in enumerate(counts):
        if i in (3, 4):
            assert rows[i, 0] == 0 and rows[i, 1] != 0
        else:
            assert rows[i, 0] > 0 and rows[i, 1] == 0
            cloud = frames[i, :count]
            owners = np.unique(cloud, axis=0, return_index=True)[1]
            expected = {tuple(sorted(owners[list(t)])) for t in reference(cloud[owners])}
            actual = cells[i, : rows[i, 0]]
            assert tet_set(actual) == expected
            check_mesh(cloud, actual)
    np.testing.assert_array_equal(rows[:, 2], [16, 96, 48, 0, 96, 96, 16])
    assert phase.numpy()[0] == len(counts)
    return dict(
        scope="Seven warmed resident producer/reconstruction/consumer phases: growth across both thresholds, shrinkage with duplicates, invalid count, nonfinite point with full fallback, full capacity and restored prefix. Only device decisions between phases and one final synchronization; setup, warmup and readback excluded from native trace. Not a throughput measurement.",
        columns=["tetrahedra", "error", "selected_capacity"],
        records=rows.tolist(),
        route_calls=mesh.route_calls.numpy().tolist(),
        independent_delaunay_and_orientation_match=True,
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
