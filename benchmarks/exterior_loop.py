"""Audit real-input exterior recovery with captured device producers/consumers."""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

import warpower


@wp.kernel
def produce(
    phase: wp.array[int],
    source_roles: wp.array[int],
    roles: wp.array[int],
    count: wp.array[int],
    source_error: wp.array[int],
    n: int,
):
    i = wp.tid()
    roles[i] = wp.where(phase[0] == 1, 2, source_roles[i])
    if i == 0:
        count[0] = wp.where(phase[0] == 2, 0, n)
        source_error[0] = int(phase[0] == 3)


@wp.kernel
def consume(
    phase: wp.array[int],
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    count: wp.array[int],
    valid: wp.array[int],
    volumes: wp.array[wp.float64],
):
    i = wp.tid()
    if valid[0] == 0 or i >= count[0]:
        return
    a = wp.vec3d(points[tets[i, 0]])
    b, c, d = (
        wp.vec3d(points[tets[i, 1]]) - a,
        wp.vec3d(points[tets[i, 2]]) - a,
        wp.vec3d(points[tets[i, 3]]) - a,
    )
    wp.atomic_add(volumes, phase[0], wp.dot(b, wp.cross(c, d)) / wp.float64(6.0))


@wp.kernel
def record(
    phase: wp.array[int],
    error: wp.array[int],
    diagnostics: wp.array[int],
    valid: wp.array[int],
    count: wp.array[int],
    removed: wp.array[int],
    rounds: wp.array[int],
    rejected: wp.array[int],
    output: wp.array2d[int],
):
    k = phase[0]
    output[k, 0] = error[0]
    output[k, 1] = diagnostics[0]
    output[k, 2] = valid[0]
    output[k, 3] = wp.where(valid[0] != 0, count[0], 0)
    output[k, 4] = removed[0]
    output[k, 5] = rounds[0]
    output[k, 6] = rejected[0]
    phase[0] = (k + 1) % 5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--dot", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = (
        Path(__file__).parents[1]
        / "results/validation/vtm/graded-armadillo-failure"
        / "failed-candidate.npz"
    )
    data = np.load(source)
    d = warpower.SurfaceDomain(
        data["input_points"],
        data["input_faces"],
        feature_angle=120,
        surface_subdivisions=1,
        crease_subdivisions=1,
    )
    roles = wp.array(data["roles"], device=d.device)
    source_roles = wp.clone(roles)
    n = len(data["full_tetrahedra"])
    count = wp.full(1, n, dtype=int, device=d.device)
    source_error = wp.zeros(1, dtype=int, device=d.device)
    tets = np.vstack([data["full_tetrahedra"], [-1, -1, -1, -1]]).astype(np.int32)
    e = warpower.ExtractionWorkspace(
        d,
        data["points"],
        tets,
        count,
        source_error=source_error,
        containment_recovery=True,
        exterior_recovery=True,
        feature_roles=roles,
    )
    phase = wp.zeros(1, dtype=int, device=d.device)
    output = wp.empty((5, 7), dtype=int, device=d.device)
    volumes = wp.zeros(5, dtype=wp.float64, device=d.device)
    wp.load_module(module=__name__, device=d.device)
    with wp.ScopedCapture() as cap:
        wp.launch(produce, len(roles), inputs=[phase, source_roles, roles, count, source_error, n])
        e.run()
        wp.launch(
            consume,
            e.capacity,
            inputs=[phase, e.points, e.tetrahedra, e.tetrahedra_count, e.valid, volumes],
        )
        wp.launch(
            record,
            1,
            inputs=[
                phase,
                e.error,
                e.diagnostics,
                e.valid,
                e.tetrahedra_count,
                e.removed_count,
                e.exterior_rounds,
                e.exterior_rejected,
                output,
            ],
        )
    for _ in range(5):
        wp.capture_launch(cap.graph)
    wp.synchronize()
    volumes.zero_()
    if args.profile:
        import torch

        torch.cuda.cudart().cudaProfilerStart()
    for _ in range(5):
        wp.capture_launch(cap.graph)
    wp.synchronize()
    if args.profile:
        torch.cuda.cudart().cudaProfilerStop()
    expected = [
        [0, 0, 1, 18767, 2, 1, 0],
        [0, 2, 0, 0, 0, 1, 1],
        [0, 0, 1, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 18767, 2, 1, 0],
    ]
    np.testing.assert_array_equal(output.numpy(), expected)
    np.testing.assert_allclose(
        volumes.numpy(), [0.5286930860623611, 0, 0, 0, 0.5286930860623611], rtol=1e-11
    )
    if args.dot:
        args.dot.parent.mkdir(parents=True, exist_ok=True)
        wp.capture_debug_dot_print(cap.graph, str(args.dot), verbose=True)
        assert all(x not in args.dot.read_text() for x in ["MEM_ALLOC", "MEM_FREE", "HOST"])
    result = dict(
        phases=["repaired", "feature_protected_rejected", "empty", "upstream_error", "restored"],
        error_diagnostics_valid_count_removed_rounds_rejected=expected,
        consumed_volumes=volumes.numpy().tolist(),
        device=d.device.name,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
