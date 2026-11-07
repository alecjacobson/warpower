"""Resident producer/consumer audit of material repair and its quality selection."""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

from tests.test_containment import cavity, fan, producer
from warpower import ExtractionWorkspace, SurfaceDomain
from warpower._selective import SelectiveUpdate


@wp.kernel
def consume(
    phase: wp.array[int],
    error: wp.array[int],
    valid: wp.array[int],
    count: wp.array[int],
    boundary: wp.array[int],
    additions: wp.array[int],
    rounds: wp.array[int],
    selection_error: wp.array[int],
    selection_count: wp.array[int],
    keep: wp.array[int],
    quality: wp.array[wp.float64],
    records: wp.array2d[int],
    quality_sums: wp.array[wp.float64],
):
    k = phase[0]
    records[k, 0] = error[0]
    records[k, 1] = valid[0]
    records[k, 2] = count[0]
    records[k, 3] = boundary[0]
    records[k, 4] = additions[0]
    records[k, 5] = rounds[0]
    records[k, 6] = selection_error[0]
    records[k, 7] = selection_count[0]
    value = wp.float64(0.0)
    if selection_error[0] == 0:
        for i in range(keep.shape[0]):
            if keep[i] != 0:
                value += quality[i]
    quality_sums[k] = value
    phase[0] += 1


class Core:
    pass


def audit(profile=False, dot=None):
    p, t = fan()
    d = SurfaceDomain(*cavity())
    base = wp.array(p, dtype=wp.vec3, device=d.device)
    points = wp.clone(base)
    core = Core()
    core.device, core.domain, core.points, core.n = d.device, d, points, len(points)
    core.full_tetrahedra = wp.array(t, device=d.device)
    core.full_tetrahedra_count = wp.full(1, len(t), dtype=int, device=d.device)
    core.mesh = Core()
    core.mesh.error = wp.zeros(1, dtype=int, device=d.device)
    core.material_recovery = wp.ones(1, dtype=int, device=d.device)
    e = ExtractionWorkspace(
        d, points, core.full_tetrahedra, core.full_tetrahedra_count, containment_recovery=True
    )
    s = SelectiveUpdate(core, 0.4)
    phase = wp.zeros(1, dtype=int, device=d.device)
    records = wp.zeros((4, 8), dtype=int, device=d.device)
    quality_sums = wp.zeros(4, dtype=wp.float64, device=d.device)
    wp.load_module(module=__name__, device=d.device)
    wp.load_module(module="tests.test_containment", device=d.device)
    with wp.ScopedCapture() as cap:
        wp.launch(producer, len(points), inputs=[base, points, phase])
        e.run()
        s.evaluate()
        wp.launch(
            consume,
            1,
            inputs=[
                phase,
                e.error,
                e.valid,
                e.tetrahedra_count,
                e.boundary_count,
                e.repaired_count,
                e.repair_rounds,
                s.error,
                s.retained_count,
                s.keep,
                s.quality,
                records,
                quality_sums,
            ],
        )
    if dot:
        dot.parent.mkdir(parents=True, exist_ok=True)
        wp.capture_debug_dot_print(cap.graph, str(dot), verbose=True)
    for _ in range(4):
        wp.capture_launch(cap.graph)
    phase.zero_()
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
    expected = np.array(
        [
            [0, 1, 5, 10, 2, 1, 0, 5],
            [4, 0, 0, 0, 0, 0, 20, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 5, 10, 2, 1, 0, 5],
        ]
    )
    np.testing.assert_array_equal(values, expected)
    # Independent ordinary geometry is reliable for this isolated fan.
    cells = p[t].astype(np.float64)
    edges = cells[:, 1:] - cells[:, :1]
    offsets = np.linalg.solve(2 * edges, np.sum(edges * edges, axis=2)[..., None])[..., 0]
    determinant = np.linalg.det(edges)
    area2 = np.zeros(len(t))
    for opposite in range(4):
        f = np.delete(cells, opposite, axis=1)
        area2 += np.linalg.norm(np.cross(f[:, 1] - f[:, 0], f[:, 2] - f[:, 0]), axis=1)
    total = np.sum(3 * determinant / (area2 * np.linalg.norm(offsets, axis=1)))
    np.testing.assert_allclose(quality_sums.numpy(), [total, 0, 0, total], rtol=1e-12)
    result = dict(
        scope="Four warmed resident repaired/nonfinite/empty/restored frames through extraction, repaired quality selection and a device consumer. Setup/warmup/readback/independent checking outside the trace.",
        columns=[
            "error",
            "valid",
            "tetrahedra",
            "boundary_faces",
            "added_cells",
            "repair_passes",
            "selection_error",
            "selected_cells",
        ],
        records=values.tolist(),
        quality_sums=quality_sums.numpy().tolist(),
        reference_quality_sum=float(total),
        graph_launches=4,
        final_synchronizations=1,
    )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--dot", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = audit(a.profile, a.dot)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
