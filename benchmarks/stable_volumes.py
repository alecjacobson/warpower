"""Audit uniform/graded accumulation of tiny cells in a resident loop."""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

from warpower import SizingField
from warpower._graded_odt import accumulate_graded_tets
from warpower._odt import accumulate_tets

wp.set_module_options({"enable_backward": False, "fast_math": False})


@wp.kernel
def produce(source: wp.array[wp.vec3], points: wp.array[wp.vec3], phase: wp.array[int]):
    i = wp.tid()
    j = i
    if phase[0] == 1 and i < 2:
        j = 1 - i
    p = source[j]
    if phase[0] == 2 and i == 0:
        p = wp.vec3(float(wp.nan), 0.0, 0.0)
    if phase[0] == 3:
        p *= 2.0
    points[i] = p


@wp.kernel
def consume(
    invalid: wp.array[int],
    volume: wp.array[wp.float64],
    energy: wp.array[wp.float64],
    phase: wp.array[int],
    offset: int,
    records: wp.array2d[wp.float64],
):
    row = phase[0]
    records[row, offset] = wp.float64(invalid[0])
    records[row, offset + 1] = volume[0]
    records[row, offset + 2] = energy[0]
    if offset == 3:
        phase[0] = (row + 1) % 4


def audit(profile=False, dot=None):
    path = Path(__file__).resolve().parents[1] / "tests/data/tiny-positive-tets.npz"
    cells = np.load(path)["points"]
    reference = json.loads(path.with_suffix(".json").read_text())
    expected = sum(p["exact_determinant_estimate"] / 6 for p in reference["cases"])
    source = wp.array(cells.reshape(-1, 3), dtype=wp.vec3, device="cuda:0")
    points = wp.empty_like(source)
    tets = wp.array(
        np.arange(len(source), dtype=np.int32).reshape(-1, 4), dtype=int, device=source.device
    )
    count = wp.full(1, len(cells), dtype=int, device=source.device)
    mesh_error = wp.zeros(1, dtype=int, device=source.device)
    phase = wp.zeros_like(mesh_error)
    records = wp.empty((4, 6), dtype=wp.float64, device=source.device)
    field = SizingField(
        wp.zeros(1, dtype=wp.vec3, device=source.device),
        wp.full(1, 0.5, dtype=float, device=source.device),
        gradation=0,
    )
    scratch = []
    for _ in range(2):
        invalid = wp.empty_like(mesh_error)
        mass = wp.empty(len(source), dtype=wp.float64, device=source.device)
        moment = wp.empty(len(source), dtype=wp.vec3d, device=source.device)
        edge = wp.empty_like(mass)
        energy, volume, quality = [
            wp.empty(1, dtype=wp.float64, device=source.device) for _ in range(3)
        ]
        scratch.append((invalid, mass, moment, edge, energy, volume, quality))
    for module in [__name__, "warpower._odt", "warpower._graded_odt"]:
        wp.load_module(module=module, device=source.device)
    with wp.ScopedCapture() as capture:
        wp.launch(produce, len(source), inputs=[source, points, phase])
        for k, values in enumerate(scratch):
            invalid, mass, moment, edge, energy, volume, quality = values
            for array in [invalid, mass, moment, energy, volume]:
                array.zero_()
            edge.fill_(1e300)
            quality.fill_(1.0)
            args = [points, tets, count, mesh_error, *values, True]
            kernel = accumulate_graded_tets if k else accumulate_tets
            wp.launch(kernel, len(cells), inputs=args + ([field.data] if k else []))
            wp.launch(consume, 1, inputs=[invalid, volume, energy, phase, 3 * k, records])
    if dot:
        wp.capture_debug_dot_print(capture.graph, str(dot), verbose=True)
    if profile:
        for _ in range(4):
            wp.capture_launch(capture.graph)
        wp.synchronize()
        import torch

        torch.cuda.cudart().cudaProfilerStart()
    for _ in range(4):
        wp.capture_launch(capture.graph)
    wp.synchronize()
    if profile:
        torch.cuda.cudart().cudaProfilerStop()
    result = records.numpy()
    for offset in [0, 3]:
        np.testing.assert_array_equal(result[:, offset], [0, 256, 256, 0])
        np.testing.assert_allclose(result[[0, 3], offset + 1], [expected, expected * 8], rtol=3e-15)
    np.testing.assert_allclose(result[:, 1], result[:, 4], rtol=3e-15)
    np.testing.assert_allclose(result[:, 5], 8 * result[:, 2], rtol=3e-15)
    assert phase.numpy()[0] == 0
    return dict(
        scope="Four resident producer/uniform and graded accumulation/consumer phases: original tiny positive cells, true inversion, nonfinite input, restored cells scaled by two. Device phase advance, one final synchronization. Native profiling first warms a complete cycle outside the trace. Not full meshing or a performance comparison.",
        warmup_replays=4 if profile else 0,
        columns=[
            "uniform_invalid",
            "uniform_volume",
            "uniform_energy",
            "graded_invalid",
            "graded_volume",
            "graded_energy",
        ],
        records=result.tolist(),
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
