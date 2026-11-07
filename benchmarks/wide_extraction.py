"""Audit uncertain pipeline geometry with resident valid/invalid/restored inputs."""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

from examples.vtm_inputs import surface
from tests.test_sphere_bounds import exact_sphere
from tests.test_wide_center import bounded_bridge, bounded_core, bridge, exact_area2
from warpower import ExtractionWorkspace, SurfaceDomain
from warpower._selective import classify_retained
from warpower._voronoi_volume import BoundedVoronoiVolumes


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
    selection_error: wp.array[int],
    selected: wp.array[int],
    quality: wp.array[wp.float64],
    quality_sums: wp.array[wp.float64],
    dual_error: wp.array[int],
    dual_count: wp.array[int],
    volume: wp.array[wp.float64],
    volume_sums: wp.array[wp.float64],
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
    records[row, 5] = selection_error[0]
    records[row, 6] = dual_error[0]
    records[row, 7] = dual_count[0]
    selected_count = int(0)
    selected_quality = wp.float64(0.0)
    if selection_error[0] == 0:
        for i in range(selected.shape[0]):
            if selected[i] != 0:
                selected_count += 1
                selected_quality += quality[i]
    records[row, 8] = selected_count
    quality_sums[row] = selected_quality
    total = wp.float64(0.0)
    for i in range(volume.shape[0]):
        total += volume[i]
    volume_sums[row] = total
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
    dual_original = bounded_bridge()
    dual_flat, dual_invalid = dual_original.copy(), dual_original.copy()
    dual_flat[3] = dual_flat[0]
    dual_invalid[0, 0] = np.nan
    dual_source = wp.array(
        np.stack([dual_original, dual_flat, dual_invalid, dual_original]),
        dtype=wp.vec3,
        device=domain.device,
    )
    core = bounded_core(dual_original)
    dual = BoundedVoronoiVolumes(core)
    selection_error = wp.zeros(1, dtype=int, device=domain.device)
    selected = wp.empty(2, dtype=int, device=domain.device)
    quality = wp.empty(2, dtype=wp.float64, device=domain.device)
    phase = wp.zeros(1, dtype=int, device=domain.device)
    records = wp.zeros((4, 9), dtype=int, device=domain.device)
    volume_sums = wp.zeros(4, dtype=wp.float64, device=domain.device)
    quality_sums = wp.zeros(4, dtype=wp.float64, device=domain.device)
    wp.load_module(module=__name__, device=domain.device)
    wp.load_module(module="warpower._selective", device=domain.device)
    with wp.ScopedCapture() as cap:
        wp.launch(produce, len(points), inputs=[source, phase, points])
        extraction.run()
        selection_error.zero_()
        wp.launch(
            classify_retained,
            2,
            inputs=[
                domain.tree,
                points,
                extraction.input_tetrahedra,
                extraction.input_count,
                0.4,
                selection_error,
                selected,
                quality,
            ],
        )
        wp.launch(produce, len(core.points), inputs=[dual_source, phase, core.points])
        dual.run()
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
                selection_error,
                selected,
                quality,
                quality_sums,
                dual.error,
                dual.count,
                dual.volume,
                volume_sums,
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
        values,
        [
            [0, 1, 1, 4, 22, 0, 0, 4, 1],
            [8, 0, 0, 0, 0, 8, 2, 0, 0],
            [4, 0, 0, 0, 0, 4, 2, 0, 0],
            [0, 1, 1, 4, 22, 0, 0, 4, 1],
        ],
    )
    from tests.test_voronoi_volume import oracle

    expected, _ = oracle(dual_original)
    np.testing.assert_allclose(
        volume_sums.numpy(), [expected.sum(), 0, 0, expected.sum()], rtol=1e-7
    )
    det, _, radius2 = exact_sphere(healthy)
    expected_quality = 3 * float(det) / (exact_area2(healthy) * np.sqrt(float(radius2)))
    np.testing.assert_allclose(
        quality_sums.numpy(), [expected_quality, 0, 0, expected_quality], rtol=1e-14
    )
    np.testing.assert_array_equal(points.numpy(), original)
    np.testing.assert_array_equal(core.points.numpy(), dual_original)
    assert phase.numpy()[0] == 0
    return dict(
        scope="Four warmed resident producer/extraction/quality-selection/bounded-dual-volume/consumer phases: exact-positive bridge whose ordinary determinant is zero; physically flat input; nonfinite input; original input restored. Fixed certified dual connectivity was built at setup. Device phase advance and one final synchronization. Setup, warmup and validation excluded from native trace; not a throughput measurement.",
        columns=[
            "error",
            "valid",
            "tetrahedra",
            "boundary_faces",
            "index_sum",
            "selection_error",
            "dual_error",
            "bounded_cells",
            "selected_cells",
        ],
        records=values.tolist(),
        volume_sums=volume_sums.numpy().tolist(),
        quality_sums=quality_sums.numpy().tolist(),
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
