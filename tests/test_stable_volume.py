from itertools import permutations
from pathlib import Path

import numpy as np
import pytest
import warp as wp

from warpower import SizingField
from warpower._graded_odt import accumulate_graded_tets
from warpower._odt import accumulate_tets
from warpower._stable_volume import determinant

wp.set_module_options({"enable_backward": False, "fast_math": False})


def fixture():
    return np.load(Path(__file__).parent / "data/tiny-positive-tets.npz")["points"]


def exact_det(points):
    def integer(x):
        numerator, denominator = float(x).as_integer_ratio()
        return numerator << (149 - (denominator.bit_length() - 1))

    p = [[integer(x) for x in row] for row in points]
    a, b, c = [[p[k][j] - p[0][j] for j in range(3)] for k in [1, 2, 3]]
    value = sum(
        a[j] * (b[(j + 1) % 3] * c[(j + 2) % 3] - b[(j + 2) % 3] * c[(j + 1) % 3]) for j in range(3)
    )
    return float(value) * 2.0**-447


@wp.kernel
def original_values(points: wp.array2d[wp.vec3], raw: wp.array[wp.float64]):
    i = wp.tid()
    p = wp.vec3d(points[i, 0])
    a, b, c = wp.vec3d(points[i, 1]) - p, wp.vec3d(points[i, 2]) - p, wp.vec3d(points[i, 3]) - p
    raw[i] = wp.dot(a, wp.cross(b, c))


@wp.kernel
def evaluate(points: wp.array2d[wp.vec3], raw: wp.array[wp.float64], stable: wp.array[wp.float64]):
    i = wp.tid()
    p0, p1, p2, p3 = points[i, 0], points[i, 1], points[i, 2], points[i, 3]
    a, b, c = wp.vec3d(p1) - wp.vec3d(p0), wp.vec3d(p2) - wp.vec3d(p0), wp.vec3d(p3) - wp.vec3d(p0)
    value = wp.dot(a, wp.cross(b, c))
    raw[i] = value
    stable[i] = determinant(p0, p1, p2, p3, a, b, c, value)


def evaluate_cases(points):
    points = wp.array(np.asarray(points, np.float32), dtype=wp.vec3, device="cuda:0")
    raw, current, stable = [
        wp.empty(points.shape[0], dtype=wp.float64, device=points.device) for _ in range(3)
    ]
    wp.launch(original_values, points.shape[0], inputs=[points, raw])
    wp.launch(evaluate, points.shape[0], inputs=[points, current, stable])
    return raw.numpy(), stable.numpy()


def test_tiny_cells_all_permutations_and_scales_match_independent_integer_values():
    cases = []
    for points in fixture():
        for scale in [2.0**-120, 1.0, 2.0**100]:
            scaled = (points.astype(np.float64) * scale).astype(np.float32)
            for order in permutations(range(4)):
                cases.append(scaled[list(order)])
    smallest = np.nextafter(np.float32(0), np.float32(1))
    largest = np.finfo(np.float32).max
    for scale in [smallest, largest]:
        cases.append(np.array([[0, 0, 0], [scale, 0, 0], [0, scale, 0], [0, 0, scale]], np.float32))
    expected = np.array([exact_det(p) for p in cases])
    raw, stable = evaluate_cases(cases)
    np.testing.assert_array_equal(np.sign(stable), np.sign(expected))
    np.testing.assert_allclose(stable, expected, rtol=3e-15, atol=0)
    assert np.any(raw != stable)


def test_well_conditioned_cells_preserve_existing_gpu_values():
    points = np.random.default_rng(110).normal(size=(512, 4, 3)).astype(np.float32)
    raw, stable = evaluate_cases(points)
    np.testing.assert_array_equal(stable, raw)
    np.testing.assert_allclose(stable, [exact_det(p) for p in points], rtol=2e-12, atol=0)


@pytest.mark.parametrize("graded", [False, True])
def test_accumulation_accepts_tiny_positive_cells_and_rejects_inversion(graded):
    cells = fixture()
    points = wp.array(cells.reshape(-1, 3), dtype=wp.vec3, device="cuda:0")
    indices = np.arange(len(points), dtype=np.int32).reshape(-1, 4)
    tets = wp.array(indices, dtype=int, device=points.device)
    count = wp.full(1, len(tets), dtype=int, device=points.device)
    mesh_error, invalid = [wp.zeros(1, dtype=int, device=points.device) for _ in range(2)]
    mass = wp.zeros(len(points), dtype=wp.float64, device=points.device)
    moment = wp.zeros(len(points), dtype=wp.vec3d, device=points.device)
    edge = wp.full(len(points), 1e300, dtype=wp.float64, device=points.device)
    energy, volume = [wp.zeros(1, dtype=wp.float64, device=points.device) for _ in range(2)]
    quality = wp.ones(1, dtype=wp.float64, device=points.device)
    field = (
        SizingField(
            np.zeros((1, 3), np.float32),
            wp.full(1, 0.5, dtype=float, device=points.device),
            gradation=0,
        )
        if graded
        else None
    )
    args = [
        points,
        tets,
        count,
        mesh_error,
        invalid,
        mass,
        moment,
        edge,
        energy,
        volume,
        quality,
        True,
    ]
    kernel = accumulate_graded_tets if graded else accumulate_tets
    wp.launch(kernel, len(tets), inputs=args + ([field.data] if graded else []))
    assert invalid.numpy()[0] == 0
    volumes = np.array([exact_det(p) / 6 for p in cells])
    density = 8 if graded else 1
    np.testing.assert_allclose(volume.numpy()[0], volumes.sum(), rtol=3e-15)
    np.testing.assert_allclose(
        mass.numpy().reshape(-1, 4), np.repeat((density * volumes)[:, None], 4, axis=1), rtol=3e-15
    )
    assert np.isfinite(energy.numpy()[0]) and energy.numpy()[0] > 0
    indices[0, [0, 1]] = indices[0, [1, 0]]
    tets.assign(indices)
    wp.launch(kernel, len(tets), inputs=args + ([field.data] if graded else []))
    assert invalid.numpy()[0] == 256


@wp.kernel
def produce(source: wp.array2d[wp.vec3], phase: wp.array[int], points: wp.array2d[wp.vec3]):
    i = wp.tid()
    points[0, i] = source[phase[0], i]


@wp.kernel
def consume(values: wp.array[wp.float64], phase: wp.array[int], history: wp.array[wp.float64]):
    history[phase[0]] = values[0]
    phase[0] = (phase[0] + 1) % 4


def test_resident_producer_volume_consumer_handles_inversion_nan_and_recovery():
    cell = fixture()[0]
    variants = np.stack([cell, cell[[1, 0, 2, 3]], cell, cell * 2])
    variants[2, 0, 0] = np.nan
    source = wp.array(variants, dtype=wp.vec3, device="cuda:0")
    points = wp.empty((1, 4), dtype=wp.vec3, device=source.device)
    phase = wp.zeros(1, dtype=int, device=source.device)
    raw, stable = [wp.empty(1, dtype=wp.float64, device=source.device) for _ in range(2)]
    history = wp.empty(4, dtype=wp.float64, device=source.device)
    wp.load_module(module=__name__, device=source.device)
    with wp.ScopedCapture() as capture:
        wp.launch(produce, 4, inputs=[source, phase, points])
        wp.launch(evaluate, 1, inputs=[points, raw, stable])
        wp.launch(consume, 1, inputs=[stable, phase, history])
    for _ in range(4):
        wp.capture_launch(capture.graph)
    expected = exact_det(cell)
    np.testing.assert_allclose(
        history.numpy(), [expected, -expected, np.nan, 8 * expected], rtol=3e-15
    )
