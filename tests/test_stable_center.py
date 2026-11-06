from fractions import Fraction
from itertools import permutations
from pathlib import Path

import numpy as np
import warp as wp

from tests.test_sphere_bounds import exact_sphere
from warpower._stable_center import circumcenter, exact_center

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


def rounded_fraction(value):
    """Independent exact nearest-even reference, with signed zero canonicalized."""
    if value == 0:
        return np.float32(0)
    sign = -1 if value < 0 else 1
    value = abs(value)
    largest = np.finfo(np.float32).max
    overflow = Fraction(float(largest)) + Fraction(2**103)
    if value >= overflow:
        return np.float32(sign * np.inf)
    with np.errstate(over="ignore"):
        middle = np.float32(float(value))
    if not np.isfinite(middle):
        middle = largest
    candidates = [middle]
    if middle > 0:
        candidates.append(np.nextafter(middle, np.float32(0)))
    upper = np.nextafter(middle, np.float32(np.inf))
    if np.isfinite(upper):
        candidates.append(upper)
    chosen = min(
        candidates,
        key=lambda x: (abs(Fraction(float(x)) - value), int(x.view(np.uint32)) & 1),
    )
    return np.float32(sign * chosen) if chosen != 0 else np.float32(0)


@wp.kernel
def evaluate(
    points: wp.array2d[wp.vec3],
    exact: bool,
    centers: wp.array[wp.vec3],
    wide: wp.array[wp.vec3d],
    radii: wp.array[wp.float64],
    valid: wp.array[int],
    fallback: wp.array[int],
):
    i = wp.tid()
    result = circumcenter(points[i, 0], points[i, 1], points[i, 2], points[i, 3])
    if exact:
        result = exact_center(points[i, 0], points[i, 1], points[i, 2], points[i, 3])
    centers[i] = result.point
    wide[i] = result.wide
    radii[i] = result.radius
    valid[i] = result.valid
    fallback[i] = result.exact


@wp.kernel
def original(points: wp.array2d[wp.vec3], centers: wp.array[wp.vec3], radii: wp.array[wp.float64]):
    i = wp.tid()
    p = wp.vec3d(points[i, 0])
    a, b, c = wp.vec3d(points[i, 1]) - p, wp.vec3d(points[i, 2]) - p, wp.vec3d(points[i, 3]) - p
    det = wp.dot(a, wp.cross(b, c))
    offset = (
        wp.dot(a, a) * wp.cross(b, c)
        + wp.dot(b, b) * wp.cross(c, a)
        + wp.dot(c, c) * wp.cross(a, b)
    ) / (wp.float64(2.0) * det)
    centers[i] = wp.vec3(p + offset)
    radii[i] = wp.length(offset)


def evaluate_cases(cases, exact=False):
    points = wp.array(np.asarray(cases, np.float32), dtype=wp.vec3, device="cuda:0")
    n = len(cases)
    outputs = [
        wp.empty(n, dtype=dtype, device=points.device)
        for dtype in [wp.vec3, wp.vec3d, wp.float64, int, int]
    ]
    wp.launch(evaluate, n, inputs=[points, exact, *outputs])
    return [x.numpy() for x in outputs]


def assert_reference(cases, exact=False):
    centers, wide, radii, valid, fallback = evaluate_cases(cases, exact)
    for i, p in enumerate(cases):
        determinant, center, radius2 = exact_sphere(p)
        if determinant == 0:
            assert valid[i] == 0
            continue
        expected = np.array([rounded_fraction(c) for c in center], np.float32)
        if not np.all(np.isfinite(expected)):
            assert valid[i] == 0
            continue
        assert valid[i] == 1, (i, p, expected)
        np.testing.assert_array_equal(centers[i], expected, err_msg=f"case {i}")
        assert np.isfinite(radii[i]) and radii[i] > 0
        if fallback[i]:
            np.testing.assert_allclose(wide[i], [float(c) for c in center], rtol=3e-15, atol=0)
            np.testing.assert_allclose(radii[i] ** 2, float(radius2), rtol=5e-14, atol=0)
    return centers, fallback


def test_hand_cospherical_centers_all_permutations_and_scales():
    points = np.load(Path(__file__).parent / "data/hand-fit256.npz")["points"]
    parents = [
        [14, 50, 96, 233],
        [50, 96, 199, 233],
        [15, 50, 199, 233],
        [91, 96, 199, 233],
        [15, 81, 199, 233],
        [81, 91, 199, 233],
    ]
    cases = [
        (points[tet].astype(np.float64) * scale).astype(np.float32)[list(order)]
        for scale in [2.0**-100, 1.0, 2.0**100]
        for tet in parents
        for order in permutations(range(4))
    ]
    centers, fallback = assert_reference(cases)
    assert np.count_nonzero(fallback) > 0
    # Every exact cosphere group produces one actual insertion position.
    for group in centers.reshape(3, 2, 72, 3).reshape(6, 72, 3):
        np.testing.assert_array_equal(group, np.repeat(group[:1], 72, axis=0))
        np.testing.assert_array_equal(group[:, 2], 0)


def test_exact_rounding_ties_subnormals_extremes_and_overflow():
    cases = []
    smallest = np.nextafter(np.float32(0), np.float32(1))
    for start in [
        np.float32(0),
        smallest,
        np.float32(1),
        np.nextafter(np.float32(1), np.float32(2)),
        np.float32(2**40),
    ]:
        end = np.nextafter(start, np.float32(np.inf))
        for sign in [1, -1]:
            cases.append(
                np.array([[start, 0, 0], [end, 0, 0], [start, 1, 0], [start, 0, 1]], np.float32)
                * sign
            )
    for scale in [smallest, np.float32(2**-120), np.float32(2**100), np.finfo(np.float32).max]:
        cases.append(np.array([[0, 0, 0], [scale, 0, 0], [0, scale, 0], [0, 0, scale]], np.float32))
    cases.append(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 2, smallest]], np.float32))
    cases.append(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], np.float32))
    assert_reference(cases, exact=True)
    assert_reference(cases)


def test_random_float32_exponents_match_fraction_reference():
    rng = np.random.default_rng(109)
    bits = rng.integers(0, 2**32, size=(96, 4, 3), dtype=np.uint32)
    # Keep all bit patterns finite, retaining both signs and the full exponent range.
    bits[(bits & 0x7F800000) == 0x7F800000] &= np.uint32(0xFF7FFFFF)
    cases = bits.view(np.float32)
    assert_reference(cases)


def test_well_conditioned_fast_path_preserves_original_values():
    cases = np.random.default_rng(1109).normal(size=(512, 4, 3)).astype(np.float32)
    centers, _, radii, valid, fallback = evaluate_cases(cases)
    points = wp.array(cases, dtype=wp.vec3, device="cuda:0")
    raw_center = wp.empty(512, dtype=wp.vec3, device=points.device)
    raw_radius = wp.empty(512, dtype=wp.float64, device=points.device)
    wp.launch(original, 512, inputs=[points, raw_center, raw_radius])
    assert np.count_nonzero(fallback == 0) >= 500
    assert np.all(valid == 1)
    np.testing.assert_array_equal(centers[fallback == 0], raw_center.numpy()[fallback == 0])
    np.testing.assert_array_equal(radii[fallback == 0], raw_radius.numpy()[fallback == 0])
    assert_reference(cases)


def test_nonfinite_input_is_invalid():
    source = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
    cases = np.repeat(source[None], 3, axis=0)
    cases[:, 2, 1] = [np.nan, np.inf, -np.inf]
    _, _, _, valid, _ = evaluate_cases(cases)
    np.testing.assert_array_equal(valid, 0)


def test_exact_radius_is_stable_for_translated_small_cells():
    origin = np.array([2.0**80, 2.0**81, -(2.0**79)], np.float32)
    steps = np.spacing(np.abs(origin)).astype(np.float64)
    offsets = np.array([[0, 0, 0], [1, 2, -1], [-2, 1, 1], [3, -1, 2]], np.float64)
    source = (origin.astype(np.float64) + steps * offsets).astype(np.float32)
    assert_reference([source[list(order)] for order in permutations(range(4))], exact=True)


@wp.kernel
def produce(source: wp.array2d[wp.vec3], points: wp.array2d[wp.vec3], phase: wp.array[int]):
    i = wp.tid()
    points[0, i] = source[phase[0], i]


@wp.kernel
def consume(
    centers: wp.array[wp.vec3],
    valid: wp.array[int],
    fallback: wp.array[int],
    phase: wp.array[int],
    history: wp.array[wp.vec3],
    states: wp.array2d[int],
):
    row = phase[0]
    history[row] = centers[0]
    states[row, 0] = valid[0]
    states[row, 1] = fallback[0]
    phase[0] = (row + 1) % 4


def test_resident_center_producer_consumer_switches_paths_and_recovers():
    points = np.load(Path(__file__).parent / "data/hand-fit256.npz")["points"]
    hand = points[[14, 50, 96, 233]]
    cases = np.stack(
        [hand, hand, np.random.default_rng(1109).normal(size=(4, 3)), hand * 2]
    ).astype(np.float32)
    cases[1, 0, 0] = np.nan
    source = wp.array(cases, dtype=wp.vec3, device="cuda:0")
    current = wp.empty((1, 4), dtype=wp.vec3, device=source.device)
    outputs = [
        wp.empty(1, dtype=t, device=source.device)
        for t in [wp.vec3, wp.vec3d, wp.float64, int, int]
    ]
    history = wp.empty(4, dtype=wp.vec3, device=source.device)
    states = wp.empty((4, 2), dtype=int, device=source.device)
    phase = wp.zeros(1, dtype=int, device=source.device)
    wp.load_module(module=__name__, device=source.device)
    with wp.ScopedCapture() as capture:
        wp.launch(produce, 4, inputs=[source, current, phase])
        wp.launch(evaluate, 1, inputs=[current, False, *outputs])
        wp.launch(consume, 1, inputs=[outputs[0], outputs[3], outputs[4], phase, history, states])
    for _ in range(4):
        wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(states.numpy(), [[1, 1], [0, 1], [1, 0], [1, 1]])
    values = history.numpy()
    for i in [0, 2, 3]:
        center = exact_sphere(cases[i])[1]
        np.testing.assert_array_equal(values[i], [rounded_fraction(x) for x in center])
