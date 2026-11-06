import numpy as np
import pytest
import warp as wp

import warpower
from tests.test_domain import combine, cube
from warpower._sampling import DoubleScan, InitialSamplingWorkspace, check_cdf


def sample(surface, n, **kwargs):
    sampler = InitialSamplingWorkspace(warpower.SurfaceDomain(*surface), n, **kwargs).run()
    assert sampler.error.numpy()[0] == 0, sampler.error.numpy()
    assert sampler.count.numpy()[0] == n
    p = sampler.points.numpy()
    assert len(np.unique(p, axis=0)) == n
    return sampler, p


@pytest.mark.parametrize("n", [1, 63, 65, 4099, 262147])
def test_double_scan_matches_high_precision_prefix_and_handles_partial_blocks(n):
    values = np.random.default_rng(43).uniform(0.1, 3.0, n)
    values[::7] = 0
    source = wp.array(values, dtype=wp.float64, device="cuda:0")
    target = wp.empty_like(source)
    scan = DoubleScan(n, source.device)
    scan.run(source, target)
    expected = np.cumsum(values.astype(np.longdouble)).astype(float)
    np.testing.assert_allclose(target.numpy(), expected, rtol=2e-15)


def test_cdf_zero_plateaus_are_monotone_after_roundoff_correction():
    rng = np.random.default_rng(2091)
    values = rng.uniform(0.1, 0.13, 32768)
    values[rng.random(len(values)) < 0.6] = 0
    source = wp.array(values, dtype=wp.float64, device="cuda:0")
    cdf = wp.empty_like(source)
    error = wp.zeros(1, dtype=int, device="cuda:0")
    scan = DoubleScan(len(values), source.device)
    scan.run(source, cdf)
    scan.run(cdf, cdf, maximum=True)
    wp.launch(check_cdf, len(values), inputs=[source, cdf, error])
    actual = cdf.numpy()
    assert error.numpy()[0] == 0
    assert np.all(np.diff(actual) >= 0)
    expected = np.cumsum(values.astype(np.longdouble)).astype(float)
    np.testing.assert_allclose(actual, expected, rtol=2e-15)


def test_probability_range_failure_is_explicit():
    source = wp.array([1e30, 1.0, 1.0], dtype=wp.float64, device="cuda:0")
    target = wp.empty_like(source)
    error = wp.zeros(1, dtype=int, device="cuda:0")
    scan = DoubleScan(3, source.device)
    scan.run(source, target)
    scan.run(target, target, maximum=True)
    wp.launch(check_cdf, 3, inputs=[source, target, error])
    assert error.numpy()[0] == 4


def test_uniform_material_distribution_seed_repeatability_and_unique_sites():
    sampler, points = sample(cube(), 20000, resolution=8)
    np.testing.assert_allclose(points.mean(axis=0), 0, atol=0.015)
    np.testing.assert_allclose(points.var(axis=0), 1 / 3, atol=0.015)
    assert np.all(np.abs(points) < 1)
    bins = np.histogramdd(points, bins=(4, 4, 4), range=((-1, 1),) * 3)[0]
    expected = len(points) / 64
    assert np.all(np.abs(bins - expected) < 6 * np.sqrt(expected))
    sampler.run()
    np.testing.assert_array_equal(sampler.points.numpy(), points)
    sampler.seed.fill_(9182)
    sampler.run()
    assert sampler.error.numpy()[0] == 0
    assert np.count_nonzero(sampler.points.numpy() != points) > 0.99 * points.size


def test_graded_density_matches_independent_volume_quadrature():
    locations = np.array([[-1, 0, 0], [1, 0.3, -0.2]], np.float32)
    sizes = np.array([0.3, 1.2], np.float32)
    field = warpower.SizingField(
        locations, wp.array(sizes, device="cuda:0"), gradation=0.5, minimum=0.25, maximum=1.5
    )
    sampler, points = sample(cube(), 30000, sizing=field, resolution=8)
    axis = (np.arange(96) + 0.5) / 48 - 1
    q = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    mu = np.full(q.shape[:-1], 1.5)
    for p, value in zip(locations.astype(float), sizes.astype(float)):
        mu = np.minimum(mu, value + 0.5 * np.linalg.norm(q - p, axis=-1))
    density = np.clip(mu, 0.25, 1.5) ** -3
    probabilities = density.reshape(4, 24, 4, 24, 4, 24).sum(axis=(1, 3, 5)) / density.sum()
    observed = np.histogramdd(points, bins=(4, 4, 4), range=((-1, 1),) * 3)[0]
    expected = len(points) * probabilities
    assert probabilities.max() > 8 * probabilities.min()
    tolerance = 6 * np.sqrt(expected * (1 - probabilities)) + 0.002 * expected
    assert np.all(np.abs(observed - expected) < tolerance)
    assert sampler.attempts.numpy().max() < sampler.max_attempts


@pytest.mark.parametrize("seed", [2027, 2032, 2034])
def test_large_graded_clouds_keep_random_streams_separate(seed):
    field = warpower.SizingField(
        np.array([[-1, 0, 0]], np.float32),
        wp.array([0.2], dtype=float, device="cuda:0"),
        gradation=0.7,
    )
    sampler, points = sample(cube(), 100000, sizing=field, resolution=8, seed=seed)
    assert sampler.attempts.numpy().max() > 1
    assert np.all(np.abs(points) < 1)


def test_rotated_thin_material_is_retained():
    p, faces = cube()
    p[:, 2] *= 0.0125
    rotation, _ = np.linalg.qr(np.array([[1.0, 0.3, 0.8], [0.2, 0.9, -0.4], [0.7, -0.2, 1.0]]))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    sampler, points = sample(((p @ rotation.T).astype(np.float32), faces), 4000, resolution=8)
    local = points @ rotation
    assert np.all(np.abs(local[:, :2]) < 1.000001)
    assert np.all(np.abs(local[:, 2]) < 0.012501)
    np.testing.assert_allclose(np.mean(local**2, axis=0), [1 / 3, 1 / 3, 0.0125**2 / 3], rtol=0.08)
    assert sampler.attempts.numpy().max() > 1


def test_small_component_is_not_removed_when_its_grid_centers_are_outside():
    sampler, points = sample(combine(cube(), cube(0.12, (1.8, 0.37, 0.29))), 12000, resolution=4)
    in_small = points[:, 0] > 1
    assert in_small.sum() > 0
    assert np.all(np.abs(points[~in_small]) < 1)
    assert np.all(np.abs(points[in_small] - [1.8, 0.37, 0.29]) < 0.120001)
    lo, hi = sampler.domain.tree.lower.numpy()[1], sampler.domain.tree.upper.numpy()[1]
    xs = lo[0] + (np.arange(4) + 0.5) * (hi[0] - lo[0]) / 4
    assert not np.any((xs > 1.68) & (xs < 1.92))
    probability = 0.12**3 / (1 + 0.12**3)
    assert abs(in_small.sum() - len(points) * probability) < 6 * np.sqrt(len(points) * probability)


def test_cavity_is_excluded_from_samples():
    p, f = cube(0.4)
    _, points = sample(combine(cube(), (p, f[:, ::-1].copy())), 4000, resolution=8)
    assert np.all(np.abs(points) < 1)
    assert np.all(np.any(np.abs(points) > 0.4, axis=1))


def test_attempt_budget_failure_publishes_no_partial_cloud():
    p, f = cube(0.8)
    sampler = InitialSamplingWorkspace(
        warpower.SurfaceDomain(*combine(cube(), (p, f[:, ::-1].copy()))),
        2000,
        resolution=1,
        max_attempts=1,
    ).run()
    assert sampler.error.numpy()[0] == 8
    assert sampler.count.numpy()[0] == 0
    np.testing.assert_array_equal(sampler.attempts.numpy(), 1)


def test_unresolvable_float32_density_is_an_explicit_duplicate_error():
    sampler = InitialSamplingWorkspace(
        warpower.SurfaceDomain(*cube(8, (2**24, 2**24, 2**24))), 4096, resolution=4
    ).run()
    assert sampler.error.numpy()[0] == 16
    assert sampler.count.numpy()[0] == 0


@wp.kernel
def producer(
    source: wp.array[wp.vec3],
    surface: wp.array[wp.vec3],
    values: wp.array[float],
    phase: wp.array[int],
    seed: wp.array[int],
):
    i = wp.tid()
    surface[i] = source[i] * (1.0 + float(wp.max(phase[0], 0)))
    if phase[0] == -1 and i == 0:
        surface[i] = wp.vec3(wp.nan)
    if i == 0:
        values[0] = 1.0
        if phase[0] == -2:
            values[0] = -1.0
        seed[0] += 1


@wp.kernel
def consume(points: wp.array[wp.vec3], count: wp.array[int], total: wp.array[wp.vec3d]):
    i = wp.tid()
    if i < count[0]:
        wp.atomic_add(total, 0, wp.vec3d(points[i]))


def test_capture_producer_refits_sampling_consumer_failure_and_recovery(monkeypatch, tmp_path):
    p, f = cube()
    source = wp.array(p, dtype=wp.vec3, device="cuda:0")
    domain = warpower.SurfaceDomain(p, f)
    values = wp.ones(1, device="cuda:0")
    field = warpower.SizingField(np.zeros((1, 3), np.float32), values)
    sampler = InitialSamplingWorkspace(domain, 257, sizing=field, resolution=4)
    phase = wp.zeros(1, dtype=int, device="cuda:0")
    total = wp.zeros(1, dtype=wp.vec3d, device="cuda:0")
    wp.load_module(module=__name__, device="cuda:0")
    arrays = [sampler.points, sampler.count, sampler.error, sampler.attempts, sampler._cdf]
    pointers = [a.ptr for a in arrays]

    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected host readback, allocation or synchronization")

    with monkeypatch.context() as patch:
        for name in (
            "empty",
            "zeros",
            "ones",
            "full",
            "empty_like",
            "clone",
            "synchronize",
            "synchronize_device",
            "synchronize_stream",
        ):
            patch.setattr(wp, name, forbidden)
        patch.setattr(wp.array, "numpy", forbidden)
        with wp.ScopedCapture() as capture:
            wp.launch(producer, len(p), inputs=[source, domain.points, values, phase, sampler.seed])
            domain.refit()
            field.refit()
            sampler.run()
            total.zero_()
            wp.launch(consume, sampler.n, inputs=[sampler.points, sampler.count, total])
        wp.capture_launch(capture.graph)
    for step in [0, 1, -1, -2, 2, 0]:
        phase.fill_(step)
        wp.capture_launch(capture.graph)
        assert [a.ptr for a in arrays] == pointers
        if step < 0:
            assert sampler.error.numpy()[0] == (1 if step == -1 else 2)
            assert sampler.count.numpy()[0] == 0
            np.testing.assert_array_equal(sampler.attempts.numpy(), 0)
            np.testing.assert_array_equal(total.numpy(), 0)
        else:
            assert sampler.error.numpy()[0] == 0
            assert sampler.count.numpy()[0] == sampler.n
            points = sampler.points.numpy()
            assert np.all(np.abs(points) < 1 + step)
            np.testing.assert_allclose(
                total.numpy()[0], points.astype(float).sum(axis=0), atol=1e-12
            )
    path = tmp_path / "sampling.dot"
    wp.capture_debug_dot_print(capture.graph, str(path), verbose=True)
    graph = path.read_text()
    assert all(node not in graph for node in ("MEM_ALLOC", "MEM_FREE", "HOST"))
