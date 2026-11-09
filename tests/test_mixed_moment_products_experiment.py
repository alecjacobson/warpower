"""Numerical and complete boundary controls for the isolated product ablation."""

import numpy as np
import warp as wp

from benchmarks.mixed_moment_products import accumulate_products32, moment_product
from tests.test_domain import cube
from warpower import BoundaryWorkspace, SurfaceDomain, _mixed_boundary


@wp.kernel
def products(weights: wp.array[wp.float64], delta: wp.array[wp.vec3d], out: wp.array[wp.vec3d]):
    i = wp.tid()
    out[i] = moment_product(weights[i], delta[i])


def test_product_arithmetic_and_range_fallback():
    rng = np.random.default_rng(274)
    weights = np.concatenate([rng.uniform(0.01, 10, 128), [2.0**120, 2.0**-120, 2.0**-120]])
    delta = np.concatenate(
        [rng.uniform(-1, 1, (128, 3)), [[2.0**40] * 3, [2.0**-20] * 3, [2.0**-80] * 3]]
    )
    with np.errstate(over="ignore", under="ignore"):
        expected = (weights[:, None].astype(np.float32) * delta.astype(np.float32)).astype(float)
    expected[-3:] = weights[-3:, None] * delta[-3:]
    out = wp.empty(len(weights), dtype=wp.vec3d, device="cuda:0")
    wp.launch(
        products,
        len(weights),
        inputs=[
            wp.array(weights, dtype=wp.float64, device="cuda:0"),
            wp.array(delta, dtype=wp.vec3d, device="cuda:0"),
            out,
        ],
    )
    np.testing.assert_array_equal(out.numpy(), expected)


def fits(domain, points, active):
    baseline = BoundaryWorkspace(domain, points, active_count=active, query_precision="mixed")
    original = _mixed_boundary.accumulate_samples_mixed
    wp.load_module(module="benchmarks.mixed_moment_products", device=domain.device)
    _mixed_boundary.accumulate_samples_mixed = accumulate_products32
    try:
        candidate = BoundaryWorkspace(domain, points, active_count=active, query_precision="mixed")
    finally:
        _mixed_boundary.accumulate_samples_mixed = original
    return baseline, candidate


def test_boundary_owners_masses_and_corner_anchors_stay_reference():
    domain = SurfaceDomain(*cube(), surface_subdivisions=3, crease_subdivisions=3)
    p = np.random.default_rng(274).uniform(-0.9, 0.9, (64, 3)).astype(np.float32)
    points = wp.array(p, dtype=wp.vec3, device=domain.device)
    active = wp.array([64], dtype=int, device=domain.device)
    baseline, candidate = fits(domain, points, active)
    baseline.run()
    candidate.run()
    assert baseline.error.numpy()[0] == candidate.error.numpy()[0] == 0
    for name in ["surface_owners", "crease_owners", "roles", "corner_owners", "corner_sites"]:
        np.testing.assert_array_equal(
            getattr(baseline, name).numpy(), getattr(candidate, name).numpy()
        )
    for name in ["surface_mass", "crease_mass", "energy", "total_weight"]:
        np.testing.assert_allclose(
            getattr(baseline, name).numpy(), getattr(candidate, name).numpy(), rtol=1e-14, atol=0
        )
    corner = baseline.roles.numpy() == 3
    np.testing.assert_array_equal(
        candidate.targets.numpy()[corner], baseline.targets.numpy()[corner]
    )
    delta = np.abs(candidate.targets.numpy().astype(float) - baseline.targets.numpy().astype(float))
    assert delta.max() < 3e-7


@wp.kernel
def producer(
    source: wp.array[wp.vec3], points: wp.array[wp.vec3], phase: wp.array[int], count: wp.array[int]
):
    i = wp.tid()
    points[i] = source[i]
    if phase[0] == 1:
        points[i] = wp.vec3(wp.nan)
    if i == 0:
        count[0] = 64


@wp.kernel
def consumer(phase: wp.array[int], error: wp.array[int], log: wp.array[int]):
    log[phase[0]] = error[0]
    phase[0] += 1


def test_captured_changing_points_failure_recovery_without_host_work(monkeypatch):
    domain = SurfaceDomain(*cube(), surface_subdivisions=2)
    source = wp.array(
        np.random.default_rng(274).uniform(-0.9, 0.9, (64, 3)).astype(np.float32),
        dtype=wp.vec3,
        device=domain.device,
    )
    points = wp.clone(source)
    active = wp.array([64], dtype=int, device=domain.device)
    _, fit = fits(domain, points, active)
    phase = wp.zeros(1, dtype=int, device=domain.device)
    log = wp.empty(3, dtype=int, device=domain.device)
    wp.load_module(module=__name__, device=domain.device)

    def forbidden(*a, **kw):
        raise AssertionError("host operation during capture/replay")

    with monkeypatch.context() as patch:
        for name in [
            "empty",
            "zeros",
            "ones",
            "full",
            "clone",
            "empty_like",
            "synchronize",
            "synchronize_device",
            "synchronize_stream",
        ]:
            patch.setattr(wp, name, forbidden)
        patch.setattr(wp.array, "numpy", forbidden)
        with wp.ScopedCapture() as cap:
            wp.launch(producer, 64, inputs=[source, points, phase, active])
            fit.run()
            wp.launch(consumer, 1, inputs=[phase, fit.error, log])
        for _ in range(3):
            wp.capture_launch(cap.graph)
    assert log.numpy()[0] == log.numpy()[2] == 0
    assert log.numpy()[1] != 0
