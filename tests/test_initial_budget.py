import numpy as np
import pytest
import warp as wp

from tests.test_domain import cube
from tests.test_mesher import validate
from warpower import InitialSamplingWorkspace, SurfaceDomain, VariationalMeshingWorkspace
from warpower._budgeted_sampling import BudgetedSamplingWorkspace


def test_device_prefix_samples_match_fixed_count_and_recover():
    domain = SurfaceDomain(*cube())
    requested = wp.array([4], dtype=int, device=domain.device)
    dynamic = BudgetedSamplingWorkspace(domain, 64, requested)
    with wp.ScopedCapture() as capture:
        dynamic.run()
    for n in [4, 32, 64, 1, 0, 65, 16]:
        requested.fill_(n)
        wp.capture_launch(capture.graph)
        if not 1 <= n <= 64:
            assert dynamic.error.numpy()[0] == 64 and dynamic.count.numpy()[0] == 0
            continue
        fixed = InitialSamplingWorkspace(domain, n).run()
        assert dynamic.error.numpy()[0] == 0 and dynamic.count.numpy()[0] == n
        np.testing.assert_array_equal(dynamic.points.numpy()[:n], fixed.points.numpy())
        np.testing.assert_array_equal(dynamic.attempts.numpy()[:n], fixed.attempts.numpy())
        np.testing.assert_array_equal(dynamic.points.numpy()[n:], 0)


@pytest.mark.parametrize("initial", [4, 8, 16])
def test_initial_budget_reserves_mandatory_corners(initial):
    domain = SurfaceDomain(*cube(), surface_subdivisions=4)
    mesh = VariationalMeshingWorkspace(domain, 64, initial_count=initial, quality_rounds=0).run()
    validate(mesh)
    assert mesh.effective_initial_count.numpy()[0] == max(initial, 12)
    assert mesh.sampler.count.numpy()[0] == max(initial, 12)
    assert mesh.initial_count == initial


def test_dynamic_target_resolves_automatic_initial_budget_before_sampling():
    domain = SurfaceDomain(*cube(), surface_subdivisions=4)
    target = wp.array([32], dtype=int, device=domain.device)
    mesh = VariationalMeshingWorkspace(domain, 64, target_count=target, quality_rounds=0)
    with wp.ScopedCapture() as capture:
        mesh.run()
    for n in [32, 64, 4, 32]:
        target.fill_(n)
        wp.capture_launch(capture.graph)
        if n < 8:
            assert mesh.error.numpy()[0] == 1 and mesh.valid.numpy()[0] == 0
            np.testing.assert_array_equal(mesh.counts.numpy(), 0)
        else:
            validate(mesh)
            assert mesh.sampler.count.numpy()[0] == n


def test_corner_count_can_change_between_captured_fresh_solves():
    domain = SurfaceDomain(*cube(), surface_subdivisions=4)
    mesh = VariationalMeshingWorkspace(domain, 64, initial_count=4, quality_rounds=0)
    with wp.ScopedCapture() as capture:
        mesh.run()
    # Exercise the device feature-prefix contract directly, independently of
    # which edges a particular surface refit classifies as sharp.
    for corners in [4, 8, 0, 8]:
        domain.corner_count.fill_(corners)
        wp.capture_launch(capture.graph)
        validate(mesh)
        assert mesh.effective_initial_count.numpy()[0] == max(4, corners + 4)
