"""Optional topology retries preserve fresh witnesses and captured failure recovery."""

import numpy as np
import pytest
import warp as wp

from benchmarks.mesher_loop import record
from tests.test_domain import cube
from tests.test_mesher import produce
from tests.test_vtm_core import consume
from warpower import SurfaceDomain, VariationalMeshingWorkspace
from warpower._mesh_topology import MeshTopology
from warpower._topology_backtracking import block_moved, eligible, flag_edges


@pytest.mark.parametrize(
    "changes,expected",
    [
        ({}, 1),
        ({"before": 4}, 0),
        ({"accepted": 1}, 0),
        ({"mode": 0}, 0),
        ({"recovered": 0}, 0),
        ({"reason": 4}, 0),
        ({"selection_error": 32}, 0),
        ({"mesh_error": 8}, 0),
        ({"boundary_error": 1}, 0),
        ({"extraction_error": 16}, 0),
        ({"extraction_valid": 1}, 0),
        ({"diagnostics": 6}, 0),
        ({"used": 1}, 0),
    ],
)
def test_rejection_freshness_budget_and_other_failures(changes, expected):
    values = dict(
        mode=1,
        recovered=1,
        accepted=0,
        rejections=4,
        before=3,
        reason=3,
        selection_error=16,
        mesh_error=0,
        boundary_error=0,
        extraction_error=0,
        extraction_valid=0,
        diagnostics=2,
        used=0,
    )
    values.update(changes)
    arrays = [wp.array([v], dtype=int, device="cuda:0") for v in values.values()]
    ready = wp.ones(1, dtype=int, device="cuda:0")
    stats = wp.zeros(4, dtype=int, device="cuda:0")
    wp.launch(eligible, 1, inputs=[*arrays, 1, ready, stats])
    assert ready.numpy()[0] == expected


def test_actual_nonmanifold_edges_respect_active_prefix_masks_and_pinned_sites():
    # Two positive tetrahedra touching only on an edge give a nonmanifold boundary.
    # Inactive padding is deliberately invalid and must never enter a query.
    p = np.array(
        [[0, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [99, 99, 99]],
        np.float32,
    )
    cells = np.array([[0, 1, 2, 3], [0, 1, 4, 5]], np.int32)
    for row in cells:
        if np.linalg.det((p[row[1:]] - p[row[0]]).T) < 0:
            row[2], row[3] = row[3], row[2]
    with wp.ScopedDevice("cuda:0"):
        original, points = wp.array(p, dtype=wp.vec3), wp.array(p + 0.1, dtype=wp.vec3)
        count, active = wp.array([2], dtype=int), wp.array([6], dtype=int)
        tets = wp.array(np.vstack([cells, np.full((3, 4), -999, np.int32)]), dtype=int)
        topology = MeshTopology(points, tets, count)
        bad, blocked = [wp.zeros(len(p), dtype=int) for _ in range(2)]
        any_blocked, statistics = wp.zeros(1, dtype=int), wp.zeros(4, dtype=int)
        roles, mask = wp.zeros(len(p), dtype=int), wp.ones(len(p), dtype=int)
        wp.load_module(module="warpower._topology_backtracking", device="cuda:0")
        with wp.ScopedCapture() as cap:
            for value in [bad, blocked, any_blocked, statistics]:
                value.zero_()
            topology.run()
            wp.launch(
                flag_edges,
                len(topology.boundary_indices),
                inputs=[
                    topology.boundary_indices,
                    topology.boundary_count,
                    topology._edge_slots,
                    topology._edge_counts,
                    topology._edge_balance,
                    bad,
                ],
            )
            wp.launch(
                block_moved,
                len(p),
                inputs=[
                    active,
                    bad,
                    roles,
                    mask,
                    original,
                    points,
                    blocked,
                    any_blocked,
                    statistics,
                ],
            )
        for n, corner, disabled, expected in [
            (2, False, False, [0, 1]),
            (1, False, False, []),
            (2, True, False, [1]),
            (2, False, True, [0]),
            (2, False, False, [0, 1]),
        ]:
            count.assign(np.array([n], np.int32))
            roles.assign(np.array([3 if corner else 0, 0, 0, 0, 0, 0, 0], np.int32))
            mask.assign(np.array([1, 0 if disabled else 1, 1, 1, 1, 1, 1], np.int32))
            wp.capture_launch(cap.graph)
            np.testing.assert_array_equal(np.flatnonzero(blocked.numpy()), expected)
            assert statistics.numpy()[1] == len(expected)


@wp.kernel
def poison_statistics(stats: wp.array[int]):
    stats[wp.tid()] = 12345


@pytest.mark.parametrize("budget", [1, 2])
def test_complete_producer_consumer_changes_invalid_inputs_and_recovers(budget, monkeypatch):
    d = SurfaceDomain(*cube(), surface_subdivisions=4)
    source = wp.clone(d.points)
    target = wp.full(1, 96, dtype=int, device=d.device)
    seed = wp.full(1, 2026, dtype=int, device=d.device)
    phase = wp.zeros(1, dtype=int, device=d.device)
    volume = wp.zeros(1, dtype=wp.float64, device=d.device)
    log = wp.zeros((5, 5), dtype=int, device=d.device)
    volumes = wp.zeros(5, dtype=wp.float64, device=d.device)
    mesh = VariationalMeshingWorkspace(
        d,
        128,
        target_count=target,
        initial_count=64,
        seed=seed,
        quality_rounds=1,
        local_topology_retries=budget,
    )
    trial = mesh.core.local_backtracking
    for module in (__name__, "tests.test_mesher", "tests.test_vtm_core", "benchmarks.mesher_loop"):
        wp.load_module(module=module, device=d.device)

    def forbidden(*args, **kwargs):
        raise AssertionError("host activity or allocation during capture/replay")

    with monkeypatch.context() as patch:
        for name in [
            "empty",
            "zeros",
            "ones",
            "full",
            "empty_like",
            "clone",
            "synchronize",
            "synchronize_device",
            "synchronize_stream",
        ]:
            patch.setattr(wp, name, forbidden)
        patch.setattr(wp.array, "numpy", forbidden)
        with wp.ScopedCapture() as cap:
            wp.launch(poison_statistics, 4, inputs=[trial.statistics])
            wp.launch(produce, len(source), inputs=[source, d.points, phase, target, seed])
            d.refit()
            mesh.run()
            volume.zero_()
            wp.launch(
                consume,
                len(mesh.tetrahedra),
                inputs=[
                    mesh.points,
                    mesh.tetrahedra,
                    mesh.tetrahedra_count,
                    mesh.error,
                    volume,
                ],
            )
            wp.launch(record, 1, inputs=[phase, mesh.error, mesh.counts, volume, log, volumes])
        for _ in range(5):
            wp.capture_launch(cap.graph)
    values, consumed = log.numpy(), volumes.numpy()
    assert values[:, 0].tolist() == [0, 0, 1, 1, 0]
    assert values[:, 1].tolist() == [96, 128, 0, 0, 96]
    assert not values[2:4, 1:].any() and not consumed[2:4].any()
    assert all(consumed[i] > 0 for i in [0, 1, 4])
    assert not trial.statistics.numpy().any()
    assert not trial.blocked.numpy().any()


def test_api_rejects_unsupported_combinations_and_invalid_budgets():
    d = SurfaceDomain(*cube(), surface_subdivisions=1)
    for budget in [-1, 2**31]:
        with pytest.raises(ValueError, match="local_topology_retries"):
            VariationalMeshingWorkspace(d, 64, local_topology_retries=budget)
    with pytest.raises(TypeError):
        VariationalMeshingWorkspace(d, 64, local_topology_retries=0.5)
    for options in [
        dict(quality_rounds=0),
        dict(local_quality_retries=1),
        dict(containment_recovery=False, exterior_recovery=False),
    ]:
        with pytest.raises(ValueError, match="local_topology_retries"):
            VariationalMeshingWorkspace(d, 64, local_topology_retries=1, **options)


@wp.kernel
def force_edge_counts(
    enabled: wp.array[int], count: wp.array[int], slots: wp.array[int], occurrences: wp.array[int]
):
    i = wp.tid()
    if enabled[0] and i < 3 * count[0]:
        wp.atomic_exch(occurrences, slots[i], 4)


@wp.kernel
def force_rejection(
    enabled: wp.array[int],
    accepted: wp.array[int],
    rejections: wp.array[int],
    error: wp.array[int],
    reason: wp.array[int],
    extraction_error: wp.array[int],
    valid: wp.array[int],
    diagnostics: wp.array[int],
):
    if enabled[0]:
        accepted[0] = 0
        rejections[0] += 1
        error[0] = 16
        reason[0] = 3
        extraction_error[0] = 0
        valid[0] = 0
        diagnostics[0] = 2


@pytest.mark.parametrize("budget", [1, 2])
def test_forced_fresh_failure_exhausts_local_budget_restores_and_recovers(budget):
    from tests.test_selective import initial
    from tests.test_vtm_core import valid_core
    from warpower import _vtm

    fitted = initial(64, 5)
    points = fitted.points.numpy().copy()
    active = wp.array([64], dtype=int, device=fitted.device)
    mode, recovery = [wp.ones(1, dtype=int, device=fitted.device) for _ in range(2)]
    core = _vtm.VTMWorkspace(
        fitted.domain,
        points.copy(),
        max_iterations=1,
        max_backtracks=2,
        active_count=active,
        selective_threshold=1.0,
        selective_enabled=mode,
        material_recovery=recovery,
        _local_topology_retries=budget,
    )
    enabled = wp.ones(1, dtype=int, device=core.device)
    selection = core.selection
    extraction = selection.recovery_extraction
    topology = extraction._topology
    original_check = selection.check

    def check():
        original_check()
        # Fault injection tests transaction routing; real geometric edge witnesses
        # are tested separately above and in the independently checked scan runs.
        wp.launch(
            force_edge_counts,
            len(topology.boundary_indices),
            inputs=[
                enabled,
                topology.boundary_count,
                topology._edge_slots,
                topology._edge_counts,
            ],
        )
        wp.launch(
            force_rejection,
            1,
            inputs=[
                enabled,
                core._accepted,
                selection.rejections,
                selection.error,
                selection.last_error,
                extraction.error,
                extraction.valid,
                extraction.diagnostics,
            ],
        )

    selection.check = check
    wp.load_module(module=__name__, device=core.device)
    with wp.ScopedCapture() as cap:
        core.local_backtracking.statistics.zero_()
        core.run()
    wp.capture_launch(cap.graph)
    actual, _ = valid_core(core)
    np.testing.assert_array_equal(actual.view(np.uint32), points.view(np.uint32))
    assert core.iterations.numpy()[0] == 0
    assert 0 < core.local_backtracking.statistics.numpy()[0] <= budget
    assert core.local_backtracking.used.numpy()[0] <= budget
    assert 3 <= core.rejected_steps.numpy()[0] <= 3 + budget
    enabled.zero_()
    core.points.assign(points)
    wp.capture_launch(cap.graph)
    actual, _ = valid_core(core)
    fresh = _vtm.VTMWorkspace(
        fitted.domain,
        points.copy(),
        max_iterations=1,
        max_backtracks=2,
        active_count=active,
        selective_threshold=1.0,
        selective_enabled=mode,
        material_recovery=recovery,
        _local_topology_retries=budget,
    )
    fresh.run()
    expected, _ = valid_core(fresh)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-8)
    for name in ["iterations", "termination", "rejected_steps"]:
        np.testing.assert_array_equal(getattr(core, name).numpy(), getattr(fresh, name).numpy())
