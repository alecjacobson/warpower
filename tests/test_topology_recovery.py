"""Captured topology-recovery contracts, policy scoping and failure propagation."""

import numpy as np
import pytest
import warp as wp

import warpower
from tests.test_domain import cube
from tests.test_mesher import validate
from warpower._exterior_band import remember_pinched
from warpower._mesh_topology import MeshTopology
from warpower._mesher import State
from warpower._quality_moves import MoveState
from warpower._topology_recovery import (
    eligible,
    propose,
    references_and_roles,
    reject_changed_references,
    restored,
)


def test_captured_eligibility_rejects_stale_witness_and_recovers():
    device = "cuda:0"
    state = State()
    state.error = wp.zeros(1, dtype=int, device=device)
    state.active = wp.full(1, 4, dtype=int, device=device)
    state.extraction_reason = wp.full(1, 4, dtype=int, device=device)
    core = wp.zeros(1, dtype=int, device=device)
    extraction = wp.zeros(1, dtype=int, device=device)
    witness = wp.full(1, 2, dtype=int, device=device)
    rejected = wp.ones(1, dtype=int, device=device)
    roles = wp.ones(4, dtype=int, device=device)
    chosen = wp.full(1, -1, dtype=int, device=device)
    ready = wp.zeros(1, dtype=int, device=device)
    with wp.ScopedCapture(device=device) as cap:
        wp.launch(
            eligible,
            1,
            inputs=[state, core, extraction, witness, rejected, roles, chosen, ready],
            device=device,
        )
    # Each rejected case follows a successful replay, so stale success is tested.
    for array, value, original in [
        (rejected, 0, 1),
        (state.error, 1, 0),
        (core, 1, 0),
        (extraction, 1, 0),
        (state.extraction_reason, 0, 4),
        (witness, 4, 2),
        (witness, -1, 2),
        (witness, 2**31 - 1, 2),
        (roles, 2, 1),
        (roles, 3, 1),
        (state.active, 2, 4),
    ]:
        wp.capture_launch(cap.graph)
        assert ready.numpy()[0] == 1 and chosen.numpy()[0] == 2
        array.fill_(value)
        wp.capture_launch(cap.graph)
        assert ready.numpy()[0] == 0 and chosen.numpy()[0] == -1
        array.fill_(original)
    wp.capture_launch(cap.graph)
    assert ready.numpy()[0] == 1 and chosen.numpy()[0] == 2


def test_captured_failed_restore_invalidates_full_count_with_precise_errors():
    device = "cuda:0"
    state = MoveState()
    state.error = wp.zeros(1, dtype=int, device=device)
    state.source_count = wp.full(1, 17, dtype=int, device=device)
    core = wp.zeros(1, dtype=int, device=device)
    errors = [wp.zeros(1, dtype=int, device=device) for _ in range(3)]
    with wp.ScopedCapture(device=device) as cap:
        wp.launch(restored, 1, inputs=[state, core, *errors], device=device)
    # The arguments are mesh, invalid geometry, boundary (bits 1,4,2).
    for bits in [0, 1, 2, 3, 4, 5, 6, 7, 0]:
        for array, bit in zip(errors, [1, 4, 2]):
            array.fill_(int(bool(bits & bit)))
        state.source_count.fill_(17)
        state.error.fill_(2)
        wp.capture_launch(cap.graph)
        assert core.numpy()[0] == bits
        assert state.error.numpy()[0] == (10 if bits else 2)
        assert state.source_count.numpy()[0] == (0 if bits else 17)


def test_recovery_policy_scopes_storage_and_healthy_solve_skips_search():
    d = warpower.SurfaceDomain(*cube(), surface_subdivisions=4)
    m = warpower.VariationalMeshingWorkspace(d, 128, initial_count=64, quality_rounds=1)
    recovery = m.topology_recovery
    assert m.extraction._exterior.pinched_site is m._pinched_site
    assert recovery.extraction._exterior.pinched_site is None
    assert recovery.extraction._proven_exterior is None
    for workspace in [m.jitter, m.interior]:
        assert workspace.extraction._exterior.pinched_site is None
    assert m.core.selection.recovery_extraction._exterior.pinched_site is None
    for array in [
        recovery.point_count,
        recovery.full_count,
        recovery.retained_count,
        recovery.metrics,
        recovery.previous_metrics,
        recovery.topology,
        recovery.previous_topology,
        recovery.last_errors,
    ]:
        array.fill_(17)
    m.run()
    validate(m)
    assert (
        recovery.invoked.numpy()[0]
        == recovery.accepted.numpy()[0]
        == recovery.attempts.numpy()[0]
        == 0
    )
    assert recovery.chosen.numpy()[0] == -1
    for array in [
        recovery.point_count,
        recovery.full_count,
        recovery.retained_count,
        recovery.metrics,
        recovery.previous_metrics,
        recovery.topology,
        recovery.previous_topology,
        recovery.last_errors,
    ]:
        np.testing.assert_array_equal(array.numpy(), 0)
    disabled = warpower.VariationalMeshingWorkspace(
        d, 32, initial_count=32, quality_rounds=0, tangential_recovery=False
    )
    assert disabled.topology_recovery is disabled._pinched_site is None
    assert disabled.extraction._exterior.pinched_site is None
    implied = warpower.VariationalMeshingWorkspace(
        d, 32, initial_count=32, quality_rounds=0, exterior_recovery=False
    )
    assert implied.topology_recovery is implied._pinched_site is None
    with pytest.raises(ValueError, match="requires exterior_recovery"):
        warpower.VariationalMeshingWorkspace(
            d, 32, tangential_recovery=True, exterior_recovery=False
        )


def test_captured_link_witness_counts_features_errors_and_recovery():
    wp.init()
    device = wp.get_device("cuda:0")
    stream = wp.Stream(device)
    p = wp.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]],
        dtype=wp.vec3,
        device=device,
    )
    t = wp.array([[0, 1, 2, 3], [0, 4, 6, 5]], dtype=int, device=device)
    count = wp.full(1, 2, dtype=int, device=device)
    top = MeshTopology(p, t, count)
    roles = wp.ones(7, dtype=int, device=device)
    error = wp.zeros(1, dtype=int, device=device)
    chosen = wp.full(1, 2**31 - 1, dtype=int, device=device)
    with wp.ScopedStream(stream), wp.ScopedCapture(stream=stream) as capture:
        top.run()
        chosen.fill_(2**31 - 1)
        wp.launch(
            remember_pinched,
            7,
            inputs=[top._first, top._degree, top._twins, error, top.diagnostics, roles, chosen],
            device=device,
        )
    for number, role, err, expected in [
        (2, 1, 0, 0),
        (1, 1, 0, 2**31 - 1),
        (2, 2, 0, 2**31 - 1),
        (2, 1, 1, 2**31 - 1),
        (2, 1, 0, 0),
    ]:
        count.assign(np.array([number], np.int32))
        r = np.ones(7, np.int32)
        r[0] = role
        roles.assign(r)
        error.assign(np.array([err], np.int32))
        wp.synchronize_device(device)
        wp.capture_launch(capture.graph, stream=stream)
        wp.synchronize_stream(stream)
        assert int(chosen.numpy()[0]) == expected, (
            number,
            role,
            err,
            int(top.diagnostics.numpy()[0]),
        )
        assert int(top.diagnostics.numpy()[0]) == (4 if number == 2 else 0)


@pytest.mark.parametrize(
    "missing,role_change,active,expected",
    [(False, False, 4, 0), (True, False, 4, 1), (False, True, 4, 1), (True, True, 2, 0)],
)
def test_reference_and_role_changes_cannot_accept(missing, role_change, active, expected):
    wp.init()
    device = "cuda:0"
    before = wp.array([1, 1, 1, 0], dtype=int, device=device)
    after = wp.array([1, 1, 0 if missing else 1, 1], dtype=int, device=device)
    old = wp.array([1, 2, 1, 0], dtype=int, device=device)
    new = wp.array([1, 2, 0 if role_change else 1, 0], dtype=int, device=device)
    n = wp.array([active], dtype=int, device=device)
    rejected = wp.zeros(1, dtype=int, device=device)
    state = MoveState()
    state.accepted = wp.ones(1, dtype=int, device=device)
    state.rejection_reason = wp.zeros(1, dtype=int, device=device)
    with wp.ScopedCapture(device=device) as cap:
        rejected.zero_()
        state.accepted.fill_(1)
        state.rejection_reason.zero_()
        wp.launch(
            references_and_roles, 4, inputs=[before, after, old, new, n, rejected], device=device
        )
        wp.launch(reject_changed_references, 1, inputs=[state, rejected], device=device)
    wp.capture_launch(cap.graph)
    wp.synchronize()
    assert rejected.numpy()[0] == expected and state.accepted.numpy()[0] == 1 - expected
    assert state.rejection_reason.numpy()[0] == 256 * expected


def test_float32_proposals_stay_inside_their_ring_or_are_rejected():
    wp.init()
    device = "cuda:0"
    state = MoveState()
    state.attempts = wp.zeros(1, dtype=int, device=device)
    state.site = wp.full(1, -1, dtype=int, device=device)
    state.position = wp.zeros(1, dtype=wp.vec3, device=device)
    state.proposal_valid = wp.zeros(1, dtype=int, device=device)
    original = wp.zeros(1, dtype=wp.vec3, device=device)
    chosen = wp.zeros(1, dtype=int, device=device)
    frame = wp.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=wp.vec3d, device=device)
    limit = wp.array([0.2], dtype=wp.float64, device=device)
    with wp.ScopedCapture(device=device) as cap:
        wp.launch(propose, 1, inputs=[state, original, chosen, frame, limit], device=device)
    for attempt in range(1, 33):
        state.attempts.assign(np.array([attempt], np.int32))
        wp.capture_launch(cap.graph)
        wp.synchronize()
        assert state.proposal_valid.numpy()[0] == 1
        distance = np.linalg.norm(state.position.numpy()[0].astype(float))
        assert 0 < distance <= (0.1 if attempt <= 16 else 0.2)
    for value in [1e20, float("inf"), float("nan")]:
        original.assign(np.array([[value, value, value]], np.float32))
        wp.capture_launch(cap.graph)
        wp.synchronize()
        assert state.proposal_valid.numpy()[0] == 0
