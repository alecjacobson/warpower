"""Diagnostic decomposition of the unchanged multi-workspace policy test.

Each named case must run in a fresh instrumented process. This does not claim
combined-process instrumentation passed; all original assertions are preserved.
"""
from tests.test_topology_recovery import *

def test_isolated_policy_healthy():
    d = warpower.SurfaceDomain(*cube(), surface_subdivisions=4)
    m = warpower.VariationalMeshingWorkspace(d, 128, initial_count=64, quality_rounds=1)
    recovery = m.topology_recovery
    assert m.extraction._exterior.pinched_site is m._pinched_site
    assert recovery.extraction._exterior.pinched_site is None
    assert recovery.extraction._proven_exterior is None
    for workspace in [m.jitter, m.interior]:
        assert workspace.extraction._exterior.pinched_site is None
    assert m.core.selection.recovery_extraction._exterior.pinched_site is None
    for array in [recovery.point_count, recovery.full_count, recovery.retained_count, recovery.metrics, recovery.previous_metrics, recovery.topology, recovery.previous_topology, recovery.last_errors]:
        array.fill_(17)
    m.run()
    validate(m)
    assert recovery.invoked.numpy()[0] == recovery.accepted.numpy()[0] == recovery.attempts.numpy()[0] == 0
    assert recovery.chosen.numpy()[0] == -1
    for array in [recovery.point_count, recovery.full_count, recovery.retained_count, recovery.metrics, recovery.previous_metrics, recovery.topology, recovery.previous_topology, recovery.last_errors]:
        np.testing.assert_array_equal(array.numpy(), 0)

def test_isolated_policy_disabled():
    d = warpower.SurfaceDomain(*cube(), surface_subdivisions=4)
    disabled = warpower.VariationalMeshingWorkspace(d, 32, initial_count=32, quality_rounds=0, tangential_recovery=False)
    assert disabled.topology_recovery is disabled._pinched_site is None
    assert disabled.extraction._exterior.pinched_site is None

def test_isolated_policy_implied():
    d = warpower.SurfaceDomain(*cube(), surface_subdivisions=4)
    implied = warpower.VariationalMeshingWorkspace(d, 32, initial_count=32, quality_rounds=0, exterior_recovery=False)
    assert implied.topology_recovery is implied._pinched_site is None
    with pytest.raises(ValueError, match='requires exterior_recovery'):
        warpower.VariationalMeshingWorkspace(d, 32, tangential_recovery=True, exterior_recovery=False)

