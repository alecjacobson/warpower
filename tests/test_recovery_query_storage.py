"""Complete-extraction recovery owns no unreachable approximate-query scratch."""

import warpower
from tests.test_domain import cube
from tests.test_mesher import validate
from warpower._topology_recovery import TopologyRecovery


def test_complete_recovery_omits_scratch_without_using_approximate_callback(monkeypatch):
    def forbidden(self):
        raise AssertionError("complete recovery reached approximate quality")

    monkeypatch.setattr(TopologyRecovery, "_approximate_quality", forbidden)
    domain = warpower.SurfaceDomain(*cube(), surface_subdivisions=4)
    mesher = warpower.VariationalMeshingWorkspace(domain, 128, initial_count=64, quality_rounds=1)
    fields = [
        "_query_cache",
        "_query_error",
        "_query_count",
        "_query_minimum",
        "_query_keep",
        "_query_quality",
    ]
    recovery = mesher.topology_recovery
    assert all(getattr(recovery, name) is None for name in fields)
    assert recovery.extraction._query_cache is not None
    for workspace in [mesher.jitter, mesher.interior]:
        assert all(getattr(workspace, name) is not None for name in fields)
    mesher.run()
    validate(mesher)
    assert recovery.invoked.numpy()[0] == 0
