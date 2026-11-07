"""Complete coarse-cavity regression including repaired quality transactions."""

import numpy as np
import warp as wp

import warpower
from tests.test_containment import cavity
from tests.test_mesher import validate


def test_coarse_cavity_completes_four_quality_rounds_with_containment_recovery():
    domain = warpower.SurfaceDomain(*cavity(), surface_subdivisions=4, crease_subdivisions=4)
    mesh = warpower.VariationalMeshingWorkspace(
        domain, 1000, initial_count=256, seed=2026, quality_rounds=4
    )
    with wp.ScopedCapture() as cap:
        domain.refit()
        mesh.run()
    wp.capture_launch(cap.graph)
    validate(mesh)
    np.testing.assert_array_equal(mesh.topology.numpy(), [1, 2, 1, 4])
    assert mesh.rounds.numpy()[2] == 4
    assert mesh.rounds.numpy()[1] == 0
    assert mesh._material_recovery.numpy()[0] == 1
    assert mesh.core.selection.error.numpy()[0] == 0
    assert mesh.jitter.error.numpy()[0] == mesh.interior.error.numpy()[0] == 0
    assert mesh.minimum_radius_ratio.numpy()[0] > 0.09
    assert mesh.moves.numpy()[0] > 0 and mesh.moves.numpy()[1] > 0
