"""Small complete repaired trajectory for unfiltered memory instrumentation."""

import numpy as np
import warp as wp

import warpower
from tests.test_domain import cube
from tests.test_mesher import validate


def test_small_repaired_graded_trajectory_includes_quality_round():
    domain = warpower.SurfaceDomain(*cube(), surface_subdivisions=6)
    field = warpower.SizingField(
        np.array([[-0.8, 0, 0], [0.8, 0, 0]], np.float32),
        wp.array([0.2, 0.5], dtype=float, device=domain.device),
        gradation=0.3,
    )
    source = wp.array(
        np.random.default_rng(2026).uniform(-0.9, 0.9, (128, 3)).astype(np.float32),
        dtype=wp.vec3,
        device=domain.device,
    )
    mesh = warpower.VariationalMeshingWorkspace(
        domain,
        136,
        target_count=128,
        initial_count=128,
        sizing=field,
        polish_iterations=5,
        quality_rounds=1,
        max_recovery_rounds=0,
    )

    def sampling(stream=None):
        wp.copy(mesh.sampler.points, source)
        mesh.sampler.count.fill_(128)
        mesh.sampler.error.zero_()

    mesh.sampler.run = sampling
    with wp.ScopedCapture() as cap:
        mesh.run()
    wp.capture_launch(cap.graph)
    validate(mesh)
    assert mesh._material_recovery.numpy()[0] == 1
    assert mesh.rounds.numpy()[1] == 0 and mesh.rounds.numpy()[2] == 1
    assert mesh.core.selection.recovery_extraction is not None
    assert mesh.jitter.extraction.containment_recovery
    assert mesh.interior.extraction.containment_recovery
