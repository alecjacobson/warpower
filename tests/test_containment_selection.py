"""The quality guard must include cells added by material-boundary recovery."""

import numpy as np
import warp as wp

import warpower
from tests.test_containment import cavity, fan
from warpower._selective import SelectiveUpdate


def test_device_selected_quality_uses_the_recovered_subcomplex_and_recovers():
    class Core:
        pass

    core = Core()
    p, t = fan()
    core.domain = warpower.SurfaceDomain(*cavity())
    core.device = core.domain.device
    base = wp.array(p, dtype=wp.vec3, device=core.device)
    core.points = wp.clone(base)
    core.full_tetrahedra = wp.array(t, device=core.device)
    core.full_tetrahedra_count = wp.full(1, len(t), dtype=int, device=core.device)
    core.n = len(p)
    core.mesh = Core()
    core.mesh.error = wp.zeros(1, dtype=int, device=core.device)
    core.material_recovery = wp.zeros(1, dtype=int, device=core.device)
    selection = SelectiveUpdate(core, 0.4)
    with wp.ScopedCapture() as capture:
        selection.evaluate()
    wp.capture_launch(capture.graph)
    assert selection.error.numpy()[0] == 0
    assert selection.retained_count.numpy()[0] == 3
    core.material_recovery.fill_(1)
    wp.capture_launch(capture.graph)
    assert selection.error.numpy()[0] == 0
    assert selection.retained_count.numpy()[0] == 5
    extraction = selection.recovery_extraction
    assert extraction.repaired_count.numpy()[0] == 2
    np.testing.assert_array_equal(selection.keep.numpy(), extraction.selected.numpy())
    np.testing.assert_allclose(selection.minimum.numpy(), extraction.minimum_radius_ratio.numpy())
    core.points.fill_(wp.vec3(float("nan")))
    wp.capture_launch(capture.graph)
    assert selection.error.numpy()[0] != 0
    assert selection.retained_count.numpy()[0] == 0
    wp.copy(core.points, base)
    wp.capture_launch(capture.graph)
    assert selection.error.numpy()[0] == 0 and selection.retained_count.numpy()[0] == 5
    core.material_recovery.zero_()
    wp.capture_launch(capture.graph)
    assert selection.error.numpy()[0] == 0 and selection.retained_count.numpy()[0] == 3
