"""Check actual captured construction, bitwise changes and reset/failure boundaries."""
import numpy as np
import warp as wp
from unchanged_input_observer_v1 import ObservedMesh


def test_bitwise_observation_and_fresh_failure_boundaries():
    wp.init()
    rng = np.random.default_rng(213)
    original = rng.normal(size=(32, 3)).astype(np.float32)
    original[0, 0] = 0.0
    points = wp.array(original, dtype=wp.vec3, device="cuda:0")
    mesh = ObservedMesh(points)
    with wp.ScopedCapture(device=points.device) as capture:
        mesh.run()
    saved = None
    for i, data in enumerate([original, original, original.copy(), original.copy(), original.copy(), original]):
        if i == 2:
            data[0, 0] = np.float32(-0.0)
        if i == 3:
            data[0, 0] = np.nan
        if i == 5:
            mesh.clear_reuse()
        points.assign(data)
        wp.capture_launch(capture.graph)
        wp.synchronize()
        if i != 3:
            assert mesh.error.numpy()[0] == 0 and mesh.tetrahedra_count.numpy()[0] > 0
            cells = {tuple(sorted(t)) for t in mesh.tetrahedra.numpy()[:mesh.tetrahedra_count.numpy()[0]]}
            if saved is None:
                saved = cells
            assert cells == saved
        else:
            assert mesh.error.numpy()[0] != 0 and mesh.tetrahedra_count.numpy()[0] == 0
    rows = mesh.observer.records.numpy()[:6]
    assert mesh.observer.calls.numpy()[0] == 6
    assert mesh.observer.overflow.numpy()[0] == 0
    np.testing.assert_array_equal(rows[:, 1], [0, 1, 1, 1, 0, 0])
    np.testing.assert_array_equal(rows[:, 2], [-1, 0, 1, 1, -1, -1])
    np.testing.assert_array_equal(rows[:, 0], [0, 0, 0, 0, 0, 1])
