"""Saved real-input regression: extraction and selection use the same recovery."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp

import warpower
from tests.test_extraction import extracted
from warpower._selective import SelectiveUpdate


def test_scanned_failure_recovers_and_quality_guard_measures_the_same_subcomplex():
    path = (
        Path(__file__).parents[1]
        / "results/validation/vtm/graded-armadillo-failure"
        / "failed-candidate.npz"
    )
    data = np.load(path)
    d = warpower.SurfaceDomain(
        data["input_points"],
        data["input_faces"],
        feature_angle=120,
        surface_subdivisions=1,
        crease_subdivisions=1,
    )
    points = wp.array(data["points"], dtype=wp.vec3, device=d.device)
    full = data["full_tetrahedra"]
    tets = wp.array(np.vstack([full, [-1, -1, -1, -1]]).astype(np.int32), device=d.device)
    count = wp.full(1, len(full), dtype=int, device=d.device)
    error = wp.zeros(1, dtype=int, device=d.device)
    roles = wp.array(data["roles"], device=d.device)
    e = warpower.ExtractionWorkspace(
        d,
        points,
        tets,
        count,
        source_error=error,
        containment_recovery=True,
        exterior_recovery=True,
        feature_roles=roles,
    )
    with wp.ScopedCapture() as cap:
        e.run()
    wp.capture_launch(cap.graph)
    p, actual, _ = extracted(e)
    expected = data["selected"].copy()
    expected[[2187, 23964]] = 0
    np.testing.assert_array_equal(e.selected.numpy(), np.r_[expected, 0])
    np.testing.assert_array_equal(actual, full[expected != 0])
    np.testing.assert_array_equal(np.unique(actual), np.unique(full[data["selected"] != 0]))
    np.testing.assert_array_equal(e.selected.numpy()[: len(full)][data["signed_distance"] <= 0], 1)
    assert e.removed_count.numpy()[0] == 2 and e.exterior_rejected.numpy()[0] == 0

    core = SimpleNamespace(
        domain=d,
        device=d.device,
        points=points,
        full_tetrahedra=tets,
        full_tetrahedra_count=count,
        n=len(points),
        mesh=SimpleNamespace(error=error),
        boundary=SimpleNamespace(roles=roles),
        containment_recovery=True,
        exterior_recovery=True,
        material_recovery=wp.ones(1, dtype=int, device=d.device),
    )

    # SelectiveUpdate holds a weak reference; use a weak-referenceable object.
    class Core:
        pass

    owner = Core()
    owner.__dict__.update(core.__dict__)
    selection = SelectiveUpdate(owner, 0.4)
    with wp.ScopedCapture() as cap_selection:
        selection.evaluate()
    wp.capture_launch(cap_selection.graph)
    assert selection.error.numpy()[0] == 0
    np.testing.assert_array_equal(selection.keep.numpy(), e.selected.numpy())
    cells = p[actual].astype(np.float64)
    edges = cells[:, 1:] - cells[:, :1]
    determinants = np.linalg.det(edges)
    centers = np.linalg.solve(2 * edges, np.sum(edges * edges, axis=2)[..., None])[..., 0]
    areas = sum(
        np.linalg.norm(np.cross(cells[:, b] - cells[:, a], cells[:, c] - cells[:, a]), axis=1)
        for a, b, c in [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    )
    q = 3 * determinants / (areas * np.linalg.norm(centers, axis=1))
    np.testing.assert_allclose(selection.minimum.numpy(), q.min(), rtol=1e-10)
    owner.material_recovery.zero_()
    wp.capture_launch(cap_selection.graph)
    assert selection.retained_count.numpy()[0] == int(data["selected"].sum())
    np.testing.assert_array_equal(selection.keep.numpy()[: len(full)], data["selected"])

    # Invalid and empty frames reset all outward-recovery statistics; restoring
    # upstream inputs must recover without reallocating or recapturing.
    error.fill_(1)
    wp.capture_launch(cap.graph)
    assert e.error.numpy()[0] == 2 and e.tetrahedra_count.numpy()[0] == 0
    assert e.removed_count.numpy()[0] == e.exterior_rounds.numpy()[0] == 0
    error.zero_()
    count.zero_()
    wp.capture_launch(cap.graph)
    assert e.valid.numpy()[0] == 1 and e.tetrahedra_count.numpy()[0] == 0
    assert e.removed_count.numpy()[0] == 0
    count.fill_(len(full))
    wp.capture_launch(cap.graph)
    assert e.valid.numpy()[0] == 1 and e.removed_count.numpy()[0] == 2
    np.testing.assert_array_equal(e.selected.numpy(), np.r_[expected, 0])
