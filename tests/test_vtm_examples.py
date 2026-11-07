import numpy as np
import pytest
import warp as wp

from examples.vtm_inputs import counterbored_plate, surface
from tests.test_extraction import independent_topology
from warpower import SurfaceDomain, VariationalMeshingWorkspace
from warpower._domain_topology import DomainTopology


@pytest.mark.parametrize(
    "name,topology,volume",
    [
        ("thin-hand", [1, 1, 0, 2], 1.041),
        ("torus", [1, 1, 0, 0], 19.06963794),
        ("cavity", [1, 2, 1, 4], 8 * (1 - 0.4**3)),
        ("disconnected", [2, 2, 0, 4], 16 * 0.7**3),
    ],
)
def test_example_domains_have_expected_geometry_and_material_topology(name, topology, volume):
    domain = SurfaceDomain(*surface(name), surface_subdivisions=1)
    assert domain.error.numpy()[0] == 0
    check = DomainTopology(domain).run()
    assert check.error.numpy()[0] == 0
    np.testing.assert_array_equal(check.counts.numpy(), topology)
    assert np.isclose(check.volume.numpy()[0], volume, rtol=1e-6)


@pytest.mark.parametrize("segments", [16, 32])
def test_counterbore_preserves_hole_shoulder_and_actual_crease_corner_constraints(segments):
    points, faces = counterbored_plate(segments)
    domain = SurfaceDomain(points, faces, surface_subdivisions=1, crease_subdivisions=1)
    assert domain.error.numpy()[0] == 0
    topology = DomainTopology(domain).run()
    assert topology.error.numpy()[0] == 0
    np.testing.assert_array_equal(topology.counts.numpy(), [1, 1, 0, 0])
    # Independent cross-section integration of the polygonal bore. The extra
    # corner rays split four circle sectors; no face triangulation is reused.
    theta = np.sort(
        np.r_[
            2 * np.pi * np.arange(segments) / segments,
            [np.arctan2(y, x) % (2 * np.pi) for x in [-1.25, 1.25] for y in [-0.85, 0.85]],
        ]
    )
    unit_area = 0.5 * np.sin(np.diff(np.r_[theta, theta[0] + 2 * np.pi])).sum()
    expected_volume = 0.7 * 2.5 * 1.7 - unit_area * (0.4 * 0.35**2 + 0.3 * 0.65**2)
    assert np.isclose(topology.volume.numpy()[0], expected_volume, rtol=1e-6)
    assert domain.feature_count.numpy()[0] == 6 * (segments + 4) + 4
    assert domain.corner_count.numpy()[0] == 8
    actual = domain.corner_points.numpy()[:8]
    expected = np.array(
        [[x, y, z] for x in [-1.25, 1.25] for y in [-0.85, 0.85] for z in [-0.35, 0.35]],
        np.float32,
    )
    assert sorted(map(tuple, actual)) == sorted(map(tuple, expected))


def test_counterbored_plate_captured_mesher_retains_hole_and_all_eight_corners():
    domain = SurfaceDomain(*counterbored_plate(), surface_subdivisions=16, crease_subdivisions=16)
    mesh = VariationalMeshingWorkspace(domain, 1000, initial_count=256, seed=2026)
    with wp.ScopedCapture() as capture:
        domain.refit()
        mesh.run()
    wp.capture_launch(capture.graph)
    wp.synchronize()
    assert mesh.valid.numpy()[0] == 1
    assert mesh.error.numpy()[0] == 0
    n, _, m, _ = mesh.counts.numpy()
    assert n == 1000
    p = mesh.points.numpy()[:n]
    t = mesh.tetrahedra.numpy()[:m]
    cells = p[t].astype(np.float64)
    assert np.all(np.linalg.det(cells[:, 1:] - cells[:, :1]) > 0)
    boundary, components, euler = independent_topology(p, t)
    assert (components, euler) == (1, 0)
    bp = p[np.unique(boundary)]
    for corner in domain.corner_points.numpy()[:8]:
        assert np.any(np.all(bp == corner, axis=1))
    assert mesh.radius_ratio.numpy()[:m].min() > 0.08
