import numpy as np
import pytest

from examples.vtm_inputs import surface
from warpower import SurfaceDomain
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
