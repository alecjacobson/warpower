"""Analytic strict-intersection controls for the offline rational diagnostic."""

from fractions import Fraction

import numpy as np
import pytest

from benchmarks.corner_intersections import clip_triangle, rational_points, verify_certificate

TET = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
CASES = [
    ([[0.1, 0.1, 0.1], [0.2, 0.1, 0.1], [0.1, 0.2, 0.1]], True, True),
    ([[-1, -1, 0.25], [4, -1, 0.25], [-1, 4, 0.25]], True, True),
    ([[0, 0, 0], [1, 0, 0], [0, 1, 0]], True, False),
    ([[0, 0, 0], [-1, 0, 0], [0, -1, 0]], True, False),
    ([[0, 0, 0], [1, 0, 0], [0, -1, -1]], True, False),
    ([[1, 1, 1], [2, 1, 1], [1, 2, 1]], False, False),
    ([[0.9, 0.9, 0], [0.9, 0, 0.9], [0, 0.9, 0.9]], False, False),
    ([[0.2, 0.2, -1], [0.2, 0.2, 2], [0.4, 0.4, 0.25]], True, True),
]


@pytest.mark.parametrize("triangle,contact,strict", CASES)
@pytest.mark.parametrize("scale", [2.0**-120, 1.0, 2.0**120])
def test_analytic_contacts_and_crossings(triangle, contact, strict, scale):
    tet = rational_points(TET * np.float32(scale))
    tri = rational_points(np.array(triangle, np.float32) * np.float32(scale))
    actual, certificate = clip_triangle(tet, tri)
    assert actual == contact and (certificate is not None) == strict
    if certificate:
        verify_certificate(tet, tri, certificate)


def test_thin_one_ulp_and_subnormal_tetrahedra():
    for scale in [2.0**-149, 2.0**-100, 1.0]:
        tet = rational_points(TET * np.float32(scale))
        # Exact rational test points need not be representable in binary32.
        tri = [tuple((p[i] + tet[3][i]) / 4 for i in range(3)) for p in tet[:3]]
        contact, certificate = clip_triangle(tet, tri)
        assert contact and certificate is not None
    low = np.float32(1)
    high = np.nextafter(low, np.float32(2))
    tiny = rational_points(
        np.array(
            [[low, low, low], [high, low, low], [low, high, low], [low, low, high]], np.float32
        )
    )
    tri = [tuple((p[i] + tiny[3][i]) / 2 for i in range(3)) for p in tiny[:3]]
    assert clip_triangle(tiny, tri)[1] is not None


def test_winding_permutations_do_not_change_intersection():
    tet = rational_points(TET)
    tri = rational_points(np.array(CASES[1][0], np.float32))
    for t in [tet, tet[::-1], [tet[i] for i in [1, 0, 2, 3]]]:
        for q in [tri, tri[::-1]]:
            assert clip_triangle(t, q)[1] is not None


def test_degenerate_tet_and_invalid_certificates_reject():
    tet = rational_points(TET)
    with pytest.raises(ValueError):
        clip_triangle([tet[0]] * 4, tet[:3])
    for cert in [(Fraction(1), 0, 0), (Fraction(-1), 1, 1), (Fraction(1), 1, 1)]:
        with pytest.raises(AssertionError):
            verify_certificate(tet, tet[:3], cert)
