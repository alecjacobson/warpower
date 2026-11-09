"""Exact rational witness checks and analytic closed-material counterexamples."""

from fractions import Fraction

import numpy as np
import pytest
import warp as wp

from benchmarks.interior_containment_witness import compare_containment, inspect_witness
from tests.test_containment import cavity
from tests.test_domain import cube
from warpower import SurfaceDomain


def determinant(p):
    a = [[Fraction(float(x)) for x in row] for row in p]
    x, y, z = [[a[i][j] - a[0][j] for j in range(3)] for i in [1, 2, 3]]
    return (
        x[0] * (y[1] * z[2] - y[2] * z[1])
        - x[1] * (y[0] * z[2] - y[2] * z[0])
        + x[2] * (y[0] * z[1] - y[1] * z[0])
    )


def rational_interior(p, q):
    sign = np.sign(determinant(p))
    if sign == 0:
        return False
    for k in range(4):
        rows = p.copy()
        rows[k] = q
        if np.sign(determinant(rows)) != sign:
            return False
    return True


def paired(domain, points, *, count=None, capacity=None):
    n = len(points)
    capacity = n if capacity is None else capacity
    p = wp.array(points.reshape(-1, 3), dtype=wp.vec3, device=domain.device)
    cells = np.full((capacity, 4), -999, np.int32)
    cells[:n] = np.arange(4 * n).reshape(-1, 4)
    t = wp.array(cells, dtype=int, device=domain.device)
    count = wp.array([n], dtype=int, device=domain.device) if count is None else count
    before = wp.zeros(capacity, dtype=int, device=domain.device)
    after = wp.zeros((capacity, 4), dtype=int, device=domain.device)
    wp.launch(compare_containment, capacity, inputs=[domain.tree, p, t, count, before, after])
    return before.numpy(), after.numpy()


@pytest.mark.parametrize("scale", [2.0**-60, 1.0, 2.0**60])
def test_boundary_only_tetrahedron_gets_proved_material_witness(scale):
    domain = SurfaceDomain(*cube(scale))
    p = scale * np.array([[[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]]], np.float32)
    before, after = paired(domain, p)
    assert before.tolist() == [0]
    assert after.tolist() == [[1, 3, 0, 1]]


def test_cavity_boundary_and_crossing_tetrahedra_are_not_material():
    domain = SurfaceDomain(*cavity())
    p = np.array([[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]], np.float32)
    before, after = paired(domain, np.array([0.4 * p, 0.8 * p]))
    assert before.tolist() == [0, 0]
    assert after[:, 0].tolist() == [0, 0]


def test_rounded_proposals_match_exact_rational_strict_interior_test():
    rng = np.random.default_rng(2026)
    p = rng.uniform(-1, 1, (160, 4, 3)).astype(np.float32)
    p[::5, :, 2] *= np.float32(2**-60)
    p[1::5] *= np.float32(2**60)
    low, high = np.float32(1), np.nextafter(np.float32(1), np.float32(2))
    one_ulp = np.array([[low, low, low], [high, low, low], [low, high, low], [low, low, high]])
    flat = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], np.float32)
    p = np.concatenate([p, one_ulp[None], flat[None]])
    device = "cuda:0"
    array = wp.array(p, dtype=wp.vec3, device=device)
    witness, valid = (
        wp.empty(len(p), dtype=wp.vec3, device=device),
        wp.empty(len(p), dtype=int, device=device),
    )
    wp.launch(inspect_witness, len(p), inputs=[array, witness, valid])
    q, ok = witness.numpy(), valid.numpy()
    expected = [int(rational_interior(row, point)) for row, point in zip(p, q)]
    np.testing.assert_array_equal(ok, expected)
    assert ok[-2:].tolist() == [0, 0]


def test_inactive_bad_indices_are_never_read_and_changed_prefix_recovers():
    domain = SurfaceDomain(*cube())
    p = np.array([[[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]]], np.float32)
    count = wp.array([1], dtype=int, device=domain.device)
    for n in [1, 0, 1]:
        count.assign(np.array([n], np.int32))
        before, after = paired(domain, p, count=count, capacity=3)
        assert after[:, 0].tolist() == [n, 0, 0]
        assert not before.any()
