from itertools import product

import numpy as np
import pytest
import warp as wp
from scipy.spatial import Delaunay

from tests.test_delaunay_recovery import valid
from warpower import DelaunayWorkspace
from warpower._canonical import canonicalize
from warpower._delaunay_certificate import DelaunayCertificate
from warpower._site_bvh import SiteHierarchy


def positive(points, tets):
    tets = np.asarray(tets, np.int32).copy()
    p = points[tets].astype(np.float64)
    signs = np.linalg.det(p[:, 1:] - p[:, :1])
    assert np.all(signs != 0)
    for i in np.flatnonzero(signs < 0):
        tets[i, [0, 1]] = tets[i, [1, 0]]
    return tets


def padded(points):
    points = np.asarray(points, np.float32)
    return np.concatenate([points, np.repeat(points[:1], max(0, 128 - len(points)), axis=0)])


def compare(points, tets):
    p = wp.array(points, dtype=wp.vec3, device="cuda:0")
    t = wp.array(tets, dtype=int, device=p.device)
    count = wp.full(1, len(tets), dtype=int, device=p.device)
    hierarchy = SiteHierarchy(p)
    weights = wp.zeros(len(p), dtype=float, device=p.device)
    owners, buffers = canonicalize(p, weights)
    hierarchy.tree.owners = owners
    results = []
    for local in [True, False]:
        c = DelaunayCertificate(p, t, count, hierarchy.tree)
        if not local:
            c._anchors = None
        with wp.ScopedCapture() as capture:
            c.run()
        wp.capture_launch(capture.graph)
        results.append((int(c.error.numpy()[0]), int(c._local.numpy()[0]) if local else -1))
    assert (results[0][0] == 0) == (results[1][0] == 0), results
    return results[0]


def test_random_mesh_uses_local_checks_and_rejects_holes_overlap_inversion():
    points = np.random.default_rng(541).normal(size=(200, 3)).astype(np.float32)
    tets = positive(points, Delaunay(points).simplices)
    assert compare(points, tets) == (0, 1)
    assert compare(points, tets[:-1])[0] != 0
    assert compare(points, np.vstack([tets, tets[:1]]))[0] & 32
    bad = tets.copy()
    bad[0, [0, 1]] = bad[0, [1, 0]]
    assert compare(points, bad)[0] & 4


@pytest.mark.parametrize("height", [0.1, 2.0])
def test_wrong_diagonal_of_convex_bipyramid_fails_local_and_global(height):
    points = padded([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.2, 0.2, 1], [0.2, 0.2, -height]])
    two = positive(points, [[0, 1, 2, 3], [0, 1, 2, 4]])
    three = positive(points, [[0, 1, 3, 4], [1, 2, 3, 4], [2, 0, 3, 4]])
    results = [compare(points, tets) for tets in [two, three]]
    assert all(local == 1 for _, local in results)
    assert sorted(error for error, _ in results) == [0, 8]


def test_missing_interior_site_is_rejected_even_with_local_convexity():
    points = padded([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.2, 0.2, 1], [0.2, 0.2, -2], [0.2, 0.2, 0]])
    tets = positive(points, [[0, 1, 2, 3], [0, 1, 2, 4]])
    error, local = compare(points, tets)
    assert local == 1
    assert error & 128


def test_cospherical_octahedron_ties_and_two_complete_overlapping_covers():
    points = padded([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    triangulations = []
    for a, b, ring in [(0, 1, [2, 4, 3, 5]), (2, 3, [0, 4, 1, 5]), (4, 5, [0, 2, 1, 3])]:
        triangulations.append(
            positive(points, [[a, b, ring[k], ring[(k + 1) % 4]] for k in range(4)])
        )
    results = [compare(points, tets) for tets in triangulations]
    assert all(local == 1 for _, local in results)
    assert sum(error == 0 for error, _ in results) == 1
    error, local = compare(points, np.vstack(triangulations[:2]))
    assert error & 32
    assert local == 0


def test_coplanar_hull_facets_fall_back_and_small_buffers_keep_original_path():
    small = np.asarray(list(product([-1, 1], repeat=3)), np.float32)
    mesh = DelaunayWorkspace(small).run()
    tets = valid(mesh, small)
    assert mesh._certificate._anchors is None
    assert compare(padded(small), tets) == (0, 0)


def test_two_cube_covers_with_disjoint_hull_diagonals_require_global_fallback():
    points = padded(list(product([-1, 1], repeat=3)))
    even = [[0, 3, 5, 6], [1, 0, 3, 5], [2, 0, 3, 6], [4, 0, 5, 6], [7, 3, 5, 6]]
    odd = [[1, 2, 4, 7], [0, 1, 2, 4], [3, 1, 2, 7], [5, 1, 4, 7], [6, 2, 4, 7]]
    # Each five-tet mesh covers the cube once. Their boundary diagonals differ
    # on every face: the union has no duplicate triangular boundary faces.
    # Indexed interior faces still pair and every boundary plane supports H,
    # while geometric coverage is two. There is no strict triangular anchor.
    tets = positive(points, even + odd)
    error, local = compare(points, tets)
    assert local == 0
    assert error == 8  # Sphere violation only, not incidence/support/usage.
