"""Compare staged speculative checks with the original and global certificates."""
from itertools import product
import numpy as np
import pytest
import warp as wp
from scipy.spatial import Delaunay
from tests.test_local_certificate import positive, padded
from tests.test_delaunay_recovery import valid
from tests.test_tetrahedra import tet_set
from warpower import DelaunayWorkspace
from warpower._delaunay_certificate import DelaunayCertificate
from warpower._canonical import canonicalize
from warpower._site_bvh import SiteHierarchy
from staged_previous_certificate import check_previous


def compare(points, tets, expected, *, count=None):
    p = wp.array(points, dtype=wp.vec3, device='cuda:0')
    t = wp.array(tets, dtype=int, device=p.device)
    n = wp.full(1, len(tets) if count is None else count, dtype=int, device=p.device)
    h = SiteHierarchy(p)
    weights = wp.zeros(len(p), dtype=float, device=p.device)
    owners, owner_buffers = canonicalize(p, weights)
    h.tree.owners = owners
    results = []
    for mode in ['original', 'staged', 'global']:
        c = DelaunayCertificate(p, t, n, h.tree)
        if mode == 'global':
            c._anchors = None
        with wp.ScopedCapture() as capture:
            if mode == 'staged':
                check_previous(c, h.tree)
            else:
                c.run()
        wp.capture_launch(capture.graph)
        error = int(c.error.numpy()[0])
        results.append(error)
        if error == 0:
            assert expected is not None
            assert tet_set(c.tetrahedra.numpy()[:c.tetrahedra_count.numpy()[0]]) == expected
    assert [r == 0 for r in results] == [expected is not None] * 3, results
    return results


@pytest.mark.parametrize('n', [32, 200])
@pytest.mark.parametrize('exponent', [-60, 0, 60])
def test_scale_extremes_and_malformed_complete_candidates(n, exponent):
    base = np.random.default_rng(182).normal(size=(n, 3)).astype(np.float32)
    tets = positive(base, Delaunay(base).simplices)
    points = base * np.float32(2.0 ** exponent)
    compare(points, tets, tet_set(tets))
    compare(points, tets[:-1], None)
    compare(points, np.vstack([tets, tets[:1]]), None)
    bad = tets.copy(); bad[0, [0, 1]] = bad[0, [1, 0]]
    compare(points, bad, None)
    for bad_id in [-1, n]:
        bad = tets.copy(); bad[0, 0] = bad_id
        compare(points, bad, None)
    for count in [-1, 0, len(tets) + 1]:
        compare(points, tets, None, count=count)


def test_missing_site_and_bipyramid_diagonals():
    for height in [.1, 2.0]:
        points = padded([[0,0,0], [1,0,0], [0,1,0], [.2,.2,1], [.2,.2,-height]])
        reference = DelaunayWorkspace(points).run()
        expected = tet_set(valid(reference, points))
        for cells in [[[0,1,2,3], [0,1,2,4]],
                      [[0,1,3,4], [1,2,3,4], [2,0,3,4]]]:
            tets = positive(points, cells)
            compare(points, tets, expected if tet_set(tets) == expected else None)
    points = padded([[0,0,0], [1,0,0], [0,1,0], [.2,.2,1], [.2,.2,-2], [.2,.2,0]])
    compare(points, positive(points, [[0,1,2,3], [0,1,2,4]]), None)


def test_ties_coplanar_hull_and_overlapping_covers():
    cube = padded(list(product([-1, 1], repeat=3)))
    w = DelaunayWorkspace(cube).run()
    tets = valid(w, cube)
    assert compare(cube, tets, tet_set(tets)) == [0, 0, 0]
    even = [[0,3,5,6], [1,0,3,5], [2,0,3,6], [4,0,5,6], [7,3,5,6]]
    odd = [[1,2,4,7], [0,1,2,4], [3,1,2,7], [5,1,4,7], [6,2,4,7]]
    compare(cube, positive(cube, even + odd), None)
    octa = padded([[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]])
    w = DelaunayWorkspace(octa).run()
    expected = tet_set(valid(w, octa))
    triangulations = []
    for a, b, ring in [(0,1,[2,4,3,5]), (2,3,[0,4,1,5]), (4,5,[0,2,1,3])]:
        tets = positive(octa, [[a,b,ring[k],ring[(k+1)%4]] for k in range(4)])
        compare(octa, tets, expected if tet_set(tets) == expected else None)
        triangulations.append(tets)
    compare(octa, np.vstack(triangulations[:2]), None)
