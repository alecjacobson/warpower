from fractions import Fraction
from itertools import product

import numpy as np
import warp as wp

from tests.test_delaunay_recovery import valid
from tests.test_sphere_bounds import det
from warpower import DelaunayWorkspace
from warpower import _delaunay_certificate as dc

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.kernel
def plane_boxes(
    points: wp.array[wp.vec3], lo: wp.array[wp.vec3], hi: wp.array[wp.vec3], inside: wp.array[int]
):
    i = wp.tid()
    plane = dc.halfspace(points[3 * i], points[3 * i + 1], points[3 * i + 2])
    inside[i] = int(dc.inside_halfspace(plane, lo[i], hi[i]))


def test_cached_support_planes_are_conservative_at_extreme_scales_and_offsets():
    rng = np.random.default_rng(8401)
    triangles, lowers, uppers = [], [], []
    for exponent, shift in [
        (-140, 0),
        (-126, 0),
        (-60, 0),
        (0, 0),
        (60, 0),
        (120, 0),
        (0, 4096),
        (100, 4096),
    ]:
        raw = rng.normal(size=(80, 3, 3))
        normal = np.cross(raw[:, 1] - raw[:, 0], raw[:, 2] - raw[:, 0])
        normal[::2] *= -1
        center = raw[:, 0] - 0.5 * normal
        triangles.extend(((raw + shift) * 2.0**exponent).astype(np.float32))
        lowers.extend(((center - 0.02 + shift) * 2.0**exponent).astype(np.float32))
        uppers.extend(((center + 0.02 + shift) * 2.0**exponent).astype(np.float32))
    p = wp.array(np.array(triangles).reshape(-1, 3), dtype=wp.vec3, device="cuda:0")
    lo = wp.array(lowers, dtype=wp.vec3, device=p.device)
    hi = wp.array(uppers, dtype=wp.vec3, device=p.device)
    output = wp.empty(len(lo), dtype=int, device=p.device)
    wp.launch(plane_boxes, len(lo), inputs=[p, lo, hi, output])
    accepted = output.numpy()
    assert accepted.sum() > 200
    for triangle, lower, upper, good in zip(triangles, lowers, uppers, accepted):
        if not good:
            continue
        a, b, c = [[Fraction(float(x)) for x in row] for row in triangle]
        u = [x - y for x, y in zip(b, a)]
        v = [x - y for x, y in zip(c, a)]
        for corner in product(*zip(lower, upper)):
            w = [Fraction(float(x)) - y for x, y in zip(corner, a)]
            assert det(u, v, w) <= 0


def test_boundary_compaction_reserves_all_four_faces_of_one_tetrahedron():
    p = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
    w = DelaunayWorkspace(p, tetrahedra_capacity=1).run()
    cells = valid(w, p)
    assert len(cells) == 1
    assert w._certificate._boundary_count.numpy()[0] == 4
    assert len(w._certificate._boundary) == 4


def test_partitioned_support_search_covers_unbalanced_trees_and_coincident_sites():
    from scipy.spatial import Delaunay

    from tests.test_delaunay_recovery import certificate

    rng = np.random.default_rng(8421)
    unique = rng.uniform(-1, 1, (64, 3)).astype(np.float32)
    unique[:48] *= 2.0**-12
    # Different radix depths, early leaves, and canonical duplicate ownership.
    points = np.concatenate([unique, np.repeat(unique[:1], 192, axis=0)])
    cells = Delaunay(unique).simplices.astype(np.int32)
    for row in cells:
        a, b, c, d = points[row].astype(np.float64)
        if np.linalg.det(np.stack([b - a, c - a, d - a])) < 0:
            row[0], row[1] = row[1], row[0]
    assert certificate(points, cells) == 0
    # Removing a cell leaves only valid lower facets, so rejection requires
    # complete support-face traversal rather than a nonempty-sphere shortcut.
    assert certificate(points, cells[1:]) & 64
