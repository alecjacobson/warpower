import numpy as np
import pytest
import warp as wp

from warpower._canonical import canonicalize
from warpower._delaunay_certificate import DelaunayCertificate, check_spheres, check_spheres_points
from warpower._site_bvh import SiteHierarchy


def compare(points, inside):
    p = wp.array(points, dtype=wp.vec3, device="cuda:0")
    t = wp.array([[0, 1, 2, 3]], dtype=int, device=p.device)
    count = wp.ones(1, dtype=int, device=p.device)
    hierarchy = SiteHierarchy(p)
    weights = wp.zeros(len(p), dtype=float, device=p.device)
    owners, retained = canonicalize(p, weights)
    hierarchy.tree.owners = owners
    c = DelaunayCertificate(p, t, count, hierarchy.tree)
    c._anchors = None
    errors = []
    for kernel, stride in [(check_spheres, 1), (check_spheres_points, 32)]:

        def run_global(tree):
            c._launch(
                kernel,
                stride * len(c.tetrahedra),
                [p, c.tetrahedra, c.tetrahedra_count, tree, c.error, c.statistics],
            )

        c._global_spheres = run_global
        with wp.ScopedCapture() as capture:
            c.run()
        wp.capture_launch(capture.graph)
        errors.append(int(c.error.numpy()[0]))
    assert errors[0] == errors[1]
    assert bool(errors[1] & 8) == inside
    assert (errors[1] & 2) == 0


@pytest.mark.parametrize("exponent", [-120, -24, 0, 60, 100, 127])
@pytest.mark.parametrize("inside", [False, True])
def test_distributed_queries_match_global_tree_across_float32_scales(exponent, inside):
    scale = np.float32(2.0**exponent)
    points = (
        np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0.25, 0.25, 0.25] if inside else [1, 1, 1.5],
            ],
            np.float32,
        )
        * scale
    )
    # Repeated sites must be excluded using canonical owners, including a
    # repeated query. More than one warp of duplicates checks the strided tail.
    points = np.vstack([points, np.tile(points, (29, 1))])
    compare(points, inside)


@pytest.mark.parametrize("n", [127, 128, 257, 511, 512, 513, 1024, 1025])
@pytest.mark.parametrize("inside", [False, True])
def test_tiny_height_and_non_multiple_of_warp_size(n, inside):
    tiny = np.nextafter(np.float32(0), np.float32(1))
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, tiny], [0.25, 0.25, 0] if inside else [1, 1.5, 0]],
        np.float32,
    )
    points = np.vstack([points, np.repeat(points[:1], n - len(points), axis=0)])
    compare(points, inside)
