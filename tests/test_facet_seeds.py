from itertools import combinations, product

import numpy as np
import pytest
import warp as wp

from tests.test_facet_completion import canonical
from warpower import DelaunayWorkspace
from warpower._canonical import canonicalize
from warpower._facet_seeds import CertifiedSeeds
from warpower._site_bvh import SiteHierarchy


def fixture(points, cells):
    p = wp.array(np.asarray(points, np.float32), dtype=wp.vec3, device="cuda:0")
    t = wp.array(np.asarray(cells, np.int32), dtype=int, device=p.device)
    count = wp.full(1, len(t), dtype=int, device=p.device)
    hierarchy = SiteHierarchy(p)
    weights = wp.zeros(len(p), dtype=float, device=p.device)
    owners, retained = canonicalize(p, weights)
    hierarchy.tree.owners = owners
    selected = CertifiedSeeds(p, t, count, hierarchy.tree)
    with wp.ScopedCapture() as capture:
        selected.run()
    return selected, capture.graph, (hierarchy, owners, retained, weights)


def output(selected, graph):
    wp.capture_launch(graph)
    assert selected.error.numpy()[0] == 0
    return selected.tetrahedra.numpy()[: selected.tetrahedra_count.numpy()[0]]


@pytest.mark.parametrize("shape", ["cube", "octahedron"])
@pytest.mark.parametrize("scale", [2.0**-60, 1.0, 2.0**60])
def test_all_candidate_combinations_reduce_to_exact_symbolic_lower_facets(shape, scale):
    points = np.array(list(product([-1.0, 1.0], repeat=3)), np.float32)
    if shape == "octahedron":
        points = np.vstack([np.eye(3), -np.eye(3)]).astype(np.float32)
    reference = DelaunayWorkspace(points).run()
    assert reference.error.numpy()[0] == 0
    expected = reference.tetrahedra.numpy()[: reference.tetrahedra_count.numpy()[0]]
    points = np.vstack([points, points[:2]]) * np.float32(scale)
    candidates = np.array(list(combinations(range(len(points)), 4)), np.int32)
    # Both orientation conventions, exact zero-volume cells and duplicate
    # canonical owners occur among these exhaustive candidate combinations.
    candidates = np.vstack([candidates, candidates[:, ::-1], candidates[:, [1, 0, 2, 3]]])
    selected, graph, retained = fixture(points, candidates)
    actual = output(selected, graph)
    np.testing.assert_array_equal(canonical(actual), canonical(expected))


def test_malformed_cells_and_nonempty_sphere_are_discarded_without_reading_indices():
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.25, 0.25, 0.25]], np.float32)
    reference = DelaunayWorkspace(points).run()
    assert reference.error.numpy()[0] == 0
    expected = reference.tetrahedra.numpy()[: reference.tetrahedra_count.numpy()[0]]
    bad = np.array([[-1, 0, 1, 2], [0, 1, 2, 99], [0, 0, 1, 2], [0, 1, 2, 3]], np.int32)
    selected, graph, retained = fixture(points, np.vstack([bad, expected]))
    np.testing.assert_array_equal(canonical(output(selected, graph)), canonical(expected))
    wp.copy(
        selected.source,
        wp.array(
            np.vstack([bad, bad])[: len(selected.source)], dtype=int, device=selected.points.device
        ),
    )
    assert len(output(selected, graph)) == 0


def test_invalid_count_rejects_batch_and_restored_prefix_replays_safely():
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
    selected, graph, retained = fixture(points, [[0, 1, 2, 3]])
    assert len(output(selected, graph)) == 1
    for count in [-1, 2]:
        selected.count.fill_(count)
        wp.capture_launch(graph)
        assert selected.error.numpy()[0] == 4
        assert selected.tetrahedra_count.numpy()[0] == 0
    selected.count.zero_()
    assert len(output(selected, graph)) == 0
    selected.count.fill_(1)
    assert len(output(selected, graph)) == 1
