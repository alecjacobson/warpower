from itertools import combinations, product

import numpy as np
import pytest
import warp as wp

from tests.test_facet_completion import canonical
from warpower import DelaunayWorkspace
from warpower._canonical import canonicalize
from warpower._facet_seeds import CertifiedSeeds
from warpower._site_bvh import SiteHierarchy


def fixture(points, cells, *, prepare_geometry=False, block_dim=256):
    p = wp.array(np.asarray(points, np.float32), dtype=wp.vec3, device="cuda:0")
    t = wp.array(np.asarray(cells, np.int32), dtype=int, device=p.device)
    count = wp.full(1, len(t), dtype=int, device=p.device)
    hierarchy = SiteHierarchy(p)
    weights = wp.zeros(len(p), dtype=float, device=p.device)
    owners, retained = canonicalize(p, weights)
    hierarchy.tree.owners = owners
    selected = CertifiedSeeds(
        p, t, count, hierarchy.tree, prepare_geometry=prepare_geometry, block_dim=block_dim
    )
    with wp.ScopedCapture() as capture:
        selected.run()
    return selected, capture.graph, (hierarchy, owners, retained, weights)


def output(selected, graph):
    wp.capture_launch(graph)
    assert selected.error.numpy()[0] == 0
    return selected.tetrahedra.numpy()[: selected.tetrahedra_count.numpy()[0]]


@pytest.mark.parametrize("shape", ["cube", "octahedron"])
@pytest.mark.parametrize("scale", [2.0**-60, 1.0, 2.0**60])
@pytest.mark.parametrize("prepare_geometry", [False, True])
def test_all_candidate_combinations_reduce_to_exact_symbolic_lower_facets(
    shape, scale, prepare_geometry
):
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
    selected, graph, retained = fixture(points, candidates, prepare_geometry=prepare_geometry)
    actual = output(selected, graph)
    np.testing.assert_array_equal(canonical(actual), canonical(expected))


@pytest.mark.parametrize("prepare_geometry", [False, True])
def test_malformed_cells_and_nonempty_sphere_are_discarded_without_reading_indices(
    prepare_geometry,
):
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.25, 0.25, 0.25]], np.float32)
    reference = DelaunayWorkspace(points).run()
    assert reference.error.numpy()[0] == 0
    expected = reference.tetrahedra.numpy()[: reference.tetrahedra_count.numpy()[0]]
    bad = np.array([[-1, 0, 1, 2], [0, 1, 2, 99], [0, 0, 1, 2], [0, 1, 2, 3]], np.int32)
    selected, graph, retained = fixture(
        points, np.vstack([bad, expected]), prepare_geometry=prepare_geometry
    )
    np.testing.assert_array_equal(canonical(output(selected, graph)), canonical(expected))
    wp.copy(
        selected.source,
        wp.array(
            np.vstack([bad, bad])[: len(selected.source)], dtype=int, device=selected.points.device
        ),
    )
    assert len(output(selected, graph)) == 0


@pytest.mark.parametrize("prepare_geometry", [False, True])
def test_invalid_count_rejects_batch_and_restored_prefix_replays_safely(prepare_geometry):
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
    selected, graph, retained = fixture(points, [[0, 1, 2, 3]], prepare_geometry=prepare_geometry)
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


@wp.kernel
def change_defining_tet(points: wp.array[wp.vec3], phase: wp.array[int]):
    scale = wp.where(phase[0] % 2 == 0, 1.0, 4.0)
    points[1] = wp.vec3(scale, 0.0, 0.0)
    points[2] = wp.vec3(0.0, scale, 0.0)
    points[3] = wp.vec3(0.0, 0.0, scale)


@wp.kernel
def consume_count(
    count: wp.array[int], error: wp.array[int], phase: wp.array[int], result: wp.array2d[int]
):
    result[phase[0], 0] = count[0]
    result[phase[0], 1] = error[0]
    phase[0] += 1


@pytest.mark.parametrize("block_dim", [32, 256])
def test_prepared_geometry_refreshes_inside_captured_producer_consumer_loop(block_dim):
    # The fixed query is outside the unit tetrahedron's sphere and strictly
    # inside the scaled tetrahedron's sphere. Reusing either old cofactor/box
    # would admit a nonempty sphere or discard a valid cell on alternate phases.
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [3, 3, 3]], np.float32)
    selected, _, retained = fixture(
        points, [[0, 1, 2, 3]], prepare_geometry=True, block_dim=block_dim
    )
    phase = wp.zeros(1, dtype=int, device=selected.device)
    result = wp.empty((4, 2), dtype=int, device=selected.device)
    wp.load_module(module=__name__, device=selected.device)
    with wp.ScopedCapture() as capture:
        phase.zero_()
        for _ in range(4):
            wp.launch(
                change_defining_tet, 1, inputs=[selected.points, phase], device=selected.device
            )
            selected.run()
            wp.launch(
                consume_count,
                1,
                inputs=[selected.tetrahedra_count, selected.error, phase, result],
                device=selected.device,
            )
    for _ in range(2):
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(result.numpy(), [[1, 0], [0, 0], [1, 0], [0, 0]])
