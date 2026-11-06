from itertools import product

import numpy as np
import pytest
import warp as wp
from scipy.spatial import Delaunay

from tests.test_local_certificate import positive
from warpower import DelaunayWorkspace
from warpower._canonical import canonicalize
from warpower._facet_completion import FacetCompletion
from warpower._site_bvh import SiteHierarchy


def canonical(t):
    t = np.sort(t, axis=1)
    return t[np.lexsort(t[:, ::-1].T)]


def fixture(points, seeds, capacity=None, max_rounds=64):
    points = np.asarray(points, np.float32)
    seeds = np.asarray(seeds, np.int32).reshape((-1, 4))
    p = wp.array(points, dtype=wp.vec3, device="cuda:0")
    source = wp.empty((max(1, len(seeds)), 4), dtype=int, device=p.device)
    if len(seeds):
        wp.copy(source, wp.array(seeds, dtype=int, device=p.device))
    count = wp.full(1, len(seeds), dtype=int, device=p.device)
    hierarchy = SiteHierarchy(p)
    weights = wp.zeros(len(p), dtype=float, device=p.device)
    owners, buffers = canonicalize(p, weights)
    hierarchy.tree.owners = owners
    workspace = FacetCompletion(
        p,
        source,
        count,
        hierarchy.tree,
        capacity=max(len(seeds), 16 * len(points)) if capacity is None else capacity,
        max_rounds=max_rounds,
    )
    return workspace, hierarchy, (owners, buffers, weights)


def execute(workspace):
    with wp.ScopedCapture() as capture:
        workspace.run()
    wp.capture_launch(capture.graph)
    return int(workspace.error.numpy()[0]), workspace.tetrahedra.numpy()[
        : workspace.tetrahedra_count.numpy()[0]
    ]


@pytest.mark.parametrize("subset", ["one", "scattered", "hole", "full"])
def test_recovers_full_random_hull_from_certified_fragments(subset):
    points = np.random.default_rng(91).normal(size=(32, 3)).astype(np.float32)
    full = positive(points, Delaunay(points).simplices)
    seeds = {"one": full[:1], "scattered": full[::7], "hole": full[1:], "full": full}[subset]
    workspace, hierarchy, retained = fixture(points, seeds)
    error, cells = execute(workspace)
    assert error == 0, (error, workspace.certificate_error.numpy())
    np.testing.assert_array_equal(canonical(cells), canonical(full))
    assert workspace.rounds.numpy()[0] == 0 if subset == "full" else workspace.rounds.numpy()[0] > 0


@pytest.mark.parametrize("shape", ["cube", "octahedron", "thin"])
def test_exact_ties_and_thin_cells_use_same_symbolic_triangulation(shape):
    points = np.array(list(product([-1.0, 1.0], repeat=3)), np.float32)
    if shape == "octahedron":
        points = np.vstack([np.eye(3), -np.eye(3)]).astype(np.float32)
    if shape == "thin":
        points[:, 2] *= np.float32(2.0**-30)
    reference = DelaunayWorkspace(points).run()
    assert reference.error.numpy()[0] == 0
    full = reference.tetrahedra.numpy()[: reference.tetrahedra_count.numpy()[0]]
    # Duplicated sites must retain the same lowest-ID symbolic ownership.
    points = np.vstack([points, points[:3]])
    workspace, hierarchy, retained = fixture(points, full[:1])
    error, cells = execute(workspace)
    assert error == 0, (error, workspace.certificate_error.numpy())
    np.testing.assert_array_equal(canonical(cells), canonical(full))


@pytest.mark.parametrize("bad", ["empty", "index", "negative", "duplicate", "sphere", "count"])
def test_uncertified_seed_is_rejected_with_zero_output_count(bad):
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.2, 0.2, 0.2]], np.float32)
    full = positive(points, Delaunay(points).simplices)
    seeds = full[:1].copy()
    if bad == "empty":
        seeds = seeds[:0]
    elif bad == "index":
        seeds[0, 0] = len(points)
    elif bad == "negative":
        seeds[0, [0, 1]] = seeds[0, [1, 0]]
    elif bad == "duplicate":
        seeds = np.vstack([seeds, seeds])
    elif bad == "sphere":
        seeds = np.array([[0, 1, 2, 3]], np.int32)
    workspace, hierarchy, retained = fixture(points, seeds)
    if bad == "count":
        workspace.seed_count.fill_(2)
    error, cells = execute(workspace)
    assert error == 1
    assert len(cells) == 0
    assert workspace.rounds.numpy()[0] == 0


@pytest.mark.parametrize("limit", ["rounds", "capacity"])
def test_bounded_failures_do_not_publish_partial_hulls(limit):
    points = np.random.default_rng(92).normal(size=(30, 3)).astype(np.float32)
    full = positive(points, Delaunay(points).simplices)
    workspace, hierarchy, retained = fixture(
        points,
        full[:1],
        capacity=1 if limit == "capacity" else None,
        max_rounds=0 if limit == "rounds" else 64,
    )
    error, cells = execute(workspace)
    assert error == (16 if limit == "rounds" else 4)
    assert len(cells) == 0


def test_resident_graph_changes_points_seeds_invalid_input_and_recovers():
    from benchmarks.facet_completion import audit

    audit()
