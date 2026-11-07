from fractions import Fraction

import numpy as np
import pytest
import warp as wp
from scipy.spatial import ConvexHull, Delaunay

import warpower


def tet_set(tets):
    return {tuple(sorted(map(int, t))) for t in tets}


def reference(points, weights=None):
    p = points.astype(np.float64)
    if weights is None:
        return tet_set(Delaunay(p).simplices)
    lifted = np.column_stack((p, np.sum(p * p, axis=1) - weights.astype(np.float64)))
    hull = ConvexHull(lifted)
    return tet_set(hull.simplices[hull.equations[:, 3] < 0])


def check_mesh(points, tets):
    assert tets.dtype == np.int32 and tets.ndim == 2 and tets.shape[1] == 4
    assert len(tet_set(tets)) == len(tets)
    p = points.astype(np.float64)[tets]
    signed_six_volumes = np.einsum(
        "ij,ij->i", p[:, 1] - p[:, 0], np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0])
    )
    # The independent checker must not repeat extraction's cancellation bug.
    # Use exact rational differences/products in ambiguous cases, never clamp
    # or ignore a zero/negative floating-point estimate.
    a, b, c = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]
    magnitude = sum(
        np.abs(a[:, j])
        * (
            np.abs(b[:, (j + 1) % 3] * c[:, (j + 2) % 3])
            + np.abs(b[:, (j + 2) % 3] * c[:, (j + 1) % 3])
        )
        for j in range(3)
    )
    for i in np.flatnonzero(np.abs(signed_six_volumes) <= 1e-12 * magnitude):
        row = [[Fraction(float(x)) for x in vertex] for vertex in p[i]]
        a0, b0, c0 = [[x - y for x, y in zip(vertex, row[0])] for vertex in row[1:]]
        signed_six_volumes[i] = float(
            sum(
                a0[j] * (b0[(j + 1) % 3] * c0[(j + 2) % 3] - b0[(j + 2) % 3] * c0[(j + 1) % 3])
                for j in range(3)
            )
        )
    assert np.all(signed_six_volumes > 0)
    assert np.isclose(signed_six_volumes.sum() / 6, ConvexHull(points).volume, rtol=1e-9)
    faces = {}
    for tet in tets:
        for k in range(4):
            face = tuple(sorted(np.delete(tet, k)))
            faces[face] = faces.get(face, 0) + 1
    assert set(faces.values()) <= {1, 2}


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("seed,n", [(2, 12), (42, 32), (8, 96), (11, 300)])
def test_complete_dual_matches_independent_reference(weighted, seed, n):
    rng = np.random.default_rng(seed)
    p = rng.normal(size=(n, 3)).astype(np.float32)
    w = rng.normal(0, 0.15, n).astype(np.float32) if weighted else None
    result = (
        warpower.power_diagram(p, w, return_tetrahedra=True)
        if weighted
        else warpower.voronoi_diagram(p, return_tetrahedra=True)
    )
    tets = result.tetrahedra.numpy()
    check_mesh(p, tets)
    assert tet_set(tets) == reference(p, w)
    assert result.diagnostics["tetrahedra_count"] == len(tets)


def test_circumcenter_far_outside_default_box():
    p = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.2, 0.2, 1e-4]], np.float32)
    result = warpower.voronoi_diagram(p, return_tetrahedra=True)
    assert tet_set(result.tetrahedra.numpy()) == {(0, 1, 2, 3)}
    assert result.diagnostics["tetrahedra_expansions"] >= 4
    check_mesh(p, result.tetrahedra.numpy())


@pytest.mark.parametrize("center_weight", [1000, -1000])
def test_weighted_full_hull_and_hidden_sites(center_weight):
    p = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.2, 0.2, 0.2]], np.float32)
    w = np.array([0, 0, 0, 0, center_weight], np.float32)
    result = warpower.power_diagram(p, w, return_tetrahedra=True)
    tets = result.tetrahedra.numpy()
    assert tet_set(tets) == reference(p, w)
    check_mesh(p, tets)
    if center_weight > 0:
        # Hull cells empty in the default finite box still appear in the full dual.
        assert np.all(result.status.numpy()[:4] == 3)
        assert len(tets) == 4
    else:
        assert 4 not in tets


def test_duplicate_owner_indices_and_chunking():
    rng = np.random.default_rng(5)
    unique = rng.normal(size=(40, 3)).astype(np.float32)
    p = np.repeat(unique, 2, axis=0)
    w = np.tile(np.array([0, 0.01], np.float32), len(unique))
    result = warpower.power_diagram(p, w, return_tetrahedra=True, chunk_size=7, leaf_size=2)
    tets = result.tetrahedra.numpy()
    assert np.all(tets % 2 == 1)
    reduced = (tets - 1) // 2
    assert tet_set(reduced) == reference(unique)
    check_mesh(p, tets)


def test_degeneracy_is_rejected_instead_of_returning_a_partial_mesh():
    cube = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], np.float32)
    with pytest.raises(ValueError, match="no partial mesh|No partial mesh"):
        warpower.voronoi_diagram(cube, return_tetrahedra=True)


@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_small_inputs_have_no_volumetric_tetrahedra(n):
    p = np.random.default_rng(4).normal(size=(n, 3)).astype(np.float32)
    result = warpower.voronoi_diagram(p, return_tetrahedra=True)
    assert result.tetrahedra.shape == (0, 4)


def test_default_never_enters_optional_builder(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Optional tetrahedron work executed for a default call")

    monkeypatch.setattr(warpower._api, "_with_tetrahedra", unexpected)
    p = np.random.default_rng(9).normal(size=(24, 3)).astype(np.float32)
    for kwargs in ({}, {"return_tetrahedra": False}):
        result = warpower.voronoi_diagram(p, **kwargs)
        assert result.tetrahedra is None
        assert not any(key.startswith("tetrahedra_") for key in result.diagnostics)
    with pytest.raises(TypeError, match="boolean"):
        warpower.voronoi_diagram(p, return_tetrahedra="yes")


def test_warp_input_and_torch_current_stream():
    p = np.random.default_rng(12).normal(size=(32, 3)).astype(np.float32)
    native = warpower.voronoi_diagram(
        wp.array(p, dtype=wp.vec3, device="cuda:0"), return_tetrahedra=True
    )
    assert native.tetrahedra.device.is_cuda
    assert tet_set(native.tetrahedra.numpy()) == reference(p)
    torch = pytest.importorskip("torch")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        tensor = torch.from_numpy(p.T.copy()).cuda().T
        result = warpower.voronoi_diagram(tensor, return_tetrahedra=True)
        assert result.tetrahedra.is_cuda and result.tetrahedra.dtype == torch.int32
        copied = result.tetrahedra.clone()
    stream.synchronize()
    assert tet_set(copied.cpu().numpy()) == reference(p)
