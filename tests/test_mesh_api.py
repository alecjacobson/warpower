import subprocess
import sys

import numpy as np
import pytest
import warp as wp

import warpower
from tests.test_tetrahedra import check_mesh, reference, tet_set


@pytest.mark.parametrize("weighted", [False, True])
def test_standalone_mesh_skips_diagram_and_matches_reference(weighted, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Standalone tetrahedra computed bounded adjacency")

    monkeypatch.setattr(warpower._api, "_build", forbidden)
    rng = np.random.default_rng(93)
    points = rng.normal(size=(70, 3)).astype(np.float32)
    weights = rng.normal(0, 0.1, len(points)).astype(np.float32) if weighted else None
    tets = (
        warpower.regular_tetrahedra(points, weights, chunk_size=13)
        if weighted
        else warpower.delaunay_tetrahedra(points, chunk_size=13)
    ).numpy()
    check_mesh(points, tets)
    assert tet_set(tets) == reference(points, weights)


def test_default_import_remains_lazy():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, warpower; "
            "assert not any(n in sys.modules for n in "
            "('warpower._mesh_api', 'warpower._tetrahedra', 'warpower._graph'))",
        ],
        check=True,
    )


@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_small_standalone_inputs(n):
    points = np.random.default_rng(1).normal(size=(n, 3)).astype(np.float32)
    assert warpower.delaunay_tetrahedra(points).shape == (0, 4)


def test_standalone_validation_and_ambiguity():
    points = np.random.default_rng(1).normal(size=(20, 3)).astype(np.float32)
    for kwargs in ({"leaf_size": 0}, {"chunk_size": 0}):
        with pytest.raises(ValueError):
            warpower.delaunay_tetrahedra(points, **kwargs)
    with pytest.raises(TypeError, match="requires weights"):
        warpower.regular_tetrahedra(points, None)
    with pytest.raises(ValueError, match="CUDA"):
        warpower.delaunay_tetrahedra(wp.array(points, dtype=wp.vec3, device="cpu"))
    points[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        warpower.delaunay_tetrahedra(points)
    cube = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], np.float32)
    with pytest.raises(ValueError, match="partial mesh"):
        warpower.delaunay_tetrahedra(cube)


def test_standalone_torch_current_stream():
    torch = pytest.importorskip("torch")
    points = np.random.default_rng(12).normal(size=(24, 3)).astype(np.float32)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        tensor = torch.from_numpy(points.T.copy()).cuda().T
        tets = warpower.delaunay_tetrahedra(tensor)
        assert tets.is_cuda and tets.dtype == torch.int32
        copied = tets.clone()
    stream.synchronize()
    assert tet_set(copied.cpu().numpy()) == reference(points)
