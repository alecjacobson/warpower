import numpy as np
import pytest
import warp as wp

import warpower
from tests.oracle import csr_sets, reference_cells


@pytest.mark.parametrize("weighted", [False, True])
def test_coincident_sites_have_global_deterministic_owner(weighted):
    rng = np.random.default_rng(8)
    unique = rng.normal(size=(24, 3)).astype(np.float32)
    p = np.repeat(unique, 4, axis=0)
    rng.shuffle(p)
    w = (
        rng.integers(-2, 3, len(p)).astype(np.float32) * 0.01
        if weighted
        else np.zeros(len(p), np.float32)
    )
    owners = []
    for point in unique:
        ids = np.flatnonzero(np.all(p == point, axis=1))
        owners.append(int(ids[np.argmax(w[ids])]))
    expected = [set() for _ in p]
    reduced, _ = reference_cells(p[owners], w[owners])
    for i, row in zip(owners, reduced):
        expected[i] = {owners[j] for j in row}
    result = warpower.power_diagram(p, w) if weighted else warpower.voronoi_diagram(p)
    assert csr_sets(result.adjacency.numpy(), result.offsets.numpy()) == expected
    status = result.status.numpy()
    assert np.all(status[owners] == 0)
    assert np.all(status[np.setdiff1d(np.arange(len(p)), owners)] == 3)


def test_all_coincident_and_signed_zero():
    p = np.zeros((1024, 3), dtype=np.float32)
    p[::2] = -0.0
    w = np.zeros(len(p), dtype=np.float32)
    w[137] = 1.0
    result = warpower.power_diagram(p, w)
    assert len(result.adjacency) == 0
    status = result.status.numpy()
    assert status[137] == 0
    assert np.count_nonzero(status == 3) == len(p) - 1


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_values_are_rejected(value):
    p = np.zeros((3, 3), dtype=np.float32)
    p[1, 2] = value
    with pytest.raises(ValueError, match="finite"):
        warpower.voronoi_diagram(p)
    p[1, 2] = 0
    with pytest.raises(ValueError, match="finite"):
        warpower.power_diagram(p, np.array([0, value, 0], dtype=np.float32))


@pytest.mark.parametrize("shape", [(1, 4), (2, 2), (4, 1, 1)])
def test_wrong_weight_shape_is_rejected(shape):
    with pytest.raises(ValueError, match="shape"):
        warpower.power_diagram(np.zeros((4, 3), np.float32), np.zeros(shape, np.float32))


@pytest.mark.parametrize("offsets", [[1, 1, 1, 1], [0, 1, 0, 1], [0, 0, 0, 2], [0, -1, 0, 1]])
def test_bad_csr_offsets_are_rejected(offsets):
    with pytest.raises(ValueError, match="CSR"):
        warpower.voronoi_diagram(
            np.zeros((3, 3), np.float32),
            initial_guesses=np.array([1], dtype=np.int32),
            initial_guesses_offsets=np.array(offsets, dtype=np.int32),
        )


def test_hint_indices_and_empty_input_are_validated():
    with pytest.raises(ValueError, match="indices"):
        warpower.voronoi_diagram(
            np.zeros((3, 3), np.float32),
            initial_guesses=np.array([3], dtype=np.int32),
            initial_guesses_offsets=np.array([0, 0, 0, 1], dtype=np.int32),
        )
    with pytest.raises(ValueError, match="together"):
        warpower.voronoi_diagram(
            np.empty((0, 3), np.float32), initial_guesses=np.empty(0, np.int32)
        )
    with pytest.raises(ValueError, match="shape"):
        warpower.power_diagram(np.empty((0, 3), np.float32), np.empty((0, 2), np.float32))


def test_warm_start_with_repeated_and_skipped_hints():
    p = np.random.default_rng(13).normal(size=(40, 3)).astype(np.float32)
    first = warpower.voronoi_diagram(p)
    rows = csr_sets(first.adjacency.numpy(), first.offsets.numpy())
    guesses = [[-1] + sorted(row) * 2 for row in rows]
    offsets = np.cumsum([0] + list(map(len, guesses)), dtype=np.int32)
    hints = np.array([j for row in guesses for j in row], dtype=np.int32)
    second = warpower.voronoi_diagram(p, initial_guesses=hints, initial_guesses_offsets=offsets)
    assert csr_sets(second.adjacency.numpy(), second.offsets.numpy()) == rows


def test_numpy_column_and_noncontiguous_warp_weights():
    p = np.random.default_rng(4).normal(size=(32, 3)).astype(np.float32)
    w = np.zeros((32, 1), np.float32)
    a = warpower.power_diagram(p, w)
    packed = wp.zeros(64, dtype=float, device="cuda:0")
    b = warpower.power_diagram(wp.array(p, dtype=wp.vec3, device="cuda:0"), packed[::2])
    assert csr_sets(a.adjacency.numpy(), a.offsets.numpy()) == csr_sets(
        b.adjacency.numpy(), b.offsets.numpy()
    )


def test_torch_nondefault_stream_and_noncontiguous_input():
    torch = pytest.importorskip("torch")
    stream = torch.cuda.Stream()
    cpu = np.random.default_rng(12).normal(size=(24, 3)).astype(np.float32)
    with torch.cuda.stream(stream):
        p = torch.from_numpy(cpu.T.copy()).cuda().T
        w = torch.zeros((24, 1), device="cuda")
        result = warpower.power_diagram(p, w)
        assert result.adjacency.is_cuda and result.adjacency.dtype == torch.int32
        assert result.offsets.dtype == torch.int32 and result.status.dtype == torch.int32
    stream.synchronize()
    assert (
        csr_sets(result.adjacency.cpu().numpy(), result.offsets.cpu().numpy())
        == reference_cells(cpu)[0]
    )
