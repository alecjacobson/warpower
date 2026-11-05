import numpy as np

from benchmarks.datasets import generate
from tests.oracle import reference_cells


def test_two_sites_and_dominated_cell():
    p = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    rows, _ = reference_cells(p)
    assert rows == [{1}, {0}]
    rows, vertices = reference_cells(p, [0, 100])
    assert rows == [set(), set()]
    assert len(vertices[0]) == 0 and len(vertices[1]) == 8


def test_paper_distributions_reproducible():
    for distribution in ("uniform", "gradient", "cluster5", "cluster10"):
        a, w = generate(100, distribution, weighted=True)
        b, v = generate(100, distribution, weighted=True)
        assert np.array_equal(a, b) and np.array_equal(w, v)
        assert a.dtype == w.dtype == np.float32
        assert np.isfinite(a).all() and np.isfinite(w).all()
