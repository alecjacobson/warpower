"""Per-query arithmetic, padded counts and failures for private parallel quadrature."""
import numpy as np
import pytest
import warp as wp
from examples.vtm_inputs import counterbored_plate, surface
from tests.test_domain import cube, torus
from warpower import SurfaceDomain, _extraction
from warpower._surface_bvh import Closest
import surface_sample_parallel_v1 as kernels


def evaluate(domain, points, faces, live, error=0, observed=True):
    points = wp.array(points, dtype=wp.vec3)
    faces = wp.array(faces, dtype=int)
    count = wp.array([live], dtype=int)
    result = []
    methods = [(_extraction.surface_error, 1), (kernels.surface_error_samples, 3)]
    if observed:
        methods += [(kernels.reference_observed, 1), (kernels.samples_observed, 3)]
    for method, factor in methods:
        integral = wp.zeros(1, dtype=wp.float64)
        maximum = wp.zeros(1, dtype=wp.float64)
        state = wp.array([error], dtype=int)
        args = [domain.tree, points, faces, count, integral, maximum, state]
        hits = wp.zeros(3 * len(faces), dtype=Closest)
        weights = wp.full(3 * len(faces), -77, dtype=wp.float64)
        if method in (kernels.reference_observed, kernels.samples_observed):
            args += [hits, weights]
        wp.launch(method, factor * len(faces), inputs=args, block_dim=32)
        result.append((integral.numpy()[0], maximum.numpy()[0], state.numpy()[0], hits.numpy(), weights.numpy()))
    return result


@pytest.mark.parametrize('case', ['cube', 'torus', 'plate', 'thin-hand', 'cavity', 'disconnected'])
@pytest.mark.parametrize('exponent', [-60, 0, 60])
@pytest.mark.parametrize('fuse', [False, True])
def test_arithmetic(case, exponent, fuse):
    wp.set_module_options({'fuse_fp': fuse}, module=kernels)
    wp.set_module_options({'fuse_fp': fuse}, module=_extraction)
    p, f = cube() if case == 'cube' else torus(16, 8) if case == 'torus' else counterbored_plate(16) if case == 'plate' else surface(case)
    scale = np.float32(2.0**exponent)
    p = p * scale
    d = SurfaceDomain(p, f, surface_subdivisions=1, crease_subdivisions=1)
    assert d.error.numpy()[0] == 0
    q = np.concatenate([p, p * np.float32(1.03) + np.array([.02, -.03, .01], np.float32) * scale])
    query_faces = np.concatenate([f, f + len(p), np.full((37,3), -7, np.int32)])
    rows = evaluate(d, q, query_faces, 2 * len(f))
    for row in rows:
        assert row[2] == 0
        np.testing.assert_allclose(row[:2], rows[0][:2], rtol=2e-12, atol=0)
    for key in ['face', 'distance2', 'point']:
        np.testing.assert_array_equal(rows[2][3][key], rows[3][3][key])
    np.testing.assert_array_equal(rows[2][4], rows[3][4])
    assert np.all(rows[3][4][6*len(f):] == -77)


@pytest.mark.parametrize('case', ['zero', 'prior-error', 'empty-tree', 'nonfinite-query'])
def test_guards_and_failed_queries(case):
    wp.set_module_options({'fuse_fp': False}, module=kernels)
    wp.set_module_options({'fuse_fp': False}, module=_extraction)
    p, f = cube()
    d = SurfaceDomain(p, f, surface_subdivisions=1, crease_subdivisions=1)
    count = len(f)
    initial = 0
    if case == 'zero':
        count = 0
        f = np.full_like(f, -7)
    elif case == 'prior-error':
        initial = 8
        f = np.full_like(f, -7)
    elif case == 'empty-tree':
        d.tree.n = 0
    else:
        p[:] = np.nan
    rows = evaluate(d, p, f, count, initial, observed=False)
    for row in rows:
        assert row[2] == (initial if case in ['zero','prior-error'] else 16)
        assert row[0] == row[1] == 0


def test_captured_changed_counts_refit_and_recovery():
    # Public extraction's captured producer/count/error/refit/consumer contracts
    # are also exercised by the unchanged extraction test module in the runner.
    from tests.test_extraction import grid_mesh
    wp.set_module_options({'fuse_fp': False}, module=kernels)
    wp.set_module_options({'fuse_fp': False}, module=_extraction)
    p, f = cube()
    d = SurfaceDomain(p, f, surface_subdivisions=1, crease_subdivisions=1)
    q, t = grid_mesh(*([np.linspace(-1, 1, 3)] * 3))
    w = _extraction.ExtractionWorkspace(d, q, t)
    stream = wp.Stream(d.device)
    observed = wp.zeros(4, dtype=int)
    wp.load_module(module=__name__, device=d.device)
    with wp.ScopedStream(stream):
        with wp.ScopedCapture() as capture:
            w.input_count.fill_(0)
            w.run()
            wp.launch(consume, 1, inputs=[w.error, w.tetrahedra_count, observed, 0])
            w.input_count.fill_(len(t))
            w.source_error.fill_(8)
            w.run()
            wp.launch(consume, 1, inputs=[w.error, w.tetrahedra_count, observed, 1])
            w.source_error.zero_()
            d.refit()
            w.run()
            wp.launch(consume, 1, inputs=[w.error, w.tetrahedra_count, observed, 2])
        wp.capture_launch(capture.graph)
    wp.synchronize_stream(stream)
    assert observed.numpy().tolist() == [0, 2, 0, len(t)]


@wp.kernel
def consume(error: wp.array[int], count: wp.array[int], rows: wp.array[int], step: int):
    rows[step] = error[0]
    if step == 2:
        rows[3] = count[0]
