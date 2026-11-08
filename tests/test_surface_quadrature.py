"""Parallel quadrature preserves arithmetic, inactive capacity and captured errors."""

import numpy as np
import pytest
import warp as wp

from examples.vtm_inputs import counterbored_plate, surface
from tests import surface_quadrature_reference as reference
from tests.test_domain import cube, torus
from warpower import SurfaceDomain, _extraction


@pytest.fixture(autouse=True)
def restore_compilation_options():
    options = [
        (module, dict(wp.get_module_options(module=module))) for module in (reference, _extraction)
    ]
    try:
        yield
    finally:
        for module, original in options:
            wp.set_module_options(original, module=module)


def evaluate(domain, points, faces, live, error=0):
    points = wp.array(points, dtype=wp.vec3)
    faces = wp.array(faces, dtype=int)
    count = wp.array([live], dtype=int)
    result = []
    for method, factor in [(reference.face_quadrature, 1), (_extraction.surface_error, 3)]:
        integral = wp.zeros(1, dtype=wp.float64)
        maximum = wp.zeros(1, dtype=wp.float64)
        state = wp.array([error], dtype=int)
        wp.launch(
            method,
            factor * len(faces),
            inputs=[domain.tree, points, faces, count, integral, maximum, state],
            block_dim=32,
        )
        result.append((integral.numpy()[0], maximum.numpy()[0], state.numpy()[0]))
    return result


@pytest.mark.parametrize("case", ["cube", "torus", "plate", "thin-hand", "cavity", "disconnected"])
@pytest.mark.parametrize("exponent", [-60, 0, 60])
@pytest.mark.parametrize("fuse", [False, True])
def test_arithmetic(case, exponent, fuse):
    for module in [reference, _extraction]:
        wp.set_module_options({"fuse_fp": fuse}, module=module)
    p, f = (
        cube()
        if case == "cube"
        else torus(16, 8)
        if case == "torus"
        else counterbored_plate(16)
        if case == "plate"
        else surface(case)
    )
    scale = np.float32(2.0**exponent)
    p = p * scale
    d = SurfaceDomain(p, f, surface_subdivisions=1, crease_subdivisions=1)
    assert d.error.numpy()[0] == 0
    q = np.concatenate(
        [p, p * np.float32(1.03) + np.array([0.02, -0.03, 0.01], np.float32) * scale]
    )
    faces = np.concatenate([f, f + len(p), np.full((37, 3), -7, np.int32)])
    a, b = evaluate(d, q, faces, 2 * len(f))
    assert a[2] == b[2] == 0
    np.testing.assert_allclose(a[0], b[0], rtol=2e-12, atol=0)
    assert a[1].tobytes() == b[1].tobytes()


@pytest.mark.parametrize("case", ["zero", "prior-error", "empty-tree", "nonfinite-query"])
def test_guards_and_failed_queries(case):
    for module in [reference, _extraction]:
        wp.set_module_options({"fuse_fp": False}, module=module)
    p, f = cube()
    d = SurfaceDomain(p, f, surface_subdivisions=1, crease_subdivisions=1)
    count = len(f)
    initial = 0
    if case == "zero":
        count = 0
        f = np.full_like(f, -7)
    elif case == "prior-error":
        initial = 8
        f = np.full_like(f, -7)
    elif case == "empty-tree":
        d.tree.n = 0
    else:
        p[:] = np.nan
    for row in evaluate(d, p, f, count, initial):
        assert row[2] == (initial if case in ["zero", "prior-error"] else 16)
        assert row[0] == row[1] == 0


@wp.kernel
def consume(error: wp.array[int], count: wp.array[int], rows: wp.array[int], step: int):
    rows[step] = error[0]
    if step == 2:
        rows[3] = count[0]


def test_captured_counts_refit_and_recovery_on_another_stream():
    from tests.test_extraction import grid_mesh

    for module in [reference, _extraction]:
        wp.set_module_options({"fuse_fp": False}, module=module)
    p, f = cube()
    d = SurfaceDomain(p, f, surface_subdivisions=1, crease_subdivisions=1)
    q, t = grid_mesh(*([np.linspace(-1, 1, 3)] * 3))
    w = _extraction.ExtractionWorkspace(d, q, t)
    stream = wp.Stream(d.device)
    observed = wp.zeros(4, dtype=int)
    wp.load_module(module=__name__, device=d.device)
    wp.synchronize_device(d.device)
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
