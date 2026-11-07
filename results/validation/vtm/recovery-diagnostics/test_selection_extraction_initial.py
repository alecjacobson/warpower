"""Private selection must preserve the full extractor's validity and geometry."""

import numpy as np
import pytest
import warp as wp

import warpower
from tests.test_current_selection import produce_controls, produce_points, recovered_selection
from tests.test_domain import cube, torus
from warpower import _extraction


@pytest.mark.parametrize("shape", ["cube", "torus"])
@pytest.mark.parametrize("scale", [1e-30, 1.0, 1e30])
@pytest.mark.parametrize(
    "mode", ["ordinary", "nan_sample", "initial_error", "missing_witness", "all_missing"]
)
def test_status_matches_full_surface_queries(shape, scale, mode):
    p, t = cube() if shape == "cube" else torus(12, 6)
    p = (p * scale).astype(np.float32)
    domain = warpower.SurfaceDomain(p, t)
    assert domain.error.numpy()[0] == 0
    if mode in ["missing_witness", "all_missing"]:
        # Deliberately poison leaf arithmetic after building finite valid boxes:
        # one bad witness must fall back; all missing hits must preserve error16.
        first = int(domain.tree.ids.numpy()[0])
        changed = p.copy()
        changed[t[first]] = np.nan
        if mode == "all_missing":
            changed[:] = np.nan
        wp.copy(domain.points, wp.array(changed, dtype=wp.vec3, device=domain.device))
    query = (np.random.default_rng(192).uniform(-4, 4, (96, 3)) * scale).astype(np.float32)
    if mode == "nan_sample":
        query[0] = np.nan
    points = wp.array(query, dtype=wp.vec3, device=domain.device)
    indices = np.full((40, 3), -7, np.int32)
    indices[:32] = np.arange(96, dtype=np.int32).reshape(-1, 3)
    faces = wp.array(indices, device=domain.device)
    count = wp.full(1, 32, dtype=int, device=domain.device)
    errors = [
        wp.full(1, 4 if mode == "initial_error" else 0, dtype=int, device=domain.device)
        for _ in range(2)
    ]
    integral = wp.zeros(1, dtype=wp.float64, device=domain.device)
    maximum = wp.zeros_like(integral)
    common = [domain.tree, points, faces, count]
    wp.launch(
        _extraction.surface_error,
        len(faces),
        inputs=common + [integral, maximum, errors[0]],
        device=domain.device,
    )
    wp.launch(
        _extraction.selection_surface_status,
        len(faces),
        inputs=common + [errors[1]],
        device=domain.device,
    )
    np.testing.assert_array_equal(errors[0].numpy(), errors[1].numpy())
    if mode == "ordinary":
        assert errors[0].numpy()[0] == 0
    if mode in ["nan_sample", "all_missing"]:
        assert errors[0].numpy()[0] == 16


@pytest.mark.parametrize("value", range(6))
def test_private_status_matches_full_extraction_after_device_changes(value, monkeypatch):
    c, s = recovered_selection()
    e = s.recovery_extraction
    assert isinstance(e, _extraction._SelectionExtractionWorkspace)
    full = warpower.ExtractionWorkspace(
        c.domain,
        c.points,
        c.full_tetrahedra,
        c.full_tetrahedra_count,
        source_error=c.mesh.error,
        containment_recovery=True,
    )
    base, surface = wp.clone(c.points), wp.clone(c.domain.points)
    phase = wp.full(1, value, dtype=int, device=c.device)
    wp.load_module(module="tests.test_current_selection", device=c.device)
    pointers = [c.points.ptr, e.selected.ptr, e.tetrahedra.ptr, e.error.ptr]

    def forbidden(*args, **kwargs):
        raise AssertionError("Host allocation, readback or synchronization in capture")

    with monkeypatch.context() as patch:
        for name in [
            "empty",
            "zeros",
            "ones",
            "empty_like",
            "clone",
            "synchronize",
            "synchronize_device",
        ]:
            patch.setattr(wp, name, forbidden)
        patch.setattr(wp.array, "numpy", forbidden)
        with wp.ScopedCapture() as capture:
            wp.launch(produce_points, c.n, inputs=[base, c.points, phase, True], device=c.device)
            wp.launch(
                produce_points,
                len(surface),
                inputs=[surface, c.domain.points, phase, False],
                device=c.device,
            )
            wp.launch(
                produce_controls,
                1,
                inputs=[phase, c.material_recovery, c.mesh.error],
                device=c.device,
            )
            c.material_recovery.fill_(1)
            c.domain.refit()
            e._surface_integral.fill_(77)
            e.surface_max_sample_distance.fill_(77)
            e.surface_rms_distance.fill_(77)
            s.evaluate()
            full.run()
        wp.capture_launch(capture.graph)
    fields = ["error", "valid", "tetrahedra_count", "boundary_count", "selected", "_input_quality"]
    for name in fields:
        if name == "_input_quality" and e.error.numpy()[0] != 0:
            continue
        np.testing.assert_array_equal(getattr(e, name).numpy(), getattr(full, name).numpy())
    assert e._surface_integral.numpy()[0] == 0
    assert e.surface_max_sample_distance.numpy()[0] == e.surface_rms_distance.numpy()[0] == 0
    assert [c.points.ptr, e.selected.ptr, e.tetrahedra.ptr, e.error.ptr] == pointers
    if value in [2, 4]:
        assert e.error.numpy()[0] != 0 and e.valid.numpy()[0] == 0
    else:
        assert e.error.numpy()[0] == 0 and e.valid.numpy()[0] == 1


def test_public_extractor_retains_full_diagnostics_with_private_selection_alongside():
    domain = warpower.SurfaceDomain(*cube())
    points = wp.array(
        [[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]], dtype=wp.vec3, device=domain.device
    )
    cells = wp.array([[0, 1, 2, 3]], dtype=int, device=domain.device)
    private = _extraction._SelectionExtractionWorkspace(domain, points, cells)
    public = warpower.ExtractionWorkspace(domain, points, cells)
    with wp.ScopedCapture() as capture:
        private.run()
        public.run()
    for _ in range(2):
        wp.capture_launch(capture.graph)
        assert private.valid.numpy()[0] == public.valid.numpy()[0] == 1
        for name in [
            "selected",
            "tetrahedra",
            "radius_ratio",
            "volume",
            "domain_volume",
            "volume_difference",
        ]:
            np.testing.assert_array_equal(
                getattr(private, name).numpy(), getattr(public, name).numpy()
            )
        assert (
            private.surface_rms_distance.numpy()[0]
            == private.surface_max_sample_distance.numpy()[0]
            == 0
        )
        assert public.surface_rms_distance.numpy()[0] > 0.8
        assert public.surface_max_sample_distance.numpy()[0] > 0.8
