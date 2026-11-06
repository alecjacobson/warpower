import numpy as np
import pytest
import warp as wp

from tests.test_delaunay_recovery import valid
from tests.test_tetrahedra import reference, tet_set
from warpower import DelaunayWorkspace

wp.set_module_options({"enable_backward": False})


def cloud():
    return np.random.default_rng(8701).uniform(-1, 1, (8192, 3)).astype(np.float32)


def test_large_interior_candidates_match_scipy_and_keep_original_scratch():
    p = cloud()
    w = DelaunayWorkspace(p).run()
    assert w.fast_candidates
    assert w.fast_attempted.numpy()[0] == 1
    assert w.fast_accepted.numpy()[0] == 1
    assert w.fast_error.numpy()[0] == 0
    assert 0 < w.fast_repaired_cells.numpy()[0] < len(p)
    assert w.attempts.numpy()[0] == 0
    assert tet_set(valid(w, p)) == reference(p)
    # Symbolic recovery still uses the original float64 scratch outside capture.
    assert w._base._robust[128].planes.dtype == wp.vec4d
    assert w._base._factor is not w._base._fast_factor


def test_small_distinct_prefix_disables_fast_cells_in_large_storage():
    p = cloud()
    p[128:] = p[0]
    w = DelaunayWorkspace(p).run()
    assert w.fast_candidates
    assert w.fast_attempted.numpy()[0] == 0
    assert w.fast_accepted.numpy()[0] == 0
    assert w.fast_repaired_cells.numpy()[0] == 0
    assert tet_set(valid(w, p)) == reference(p[:128])


@wp.kernel
def duplicate_first(tets: wp.array2d[int]):
    for k in range(4):
        tets[0, k] = tets[1, k]


@pytest.mark.parametrize("cache", [True, False])
@pytest.mark.parametrize("failure", ["certificate", "construction"])
def test_rejection_falls_back_to_original_and_respects_cache(monkeypatch, cache, failure):
    p = cloud()
    w = DelaunayWorkspace(p, cache_strategy=cache)
    original = w._base._record_hybrid

    def corrupt():
        original()
        if failure == "certificate":
            wp.launch(duplicate_first, 1, inputs=[w._base.tetrahedra])
        else:
            w._base.error.fill_(32)

    monkeypatch.setattr(w._base, "_record_hybrid", corrupt)
    wp.load_module(module=__name__, device=w.device)
    with wp.ScopedCapture() as cap:
        w.run()
    wp.capture_launch(cap.graph)
    cells = tet_set(valid(w, p))
    assert cells == reference(p)
    assert w.fast_attempted.numpy()[0] == 1
    assert w.fast_accepted.numpy()[0] == 0
    expected_error = 2 if failure == "certificate" else 1
    assert w.fast_error.numpy()[0] == expected_error
    assert w._fast_available.numpy()[0] == int(not cache)
    assert w.attempts.numpy()[0] == 0
    wp.capture_launch(cap.graph)
    assert tet_set(valid(w, p)) == cells
    assert w.fast_attempted.numpy()[0] == int(not cache)
    assert w.fast_error.numpy()[0] == (0 if cache else expected_error)


def test_invalid_replay_clears_counts_and_restores_fast_preference():
    p = cloud()
    w = DelaunayWorkspace(p).run()
    w._fast_available.zero_()
    bad = p.copy()
    bad[10, 1] = np.nan
    w.points.assign(bad)
    w.run()
    assert w.error.numpy()[0] == 1
    assert w.tetrahedra_count.numpy()[0] == 0
    assert w.fast_attempted.numpy()[0] == 0
    assert w._fast_available.numpy()[0] == 1
    w.points.assign(p)
    w.run()
    assert w.fast_accepted.numpy()[0] == 1
    assert tet_set(valid(w, p)) == reference(p)


@wp.kernel
def producer(
    source: wp.array[wp.vec3],
    invalid: wp.array[float],
    points: wp.array[wp.vec3],
    phase: wp.array[int],
):
    i = wp.tid()
    p = source[i]
    if phase[0] == 1 and i == 0:
        p[0] += 0.0001
    if phase[0] == 2 and i >= 128:
        p = source[0]
    if phase[0] == 3 and i == 0:
        p[0] = invalid[0]
    points[i] = p


def test_captured_producer_gate_fallback_and_consumer_with_fixed_storage(monkeypatch, tmp_path):
    from tests.test_delaunay_recovery import consume

    p = cloud()
    w = DelaunayWorkspace(p)
    source = wp.array(p, dtype=wp.vec3, device=w.device)
    invalid = wp.array([np.nan], dtype=float, device=w.device)
    phase = wp.zeros(1, dtype=int, device=w.device)
    total = wp.zeros(1, dtype=int, device=w.device)
    wp.load_module(module=__name__, device=w.device)
    wp.load_module(module="tests.test_delaunay_recovery", device=w.device)

    def forbidden(*args, **kwargs):
        raise AssertionError("Host allocation, readback or synchronization inside capture")

    with monkeypatch.context() as patch:
        for name in (
            "empty",
            "empty_like",
            "zeros",
            "ones",
            "full",
            "clone",
            "synchronize",
            "synchronize_device",
            "synchronize_stream",
        ):
            patch.setattr(wp, name, forbidden)
        patch.setattr(wp.array, "numpy", forbidden)
        with wp.ScopedCapture() as cap:
            wp.launch(producer, len(p), inputs=[source, invalid, w.points, phase])
            w.run()
            total.zero_()
            wp.launch(
                consume,
                len(w.tetrahedra),
                inputs=[w.tetrahedra, w.tetrahedra_count, w.error, total],
            )
        wp.capture_launch(cap.graph)
    pointers = (w.points.ptr, w.tetrahedra.ptr, w.tetrahedra_count.ptr)
    for value in [0, 1, 2, 3, 0]:
        phase.fill_(value)
        wp.capture_launch(cap.graph)
        expected = p.copy()
        if value == 3:
            assert w.error.numpy()[0] == 1
            assert w.tetrahedra_count.numpy()[0] == 0
            assert total.numpy()[0] == 0
            continue
        if value == 1:
            expected[0, 0] += np.float32(0.0001)
        if value == 2:
            expected[128:] = expected[0]
        cells = valid(w, expected)
        assert tet_set(cells) == reference(expected[:128] if value == 2 else expected)
        assert w.fast_attempted.numpy()[0] == int(value != 2)
        assert total.numpy()[0] == cells.sum()
        assert pointers == (w.points.ptr, w.tetrahedra.ptr, w.tetrahedra_count.ptr)
    dot = tmp_path / "fast-candidates.dot"
    wp.capture_debug_dot_print(cap.graph, str(dot), verbose=True)
    assert all(word not in dot.read_text() for word in ("MEM_ALLOC", "MEM_FREE", "HOST"))


def test_fast_candidates_can_be_disabled_without_changing_the_geometry():
    p = cloud()
    w = DelaunayWorkspace(p, fast_candidates=False).run()
    assert not w.fast_candidates
    assert w.fast_attempted.numpy()[0] == 0
    assert w.fast_accepted.numpy()[0] == 0
    assert tet_set(valid(w, p)) == reference(p)


def test_fresh_meshing_resets_fast_preference_but_small_active_prefix_stays_float64():
    from tests.test_domain import cube
    from tests.test_mesher import validate
    from warpower import SurfaceDomain, VariationalMeshingWorkspace

    domain = SurfaceDomain(*cube(), surface_subdivisions=4)
    mesh = VariationalMeshingWorkspace(
        domain, 8192, target_count=128, initial_count=64, quality_rounds=0
    )
    assert mesh.core.mesh.fast_candidates
    mesh.core.mesh._fast_available.zero_()
    mesh.run()
    validate(mesh)
    assert mesh.point_count.numpy()[0] == 128
    assert mesh.core.mesh._fast_available.numpy()[0] == 1
    assert mesh.core.mesh.fast_attempted.numpy()[0] == 0
