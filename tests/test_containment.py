"""Independent analytic and rational cases for conservative material containment."""

import numpy as np
import warp as wp

import warpower
from tests.test_domain import combine, cube
from tests.test_extraction import extracted
from warpower._containment import contained, ray_count, strict_material_side
from warpower._surface_bvh import SurfaceTree


def cavity():
    inner = cube(0.4)
    return combine(cube(), (inner[0], inner[1][:, ::-1].copy()))


def fan():
    # Five certified full-hull cells around edge 86--303 in the coarse cavity.
    # All lie in x >= 0.625, so none intersects the cavity [-0.4,0.4]^3.
    points = np.array(
        [
            [1.0, -0.6056623458862305, 1.0],
            [0.625, -0.5, 1.0],
            [0.9375, -0.625, 0.9375],
            [1.0, -0.39433759450912476, 1.0],
            [0.7652081847190857, -0.763614296913147, 0.9999986886978149],
            [0.9999988079071045, -0.6420665383338928, 0.7135433554649353],
            [1.0, -0.89433753490448, 0.9999998807907104],
        ],
        np.float32,
    )
    tets = np.array(
        [[0, 1, 2, 4], [0, 2, 5, 6], [2, 0, 4, 6], [1, 0, 2, 3], [0, 2, 3, 5]], np.int32
    )
    return points, tets


@wp.kernel
def classify(
    tree: SurfaceTree, points: wp.array[wp.vec3], tets: wp.array2d[int], result: wp.array[int]
):
    t = wp.tid()
    result[t] = contained(tree, points, wp.vec4i(tets[t, 0], tets[t, 1], tets[t, 2], tets[t, 3]))


@wp.kernel
def query(
    tree: SurfaceTree, points: wp.array[wp.vec3], result: wp.array[int], ambiguity: wp.array[int]
):
    t = wp.tid()
    result[t] = strict_material_side(tree, points[t])
    ambiguity[t] = ray_count(tree, points[t], wp.vec3(1.0, 1.0, 1.0))[1]


def test_exact_containment_rejects_cavity_crossing_despite_material_vertices():
    d = warpower.SurfaceDomain(*cavity())
    p, t = fan()
    adversarial = np.array(
        [
            [[0.8, 0.8, 0.8], [-0.8, -0.8, 0.8], [-0.8, 0.8, -0.8], [0.8, -0.8, -0.8]],
            [[2, 2, 2], [3, 2, 2], [2, 3, 2], [2, 2, 3]],
            [[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]],
            [[0.6, 0.6, 0.6], [0.7, 0.6, 0.6], [0.6, 0.7, 0.6], [0.6, 0.6, 0.7]],
        ],
        np.float32,
    ).reshape(-1, 3)
    # Analytic oracle: all four first-case sites are material, but their mean
    # is the cavity center. A site-membership-only repair would be unsound.
    assert np.all(np.max(np.abs(adversarial[:4]), axis=1) > 0.4)
    assert np.all(adversarial[:4].mean(axis=0) == 0)
    points = wp.array(np.vstack([p, adversarial]), dtype=wp.vec3, device=d.device)
    tets = wp.array(
        np.vstack([t[:2], len(p) + np.arange(16).reshape(-1, 4)]).astype(np.int32), device=d.device
    )
    result = wp.empty(len(tets), dtype=int, device=d.device)
    wp.launch(classify, len(tets), inputs=[d.tree, points, tets, result])
    np.testing.assert_array_equal(result.numpy(), [1, 1, 0, 0, 0, 1])


def test_corner_ray_ambiguity_retries_and_disconnected_gap_is_rejected():
    d = warpower.SurfaceDomain(*cube())
    points = wp.array(
        np.array([[0, 0, 0], [0.2, 0.1, 0.3], [2, 0, 0]], np.float32),
        dtype=wp.vec3,
        device=d.device,
    )
    result, ambiguity = (
        wp.empty(3, dtype=int, device=d.device),
        wp.empty(3, dtype=int, device=d.device),
    )
    wp.launch(query, 3, inputs=[d.tree, points, result, ambiguity])
    np.testing.assert_array_equal(result.numpy(), [1, 1, 0])
    assert ambiguity.numpy()[0] == 1
    d = warpower.SurfaceDomain(*combine(cube(center=(-2, 0, 0)), cube(center=(2, 0, 0))))
    p = np.array([[-2, 0, 0], [-2, 0.1, 0.1], [2, 0, 0.1], [2, 0.1, 0]], np.float32)
    t = wp.array(np.array([[0, 1, 2, 3]], np.int32), device=d.device)
    wp.launch(classify, 1, inputs=[d.tree, wp.array(p, dtype=wp.vec3, device=d.device), t, result])
    assert result.numpy()[0] == 0


def test_recovery_adds_only_contained_full_hull_cells_and_validates_topology():
    d = warpower.SurfaceDomain(*cavity())
    p, t = fan()
    plain = warpower.ExtractionWorkspace(d, p, t).run()
    assert plain.error.numpy()[0] == 0 and plain.diagnostics.numpy()[0] == 2
    assert plain.tetrahedra_count.numpy()[0] == 3
    e = warpower.ExtractionWorkspace(d, p, t, containment_recovery=True).run()
    _, actual, _ = extracted(e)
    assert len(actual) == 5
    assert e.repaired_count.numpy()[0] == 2 and e.repair_rounds.numpy()[0] == 1
    assert e.component_count.numpy()[0] == e.shell_count.numpy()[0] == 1
    assert e.shell_euler.numpy().sum() == 2
    np.testing.assert_array_equal(e.source_indices.numpy()[:5], np.arange(5))
    # A valid selection has no repair work or added cells.
    healthy = warpower.ExtractionWorkspace(
        warpower.SurfaceDomain(*cube(2)), p, t, containment_recovery=True
    ).run()
    extracted(healthy)
    assert healthy.repaired_count.numpy()[0] == healthy.repair_rounds.numpy()[0] == 0


@wp.kernel
def producer(base: wp.array[wp.vec3], points: wp.array[wp.vec3], phase: wp.array[int]):
    i = wp.tid()
    points[i] = base[i]
    if phase[0] == 1:
        points[i] = wp.vec3(wp.nan)
    elif phase[0] == 2:
        points[i] = base[i] + wp.vec3(8.0, 0.0, 0.0)


@wp.kernel
def consume(
    phase: wp.array[int],
    error: wp.array[int],
    valid: wp.array[int],
    count: wp.array[int],
    added: wp.array[int],
    out: wp.array2d[int],
):
    k = phase[0]
    out[k, 0] = error[0]
    out[k, 1] = valid[0]
    out[k, 2] = count[0]
    out[k, 3] = added[0]
    phase[0] += 1


def test_captured_recovery_clears_failed_or_empty_frames_and_restores(monkeypatch, tmp_path):
    d = warpower.SurfaceDomain(*cavity())
    p, t = fan()
    base = wp.array(p, dtype=wp.vec3, device=d.device)
    points = wp.clone(base)
    # Inactive invalid padding must never reach recovery.
    t = np.vstack([t, [-1, -1, -1, -1]]).astype(np.int32)
    count = wp.full(1, 5, dtype=int, device=d.device)
    e = warpower.ExtractionWorkspace(d, points, t, count, containment_recovery=True)
    phase = wp.zeros(1, dtype=int, device=d.device)
    out = wp.zeros((4, 4), dtype=int, device=d.device)
    wp.load_module(module=__name__, device=d.device)

    def forbidden(*args, **kwargs):
        raise AssertionError("Host allocation, readback or sync during capture")

    with monkeypatch.context() as patch:
        for name in [
            "empty",
            "zeros",
            "ones",
            "full",
            "empty_like",
            "clone",
            "synchronize",
            "synchronize_device",
            "synchronize_stream",
        ]:
            patch.setattr(wp, name, forbidden)
        patch.setattr(wp.array, "numpy", forbidden)
        with wp.ScopedCapture() as cap:
            wp.launch(producer, len(points), inputs=[base, points, phase])
            e.run()
            wp.launch(
                consume,
                1,
                inputs=[phase, e.error, e.valid, e.tetrahedra_count, e.repaired_count, out],
            )
        for _ in range(4):
            wp.capture_launch(cap.graph)
    np.testing.assert_array_equal(
        out.numpy(), [[0, 1, 5, 2], [4, 0, 0, 0], [0, 1, 0, 0], [0, 1, 5, 2]]
    )
    extracted(e)
    dot = tmp_path / "contained.dot"
    wp.capture_debug_dot_print(cap.graph, str(dot), verbose=True)
    text = dot.read_text()
    assert "MEM_ALLOC" not in text and "MEM_FREE" not in text and "HOST" not in text
