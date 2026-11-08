"""Original analytic cases plus exhaustive-proof differential tests and captured refit."""

from fractions import Fraction as F

import numpy as np
import pytest
import warp as wp

import tests.test_exterior_proof as original
from warpower._exterior_proof import centroid_ray as exhaustive
from warpower._exterior_proof_bvh import centroid_box, centroid_ray, exterior
from warpower._surface_bvh import SurfaceHierarchy, SurfaceTree

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.kernel
def prove_bvh(
    tree: SurfaceTree, points: wp.array[wp.vec3], tets: wp.array2d[int], result: wp.array[int]
):
    i = wp.tid()
    result[i] = exterior(tree, points, wp.vec4i(tets[i, 0], tets[i, 1], tets[i, 2], tets[i, 3]))


@pytest.fixture(autouse=True)
def compare_original_results(monkeypatch):
    old = original.results

    def compare(vertices, faces, cells):
        expected = old(vertices, faces, cells)
        p = wp.array(vertices, dtype=wp.vec3, device="cuda:0")
        f = wp.array(faces.ravel(), dtype=int, device="cuda:0")
        hierarchy = SurfaceHierarchy(p, f)
        points = wp.array(cells.reshape(-1, 3), dtype=wp.vec3, device="cuda:0")
        tets = wp.array(
            np.arange(len(cells) * 4, dtype=np.int32).reshape(-1, 4), dtype=int, device="cuda:0"
        )
        actual = wp.empty(len(cells), dtype=int, device="cuda:0")
        wp.load_module(module=__name__, device="cuda:0")
        with wp.ScopedCapture() as cap:
            hierarchy.refit()
            wp.launch(
                prove_bvh, len(cells), inputs=[hierarchy.tree, points, tets, actual], block_dim=32
            )
        wp.capture_launch(cap.graph)
        np.testing.assert_array_equal(actual.numpy(), expected)
        return actual.numpy()

    monkeypatch.setattr(original, "results", compare)


@pytest.mark.parametrize("exponent", [-140, -100, 0, 100, 125])
def test_random_cells_match_exhaustive_plane_scan(exponent):
    rng = np.random.default_rng(239)
    vertices, faces = original.cube()
    cells = original.positive(rng.uniform(-3, 3, size=(256, 4, 3)))
    scale = np.float32(2.0**exponent)
    original.results(vertices * scale, faces, cells * scale)


@wp.kernel
def deform(source: wp.array[wp.vec3], dest: wp.array[wp.vec3], offset: wp.array[float]):
    i = wp.tid()
    dest[i] = source[i] + wp.vec3(offset[0], 0.0, 0.0)


@pytest.mark.parametrize("faces_count", [1, 3, 4, 5, 12])
def test_nondefault_stream_refit_updates_bounds_and_odd_leaf_counts(faces_count):
    v, f = original.cube()
    f = f[:faces_count]
    p = wp.array(v, dtype=wp.vec3, device="cuda:0")
    source = wp.clone(p)
    faces = wp.array(f.ravel(), dtype=int, device="cuda:0")
    h = SurfaceHierarchy(p, faces)
    cells = original.positive(np.random.default_rng(14).uniform(-3, 3, size=(64, 4, 3)))
    points = wp.array(cells.reshape(-1, 3), dtype=wp.vec3, device="cuda:0")
    tets = wp.array(np.arange(256, dtype=np.int32).reshape(-1, 4), dtype=int, device="cuda:0")
    old = wp.empty(64, dtype=int, device="cuda:0")
    new = wp.empty_like(old)
    offset = wp.zeros(1, dtype=float, device="cuda:0")
    stream = wp.Stream("cuda:0")
    wp.load_module(module=__name__, device="cuda:0")
    wp.load_module(module=original.prove.module, device="cuda:0")
    wp.synchronize()
    with wp.ScopedStream(stream), wp.ScopedCapture(stream=stream) as cap:
        wp.launch(deform, len(v), inputs=[source, p, offset])
        h.refit()
        wp.launch(original.prove, 64, inputs=[h.tree, points, tets, old], block_dim=32)
        wp.launch(prove_bvh, 64, inputs=[h.tree, points, tets, new], block_dim=32)
    for value in [0.0, 100.0, -100.0, 0.0]:
        offset.assign(np.array([value], np.float32))
        wp.synchronize()
        wp.capture_launch(cap.graph, stream=stream)
        wp.synchronize_stream(stream)
        np.testing.assert_array_equal(new.numpy(), old.numpy())


@pytest.mark.parametrize("exponent", [-140, -100, 0, 100, 125])
def test_nested_cavity_enclosing_cell(exponent):
    original.test_nested_cavity_and_enclosing_cell_are_distinguished(exponent)


@pytest.mark.parametrize("exponent", [-100, 0, 100])
def test_disconnected_gap_material_bridge(exponent):
    original.test_disconnected_gap_and_material_bridge(exponent)


def test_unrounded_centroid_near_surface():
    original.test_rational_centroid_is_not_rounded_onto_surface()


@pytest.mark.parametrize("bad", ["nan", "infinity", "flat", "negative"])
def test_invalid_cells_never_proved(bad):
    original.test_nonfinite_or_nonpositive_tetrahedron_never_proved(bad)


@wp.kernel
def boxes(
    lo: wp.array[wp.vec3],
    hi: wp.array[wp.vec3],
    p: wp.array2d[wp.vec3],
    d: wp.array[wp.vec3],
    out: wp.array[int],
):
    i = wp.tid()
    out[i] = int(centroid_box(lo[i], hi[i], p[i, 0], p[i, 1], p[i, 2], p[i, 3], d[i]))


def intersects(lo, hi, p, d):
    origin = [sum(F(float(v)) for v in p[:, k]) / 4 for k in range(3)]
    enter = [F(0)]
    leave = []
    for k in range(3):
        a = (F(float(lo[k])) - origin[k]) / F(float(d[k]))
        b = (F(float(hi[k])) - origin[k]) / F(float(d[k]))
        enter.append(min(a, b))
        leave.append(max(a, b))
    return max(enter) <= min(leave)


def test_box_pruning_never_discards_exact_rational_intersection():
    rng = np.random.default_rng(239)
    data = []
    dirs = np.array([[11, 23, 37], [-17, 29, 43], [31, -47, 19]], np.float32)
    for exponent in [-140, -100, -20, 0, 20, 100, 125]:
        scale = np.float32(2.0**exponent)
        for i in range(192):
            p = rng.uniform(-2, 2, (4, 3)).astype(np.float32) * scale
            a = rng.uniform(-3, 3, 3).astype(np.float32) * scale
            b = rng.uniform(-3, 3, 3).astype(np.float32) * scale
            if i % 3 == 0:
                center = p.astype(np.float64).sum(axis=0) / 4
                a = np.nextafter(center.astype(np.float32), np.float32(-np.inf))
                b = np.nextafter(center.astype(np.float32), np.float32(np.inf))
            data.append((np.minimum(a, b), np.maximum(a, b), p, dirs[i % 3]))
    lo, hi, p, d = (np.stack([row[k] for row in data]) for k in range(4))
    assert np.isfinite(lo).all() and np.isfinite(hi).all() and np.isfinite(p).all()
    args = [wp.array(v, dtype=wp.vec3, device="cuda:0") for v in [lo, hi, p, d]]
    result = wp.empty(len(data), dtype=int, device="cuda:0")
    wp.load_module(module=__name__, device="cuda:0")
    with wp.ScopedCapture() as cap:
        wp.launch(boxes, len(data), inputs=args + [result])
    wp.capture_launch(cap.graph)
    actual = result.numpy()
    exact = np.array([intersects(*row) for row in data])
    assert exact.sum() > 400 and (actual == 0).sum() > 100
    assert np.all(actual[exact] == 1), np.flatnonzero(exact & (actual == 0))


@wp.kernel
def ray_pair(tree: SurfaceTree, p: wp.array[wp.vec3], out: wp.array[wp.vec2i]):
    d = wp.vec3(11.0, 23.0, 37.0)
    out[0] = exhaustive(tree, p[0], p[1], p[2], p[3], d)
    out[1] = centroid_ray(tree, p[0], p[1], p[2], p[3], d)


def test_disjoint_coplanar_triangle_is_safely_pruned():
    direction = np.array([11, 23, 37], np.float32)
    edge = np.array([23, -11, 0], np.float32)
    v = np.array([1000 * edge, 1001 * edge, 1000 * edge + direction], np.float32)
    cells = np.array([[-1, -1, -1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]], np.float32)
    assert not intersects(v.min(0), v.max(0), cells, direction)
    points = wp.array(v, dtype=wp.vec3, device="cuda:0")
    faces = wp.array([0, 1, 2], dtype=int, device="cuda:0")
    h = SurfaceHierarchy(points, faces)
    p = wp.array(cells, dtype=wp.vec3, device="cuda:0")
    out = wp.empty(2, dtype=wp.vec2i, device="cuda:0")
    wp.launch(ray_pair, 1, inputs=[h.tree, p, out])
    actual = out.numpy()
    # Exhaustive code conservatively reports an infinite-plane ambiguity. The
    # exact AABB proof establishes that this triangle cannot meet the forward ray.
    np.testing.assert_array_equal(actual, [[0, 1], [0, 0]])
