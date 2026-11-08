"""Analytic exterior/interior/crossing cases at extreme binary32 scales."""

from fractions import Fraction

import numpy as np
import pytest
import warp as wp

from tests.test_domain import combine, cube
from warpower._exterior_proof import centroid_ray, exterior
from warpower._surface_bvh import SurfaceTree

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.kernel
def prove(
    tree: SurfaceTree, points: wp.array[wp.vec3], tets: wp.array2d[int], result: wp.array[int]
):
    i = wp.tid()
    result[i] = exterior(tree, points, wp.vec4i(tets[i, 0], tets[i, 1], tets[i, 2], tets[i, 3]))


def surface_tree(vertices, faces):
    tree = SurfaceTree()
    tree.points = wp.array(vertices, dtype=wp.vec3, device="cuda:0")
    tree.indices = wp.array(faces.ravel(), dtype=int, device="cuda:0")
    tree.n = len(faces)
    return tree


def positive(cells):
    cells = np.asarray(cells, np.float32).copy()
    for cell in cells:
        # Use the unscaled geometry to choose a positive vertex ordering.
        if np.linalg.det(cell[1:].astype(float) - cell[0].astype(float)) < 0:
            cell[[1, 2]] = cell[[2, 1]]
    return cells


def results(vertices, faces, cells):
    tree = surface_tree(vertices, faces)
    points = wp.array(cells.reshape(-1, 3), dtype=wp.vec3, device="cuda:0")
    tets = wp.array(
        np.arange(len(cells) * 4, dtype=np.int32).reshape(-1, 4), dtype=int, device="cuda:0"
    )
    result = wp.empty(len(cells), dtype=int, device="cuda:0")
    wp.load_module(module=prove.module, device="cuda:0")
    with wp.ScopedCapture() as capture:
        wp.launch(prove, len(cells), inputs=[tree, points, tets, result])
    wp.capture_launch(capture.graph)
    return result.numpy()


@pytest.mark.parametrize("exponent", [-140, -100, 0, 100, 125])
def test_nested_cavity_and_enclosing_cell_are_distinguished(exponent):
    outer, inner = cube(2), cube(1)
    vertices, faces = combine(outer, (inner[0], inner[1][:, ::-1].copy()))
    cells = positive(
        [
            # Every vertex of this cavity cell is on surface planes. Its exact
            # centroid, not a strict original vertex, witnesses exterior material.
            [[-1, -1, -1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]],
            [[1.5, 0, 0], [1.75, 0, 0], [1.5, 0.25, 0], [1.5, 0, 0.25]],
            [[3, 3, 3], [3.5, 3, 3], [3, 3.5, 3], [3, 3, 3.5]],
            # Exterior vertices and exterior centroid do not suffice: this cell
            # crosses/encloses the surface and must fail plane separation.
            [[-3, -3, -3], [3, 3, -3], [3, -3, 3], [-3, 3, 3]],
        ]
    )
    scale = np.float32(2.0**exponent)
    np.testing.assert_array_equal(results(vertices * scale, faces, cells * scale), [1, 0, 1, 0])


@pytest.mark.parametrize("exponent", [-100, 0, 100])
def test_disconnected_gap_and_material_bridge(exponent):
    vertices, faces = combine(cube(center=(-2, 0, 0)), cube(center=(2, 0, 0)))
    cells = positive(
        [
            [[-1, -0.5, -0.5], [-1, 0.5, 0.5], [1, -0.5, 0.5], [1, 0.5, -0.5]],
            [[-2, -0.5, -0.5], [-2, 0.5, 0.5], [2, -0.5, 0.5], [2, 0.5, -0.5]],
        ]
    )
    scale = np.float32(2.0**exponent)
    np.testing.assert_array_equal(results(vertices * scale, faces, cells * scale), [1, 0])


def test_rational_centroid_is_not_rounded_onto_surface():
    vertices, faces = cube()
    next_x = np.nextafter(np.float32(1), np.float32(2))
    cells = positive([[[1, 0, 0], [1, 0.5, 0], [1, 0, 0.5], [next_x, 0, 0]]])
    exact = sum(Fraction(float(p[0])) for p in cells[0]) / 4
    assert exact > 1 and np.float32(float(exact)) == 1
    np.testing.assert_array_equal(results(vertices, faces, cells), [1])


@pytest.mark.parametrize("bad", ["nan", "infinity", "flat", "negative"])
def test_nonfinite_or_nonpositive_tetrahedron_never_proved(bad):
    vertices, faces = cube()
    cells = positive([[[2, 0, 0], [3, 0, 0], [2, 1, 0], [2, 0, 1]]])
    if bad == "nan":
        cells[0, 0, 0] = np.nan
    elif bad == "infinity":
        cells[0, 0, 0] = np.inf
    elif bad == "flat":
        cells[0, 3] = cells[0, 2]
    else:
        cells[0, [1, 2]] = cells[0, [2, 1]]
    np.testing.assert_array_equal(results(vertices, faces, cells), [0])


@wp.kernel
def ray_probe(tree: SurfaceTree, cells: wp.array[wp.vec3], values: wp.array[wp.vec2i]):
    i = wp.tid()
    d = wp.vec3(1.0, 1.0, 1.0)
    if i == 1:
        d = wp.vec3(11.0, 23.0, 37.0)
    values[i] = centroid_ray(tree, cells[0], cells[1], cells[2], cells[3], d)


def test_ambiguous_centroid_ray_is_reported_and_other_direction_resolves():
    vertices, faces = cube()
    cells = np.array(
        [[-0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [0.5, -0.5, 0.5], [-0.5, 0.5, 0.5]], np.float32
    )
    assert np.all(cells.astype(float).sum(axis=0) == 0)
    tree = surface_tree(vertices, faces)
    points = wp.array(cells, dtype=wp.vec3, device="cuda:0")
    values = wp.empty(2, dtype=wp.vec2i, device="cuda:0")
    wp.launch(ray_probe, 2, inputs=[tree, points, values])
    result = values.numpy()
    assert result[0, 1] == 1
    np.testing.assert_array_equal(result[1], [1, 0])
