import numpy as np
import pytest
import warp as wp

from examples.vtm_inputs import counterbored_plate, surface
from tests.surface_traversal_reference import (
    reference_closest_point,
    reference_material_side,
    reference_ray_count,
)
from tests.test_domain import brute_distances, cube, solid_angle_inside, torus
from warpower import DomainQueryWorkspace, SurfaceDomain
from warpower._sampling import boundary_box
from warpower._surface_bvh import (
    SurfaceHierarchy,
    SurfaceTree,
    closest_point,
    material_side,
    ray_count,
)

# Exercise both caller contraction settings with triangle arithmetic in place.
wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.kernel
def compare(
    tree: SurfaceTree,
    segment: SurfaceTree,
    points: wp.array[wp.vec3d],
    excluded: wp.array[int],
    differences: wp.array[int],
    membership: wp.array[int],
    distances: wp.array[wp.float64],
    point_difference: wp.array[wp.float64],
):
    i = wp.tid()
    p = points[i]
    direction = wp.vec3d(1.0, 1.0, 1.0)
    before = reference_ray_count(segment, p, direction, excluded[i])
    after = ray_count(tree, p, direction, excluded[i])
    side = material_side(tree, p, excluded[i])
    differences[i] = int(
        before[0] != after[0]
        or before[1] != after[1]
        or side != reference_material_side(segment, p, excluded[i])
    )
    membership[i] = side
    old_closest = reference_closest_point(segment, p, wp.float64(1e100))
    new_closest = closest_point(tree, p, wp.float64(1e100))
    differences[i] |= 4 * int(old_closest.distance2 != new_closest.distance2)
    distances[i] = wp.sqrt(new_closest.distance2)
    point_difference[i] = wp.length(old_closest.point - new_closest.point)


@pytest.mark.parametrize("exponent", [-60, 0, 60])
@pytest.mark.parametrize("case", ["cube", "torus", "plate", "thin-hand", "cavity", "disconnected"])
@pytest.mark.parametrize("fuse_fp", [False, True])
def test_radix_queries_match_heap_and_independent_geometry(case, exponent, fuse_fp):
    wp.set_module_options({"fuse_fp": fuse_fp}, module=__name__)
    p, f = (
        cube()
        if case == "cube"
        else torus(16, 8)
        if case == "torus"
        else counterbored_plate(16)
        if case == "plate"
        else surface(case)
    )
    p = (p * np.float32(2.0**exponent)).astype(np.float32)
    domain = SurfaceDomain(p, f, surface_subdivisions=1, crease_subdivisions=1)
    assert domain.error.numpy()[0] == 0
    rng = np.random.default_rng(918)
    lo, hi = p.min(axis=0).astype(float), p.max(axis=0).astype(float)
    ordinary = rng.uniform(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo), (100, 3))
    triangles = p[f].astype(float)
    centers = triangles.mean(axis=1)
    probes = np.concatenate([ordinary, p, triangles[:, :2].mean(axis=1), centers, centers])
    excluded = np.r_[np.full(len(probes) - len(f), -1, np.int32), np.arange(len(f), dtype=np.int32)]
    points = wp.array(probes, dtype=wp.vec3d)
    faces = wp.array(excluded, dtype=int)
    differences = wp.empty(len(probes), dtype=int)
    membership = wp.empty_like(differences)
    distances = wp.empty(len(probes), dtype=wp.float64)
    point_difference = wp.empty_like(distances)
    wp.launch(
        compare,
        len(probes),
        inputs=[
            domain.tree,
            domain._hierarchy._segment,
            points,
            faces,
            differences,
            membership,
            distances,
            point_difference,
        ],
    )
    np.testing.assert_array_equal(differences.numpy(), 0)
    np.testing.assert_allclose(point_difference.numpy() / (2.0**exponent), 0, atol=2e-14)
    expected_distance = brute_distances(p, f, ordinary)
    np.testing.assert_allclose(distances.numpy()[: len(ordinary)], expected_distance, rtol=2e-10)
    np.testing.assert_array_equal(
        membership.numpy()[: len(ordinary)], solid_angle_inside(p, f, ordinary).astype(np.int32)
    )


@wp.kernel
def producer(
    source: wp.array[wp.vec3],
    output: wp.array[wp.vec3],
    phase: wp.array[int],
    is_domain: bool,
):
    i = wp.tid()
    p = source[i]
    state = phase[0]
    if state == 1 or state == 6:
        p = wp.vec3(1.5 * p[0] + 0.2 * p[1] + 0.125, 0.5 * p[1] - 0.25, 2.0 * p[2])
    if is_domain:
        if state == 2:
            p[2] = 0.0
        elif state == 3 and i == 0:
            p[0] = wp.nan
        elif state == 4:
            p[0] = -p[0]
    output[i] = p


@wp.kernel
def advance(phase: wp.array[int]):
    phase[0] += 1


@wp.kernel
def query_boundary_boxes(
    tree: SurfaceTree,
    lower: wp.array[wp.vec3d],
    upper: wp.array[wp.vec3d],
    output: wp.array[int],
):
    i = wp.tid()
    output[i] = int(boundary_box(tree, lower[i], upper[i]))


def test_radix_bounds_refresh_in_captured_deformation_invalid_and_recovery_phases():
    p, f = cube()
    # A thin box tests refits without a normal-offset classification shortcut.
    p[:, 2] *= np.float32(1e-7)
    domain = SurfaceDomain(p, f, surface_subdivisions=1, crease_subdivisions=1)
    probes = np.array([[0, 0, 0], [1.1, 0, 0], [0, 0, 2e-7], [0.2, -0.3, -0.9e-7]], np.float32)
    query = DomainQueryWorkspace(domain, probes)
    source = wp.array(p, dtype=wp.vec3)
    query_source = wp.array(probes, dtype=wp.vec3)
    phase = wp.zeros(1, dtype=int)
    wp.load_module(module=__name__, device=domain.device)
    with wp.ScopedCapture() as capture:
        wp.launch(producer, len(p), inputs=[source, domain.points, phase, True])
        wp.launch(producer, len(probes), inputs=[query_source, query.points, phase, False])
        domain.refit()
        query.run()
        wp.launch(advance, 1, inputs=[phase])
    for state in range(7):
        wp.capture_launch(capture.graph)
        wp.synchronize()
        if state in [2, 3, 4]:
            assert domain.error.numpy()[0] != 0
            np.testing.assert_array_equal(query.valid.numpy(), 0)
        else:
            assert domain.error.numpy()[0] == query.error.numpy()[0] == 0
            np.testing.assert_array_equal(query.valid.numpy(), 1)
            np.testing.assert_array_equal(query.inside.numpy(), [1, 0, 0, 1])
            # Independent current-geometry winding catches stale bounds even
            # when the fixed hierarchy bounds were correctly refreshed.
            expected = solid_angle_inside(domain.points.numpy(), f, query.points.numpy())
            np.testing.assert_array_equal(query.inside.numpy(), expected.astype(np.int32))


def verify_topology_and_bounds(hierarchy, points, faces):
    tree = hierarchy.tree
    ids = tree.ids.numpy()[: len(faces)]
    np.testing.assert_array_equal(np.sort(ids), np.arange(len(faces)))
    left, right, escape = tree.left.numpy(), tree.right.numpy(), tree.escape.numpy()
    lower, upper = tree.lower.numpy(), tree.upper.numpy()
    visited = set()
    preorder = []
    expected_escape = {}

    def visit(node, following, depth):
        assert node not in visited and 1 <= node < 2 * tree.leaf_base
        assert depth < 96
        visited.add(node)
        preorder.append(node)
        expected_escape[node] = following
        if node >= tree.leaf_base:
            group = node - tree.leaf_base
            descendant_faces = list(ids[4 * group : min(4 * group + 4, len(faces))])
            assert left[node] == right[node] == 0
        else:
            # The right-first links must skip whole subtrees, including the root.
            descendant_faces = visit(right[node], left[node], depth + 1)
            descendant_faces += visit(left[node], following, depth + 1)
        vertices = points[faces[descendant_faces]].reshape(-1, 3)
        np.testing.assert_array_equal(lower[node], vertices.min(axis=0))
        np.testing.assert_array_equal(upper[node], vertices.max(axis=0))
        return descendant_faces

    descendants = visit(1, 0, 0)
    np.testing.assert_array_equal(np.sort(descendants), np.arange(len(faces)))
    assert len(visited) == 2 * tree.leaf_base - 1
    for node, following in expected_escape.items():
        assert escape[node] == following
    node, walk = 1, []
    while node:
        assert node not in walk
        walk.append(node)
        node = right[node] if node < tree.leaf_base else escape[node]
    assert walk == preorder

    # The sampler's conservative box test must retain every overlapping face.
    rng = np.random.default_rng(781)
    lo, hi = points.min(axis=0).astype(float), points.max(axis=0).astype(float)
    widths = np.maximum(hi - lo, 1.0)
    centers = rng.uniform(lo - widths, hi + widths, (100, 3))
    lower = centers - 0.1 * widths
    upper = centers + 0.1 * widths
    face_lower, face_upper = points[faces].min(axis=1), points[faces].max(axis=1)
    expected = np.array(
        [
            np.any(np.all(face_lower <= b, axis=1) & np.all(face_upper >= a, axis=1))
            for a, b in zip(lower, upper)
        ],
        np.int32,
    )
    output = wp.empty(len(lower), dtype=int)
    wp.launch(
        query_boundary_boxes,
        len(lower),
        inputs=[tree, wp.array(lower, dtype=wp.vec3d), wp.array(upper, dtype=wp.vec3d), output],
    )
    np.testing.assert_array_equal(output.numpy(), expected)


@pytest.mark.parametrize("count", [1, 4, 5, 17, 128, 257])
@pytest.mark.parametrize("equal_keys", [False, True])
def test_radix_ranges_escape_and_exact_bounds_after_deformation(count, equal_keys):
    rng = np.random.default_rng(712)
    points = rng.normal(size=(count * 3, 3)).astype(np.float32)
    faces = np.arange(count * 3, dtype=np.int32).reshape(-1, 3)
    if equal_keys:
        # All centers coincide, requiring sorted-index prefix tie breaking.
        points[:] = np.tile(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32), (count, 1))
    device_points = wp.array(points, dtype=wp.vec3)
    hierarchy = SurfaceHierarchy(device_points, wp.array(faces.ravel(), dtype=int))
    verify_topology_and_bounds(hierarchy, points, faces)
    ids = hierarchy.tree.ids.numpy().copy()
    topology = [
        a.numpy().copy() for a in [hierarchy.tree.left, hierarchy.tree.right, hierarchy.tree.escape]
    ]
    # Refit can scramble all initial spatial positions without changing coverage.
    moved = rng.normal(size=points.shape).astype(np.float32) * np.float32(100)
    device_points.assign(moved)
    with wp.ScopedCapture() as capture:
        hierarchy.refit()
    wp.capture_launch(capture.graph)
    verify_topology_and_bounds(hierarchy, moved, faces)
    np.testing.assert_array_equal(hierarchy.tree.ids.numpy(), ids)
    for value, old in zip(
        [hierarchy.tree.left, hierarchy.tree.right, hierarchy.tree.escape], topology
    ):
        np.testing.assert_array_equal(value.numpy()[1:], old[1:])
