import numpy as np
import pytest
import warp as wp

import warpower
from tests.test_domain import torus
from tests.test_vtm_core import torus_sites
from warpower._graded_odt import accumulate_graded_tets
from warpower._vtm import VTMWorkspace


def oracle(samples, values, queries, gradation, minimum=1e-6, maximum=1e30):
    distances = np.linalg.norm(
        queries.astype(np.float64)[:, None] - samples.astype(np.float64)[None], axis=2
    )
    return np.clip(
        np.min(values.astype(np.float64)[None] + gradation * distances, axis=1), minimum, maximum
    )


@pytest.mark.parametrize("n,gradation", [(1, 0.0), (1, 1.0), (13, 0.0), (257, 0.4), (257, 2.0)])
def test_sampled_envelope_matches_brute_force_and_lipschitz_bound(n, gradation):
    rng = np.random.default_rng(8563)
    samples = rng.normal(size=(n, 3)).astype(np.float32)
    values = rng.uniform(0.03, 1.0, n).astype(np.float32)
    queries = rng.normal(size=(130, 3)).astype(np.float32)
    field = warpower.SizingField(
        samples, wp.array(values, device="cuda:0"), gradation=gradation, minimum=0.1, maximum=1.2
    )
    query = warpower.SizingQueryWorkspace(field, queries).run()
    assert field.error.numpy()[0] == 0
    assert query.error.numpy()[0] == 0
    actual = query.values.numpy()
    expected = oracle(samples, values, queries, gradation, 0.1, 1.2)
    np.testing.assert_allclose(actual, expected, rtol=8e-8)
    distances = np.linalg.norm(queries.astype(float)[:, None] - queries.astype(float)[None], axis=2)
    assert np.all(np.abs(actual[:, None] - actual[None]) <= gradation * distances + 2e-7)


def test_samples_are_upper_constraints_not_interpolation_knots():
    samples = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]], np.float32)
    values = wp.array([0.1, 3.0, 2.0], dtype=float, device="cuda:0")
    field = warpower.SizingField(samples, values, gradation=0.5)
    query = warpower.SizingQueryWorkspace(field, samples).run()
    np.testing.assert_allclose(query.values.numpy(), [0.1, 0.6, 0.6], rtol=1e-7)


@wp.kernel
def producer(points: wp.array[wp.vec3], values: wp.array[float], phase: wp.array[int]):
    i = wp.tid()
    points[i] = wp.vec3(float(i), float(phase[0]), 0.0)
    values[i] = 0.1 + 0.03 * float(i) + 0.02 * float(phase[0])
    if phase[0] == 2 and i == 0:
        values[i] = -1.0


@wp.kernel
def consumer(values: wp.array[float], error: wp.array[int], total: wp.array[wp.float64]):
    if error[0] == 0:
        wp.atomic_add(total, 0, wp.float64(values[wp.tid()]))


def test_capture_updates_samples_rejects_invalid_and_recovers_without_host_work(
    monkeypatch, tmp_path
):
    points = wp.zeros(17, dtype=wp.vec3, device="cuda:0")
    values = wp.ones(17, dtype=float, device="cuda:0")
    phase = wp.zeros(1, dtype=int, device="cuda:0")
    field = warpower.SizingField(points, values, gradation=0.3)
    positions = np.random.default_rng(34).normal(size=(37, 3)).astype(np.float32)
    query = warpower.SizingQueryWorkspace(field, positions)
    total = wp.zeros(1, dtype=wp.float64, device="cuda:0")
    wp.load_module(module=__name__, device="cuda:0")

    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected host readback, allocation or synchronization")

    with monkeypatch.context() as patch:
        for name in (
            "empty",
            "zeros",
            "ones",
            "full",
            "empty_like",
            "clone",
            "synchronize",
            "synchronize_device",
            "synchronize_stream",
        ):
            patch.setattr(wp, name, forbidden)
        patch.setattr(wp.array, "numpy", forbidden)
        with wp.ScopedCapture() as capture:
            wp.launch(producer, len(points), inputs=[points, values, phase])
            field.refit()
            query.run()
            total.zero_()
            wp.launch(consumer, len(positions), inputs=[query.values, query.error, total])
        wp.capture_launch(capture.graph)
    dot = tmp_path / "sizing.dot"
    wp.capture_debug_dot_print(capture.graph, str(dot), verbose=True)
    graph = dot.read_text()
    assert all(node not in graph for node in ("MEM_ALLOC", "MEM_FREE", "HOST"))
    for p in (0, 1, 2, 3):
        phase.fill_(p)
        wp.capture_launch(capture.graph)
        if p == 2:
            assert field.error.numpy()[0] == 2
            assert query.error.numpy()[0] == 1
            np.testing.assert_array_equal(query.values.numpy(), 0)
            assert total.numpy()[0] == 0
        else:
            assert field.error.numpy()[0] == 0
            assert query.error.numpy()[0] == 0
            expected = oracle(points.numpy(), values.numpy(), positions, 0.3)
            np.testing.assert_allclose(query.values.numpy(), expected, rtol=8e-8)
            np.testing.assert_allclose(
                total.numpy()[0], query.values.numpy().astype(float).sum(), rtol=1e-14
            )


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_sizes_are_device_errors(bad):
    field = warpower.SizingField(
        np.zeros((1, 3), np.float32), wp.array([bad], dtype=float, device="cuda:0")
    )
    query = warpower.SizingQueryWorkspace(field, np.zeros((2, 3), np.float32)).run()
    assert field.error.numpy()[0] == 2
    assert query.error.numpy()[0] == 1
    np.testing.assert_array_equal(query.values.numpy(), 0)


def test_invalid_sample_positions_and_queries_have_distinct_errors():
    field = warpower.SizingField(
        np.array([[np.nan, 0, 0]], np.float32), wp.ones(1, device="cuda:0")
    )
    assert field.error.numpy()[0] == 1
    field.points.zero_()
    field.refit()
    query = warpower.SizingQueryWorkspace(
        field, np.array([[np.inf, 0, 0], [0, 0, 0]], np.float32)
    ).run()
    assert field.error.numpy()[0] == 0
    assert query.error.numpy()[0] == 2
    np.testing.assert_array_equal(query.values.numpy(), [0, 1])


def test_graded_accumulation_matches_independent_circumcenter_oracle():
    from scipy.spatial import Delaunay

    rng = np.random.default_rng(590)
    points = rng.normal(size=(40, 3)).astype(np.float32)
    tets = Delaunay(points).simplices.copy().astype(np.int32)
    for tet in tets:
        if np.linalg.det(points[tet[1:]].astype(float) - points[tet[0]]) < 0:
            tet[2], tet[3] = tet[3], tet[2]
    samples = np.array([[-1, 0, 0], [1, 0, 0]], np.float32)
    sizes = np.array([0.2, 0.8], np.float32)
    expected_mass = np.zeros(len(points))
    expected_moment = np.zeros((len(points), 3))
    expected_energy, expected_volume = 0.0, 0.0
    for tet in tets:
        p = points[tet].astype(float)
        a = p[1:] - p[0]
        volume = np.linalg.det(a) / 6
        center = p[0] + np.linalg.solve(2 * a, np.sum(a * a, axis=1))
        mu = oracle(samples, sizes, p.mean(axis=0)[None], 0.4)[0]
        mass = volume / mu**3
        expected_mass[tet] += mass
        expected_moment[tet] += mass * (center - p)
        expected_energy += (
            mass / 20 * sum(np.dot(p[i] - p[j], p[i] - p[j]) for i in range(4) for j in range(i))
        )
        expected_volume += volume
    with wp.ScopedDevice("cuda:0"):
        field = warpower.SizingField(samples, wp.array(sizes), gradation=0.4)
        p, t = wp.array(points, dtype=wp.vec3), wp.array(tets)
        count, error, invalid = (
            wp.array([len(tets)], dtype=int),
            wp.zeros(1, dtype=int),
            wp.zeros(1, dtype=int),
        )
        mass, moment = (
            wp.zeros(len(points), dtype=wp.float64),
            wp.zeros(len(points), dtype=wp.vec3d),
        )
        edge = wp.full(len(points), 1e300, dtype=wp.float64)
        energy, volume, quality = (
            wp.zeros(1, dtype=wp.float64),
            wp.zeros(1, dtype=wp.float64),
            wp.ones(1, dtype=wp.float64),
        )
        wp.launch(
            accumulate_graded_tets,
            len(tets),
            inputs=[
                p,
                t,
                count,
                error,
                invalid,
                mass,
                moment,
                edge,
                energy,
                volume,
                quality,
                True,
                field.data,
            ],
        )
    assert invalid.numpy()[0] == 0
    np.testing.assert_allclose(mass.numpy(), expected_mass, rtol=2e-13)
    np.testing.assert_allclose(moment.numpy(), expected_moment, rtol=3e-12, atol=1e-12)
    np.testing.assert_allclose(energy.numpy()[0], expected_energy, rtol=2e-13)
    np.testing.assert_allclose(volume.numpy()[0], expected_volume, rtol=2e-13)


def test_constant_unit_sizing_reproduces_uniform_vtm_and_graded_field_changes_it():
    domain = warpower.SurfaceDomain(*torus(16, 8), surface_subdivisions=2)
    original = torus_sites(90)
    plain = VTMWorkspace(domain, original.copy(), max_iterations=3).run()
    unit = warpower.SizingField(
        np.zeros((1, 3), np.float32), wp.ones(1, device="cuda:0"), gradation=0
    )
    same = VTMWorkspace(domain, original.copy(), max_iterations=3, sizing=unit).run()
    assert plain.error.numpy()[0] == same.error.numpy()[0] == 0
    np.testing.assert_allclose(same.points.numpy(), plain.points.numpy(), rtol=1e-6, atol=2e-7)
    np.testing.assert_allclose(same.energy_history.numpy(), plain.energy_history.numpy(), rtol=1e-6)
    field = warpower.SizingField(
        np.array([[2, 0, 0]], np.float32),
        wp.array([0.3], dtype=float, device="cuda:0"),
        gradation=0.4,
    )
    graded = VTMWorkspace(domain, original.copy(), max_iterations=3, sizing=field).run()
    assert graded.error.numpy()[0] == 0
    assert graded.iterations.numpy()[0] == 3
    assert np.max(np.linalg.norm(graded.points.numpy() - plain.points.numpy(), axis=1)) > 0.05
    # Surface/crease weights use the same field; evaluate their actual locations.
    np.testing.assert_allclose(
        graded.boundary.surface_sizing.numpy(),
        oracle(field.points.numpy(), field.values.numpy(), domain.surface_points.numpy(), 0.4),
        rtol=8e-8,
    )


@wp.kernel
def produce_vtm(
    source: wp.array[wp.vec3],
    points: wp.array[wp.vec3],
    values: wp.array[float],
    phase: wp.array[int],
):
    i = wp.tid()
    points[i] = source[i]
    if i == 0:
        values[0] = 0.3 + float(phase[0]) * 0.05
        if phase[0] == 2:
            values[0] = -1.0


def test_graded_vtm_captures_producers_field_and_consumers_and_recovers(monkeypatch, tmp_path):
    from tests.test_vtm_core import consume, valid_core

    domain = warpower.SurfaceDomain(*torus(16, 8), surface_subdivisions=2)
    with wp.ScopedDevice("cuda:0"):
        source = wp.array(torus_sites(90), dtype=wp.vec3)
        points = wp.clone(source)
        phase = wp.zeros(1, dtype=int)
        values = wp.array([0.3], dtype=float)
        field = warpower.SizingField(np.array([[2, 0, 0]], np.float32), values, gradation=0.4)
        workspace = VTMWorkspace(domain, points, max_iterations=2, sizing=field)
        volume = wp.zeros(1, dtype=wp.float64)
        wp.load_module(module=__name__)
        wp.load_module(module="tests.test_vtm_core")

    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected host work during composed graded VTM")

    with monkeypatch.context() as patch:
        for name in (
            "empty",
            "zeros",
            "ones",
            "full",
            "empty_like",
            "zeros_like",
            "clone",
            "synchronize",
            "synchronize_device",
            "synchronize_stream",
        ):
            patch.setattr(wp, name, forbidden)
        patch.setattr(wp.array, "numpy", forbidden)
        with wp.ScopedCapture() as capture:
            wp.launch(produce_vtm, len(points), inputs=[source, points, values, phase])
            field.refit()
            workspace.run()
            volume.zero_()
            wp.launch(
                consume,
                len(workspace.full_tetrahedra),
                inputs=[
                    points,
                    workspace.full_tetrahedra,
                    workspace.full_tetrahedra_count,
                    workspace.error,
                    volume,
                ],
            )
        wp.capture_launch(capture.graph)
    dot = tmp_path / "graded-vtm.dot"
    wp.capture_debug_dot_print(capture.graph, str(dot), verbose=True)
    assert all(node not in dot.read_text() for node in ("MEM_ALLOC", "MEM_FREE", "HOST"))
    for p in (0, 1, 2, 3):
        phase.fill_(p)
        wp.capture_launch(capture.graph)
        if p == 2:
            assert workspace.error.numpy()[0] & 4
            assert workspace.full_tetrahedra_count.numpy()[0] == 0
            assert volume.numpy()[0] == 0
        else:
            valid_core(workspace)
            assert workspace.iterations.numpy()[0] == 2
            np.testing.assert_allclose(volume.numpy(), workspace.volume.numpy(), rtol=1e-12)


@pytest.mark.parametrize("scale", [1e-12, 1e12])
def test_field_rescales_with_geometry_without_changing_relative_grading(scale):
    rng = np.random.default_rng(122)
    positions = (scale * rng.normal(size=(32, 3))).astype(np.float32)
    queries = (scale * rng.normal(size=(32, 3))).astype(np.float32)
    values = (scale * rng.uniform(0.2, 0.4, 32)).astype(np.float32)
    field = warpower.SizingField(
        positions,
        wp.array(values, device="cuda:0"),
        gradation=0.3,
        minimum=scale * 0.1,
        maximum=scale * 2,
    )
    query = warpower.SizingQueryWorkspace(field, queries).run()
    assert field.error.numpy()[0] == query.error.numpy()[0] == 0
    np.testing.assert_allclose(
        query.values.numpy(),
        oracle(positions, values, queries, 0.3, scale * 0.1, scale * 2),
        rtol=8e-8,
    )
