from itertools import product

import numpy as np
import pytest
import warp as wp

from tests.test_delaunay_recovery import consume, exact_empty, produce, valid
from tests.test_domain import torus
from tests.test_tetrahedra import reference, tet_set
from warpower import DelaunayWorkspace


def test_structured_torus_uses_original_coordinate_symbolic_recovery():
    points, _ = torus(16, 8)
    w = DelaunayWorkspace(points).run()
    tets = valid(w, points)
    assert w.base_error.numpy()[0] == 2
    assert w.attempts.numpy()[0] == w.max_attempts
    assert w.symbolic_attempted.numpy()[0] == 1
    assert w.symbolic_recovered.numpy()[0] == 1
    assert w.symbolic_error.numpy()[0] == 0
    assert w.symbolic_expansions.numpy()[0] == 0
    assert len(tets) == 519
    # SciPy/CGAL can select different diagonals for exact cospherical sites.
    # Check every tetrahedron with a separate rational determinant oracle.
    exact_empty(points, tets)
    w.run()
    assert tet_set(valid(w, points)) == tet_set(tets)
    assert w.recovery_first.numpy()[0] == 1
    assert w.base_attempted.numpy()[0] == 0
    assert w.attempts.numpy()[0] == 0
    assert w.symbolic_recovered.numpy()[0] == 1


def test_symbolic_strategy_can_be_disabled():
    points, _ = torus(16, 8)
    w = DelaunayWorkspace(points, symbolic_recovery=False).run()
    assert w.error.numpy()[0] == 2
    assert w.tetrahedra_count.numpy()[0] == 0
    assert w.symbolic_attempted.numpy()[0] == 0
    assert w.symbolic_error.numpy()[0] == 0


@pytest.mark.parametrize("case", ["cube", "grid", "random", "duplicates"])
def test_symbolic_constructor_alone_matches_independent_geometry(case):
    if case in ("cube", "grid"):
        points = np.array(list(product(range(2 if case == "cube" else 3), repeat=3)), np.float32)
    else:
        points = np.random.default_rng(918).uniform(-1, 1, (20, 3)).astype(np.float32)
        if case == "duplicates":
            points[::3] = points[1]
    w = DelaunayWorkspace(points)
    w._preferred.fill_(2)
    w.run()
    tets = valid(w, points)
    assert w.symbolic_recovered.numpy()[0] == 1
    assert w.base_attempted.numpy()[0] == 0
    assert w.attempts.numpy()[0] == 0
    if case in ("random", "duplicates"):
        assert tet_set(tets) == reference(points)
    exact_empty(points, tets)


@pytest.mark.parametrize("shape", ["torus", "cube"])
def test_symbolic_replay_composes_changing_producer_and_consumer(tmp_path, shape):
    points, _ = torus(16, 8)
    if shape == "cube":
        points = np.array(list(product([-1, 1], repeat=3)), np.float32)
    w = DelaunayWorkspace(points)
    w._preferred.fill_(2)
    source = wp.array(points, dtype=wp.vec3, device=w.device)
    phase = wp.zeros(1, dtype=int, device=w.device)
    total = wp.zeros(1, dtype=int, device=w.device)
    wp.load_module(module="tests.test_delaunay_recovery", device=w.device)
    pointers = (w.points.ptr, w.tetrahedra.ptr, w.tetrahedra_count.ptr)
    with wp.ScopedCapture() as capture:
        wp.launch(produce, len(points), inputs=[source, w.points, phase])
        w.run()
        total.zero_()
        wp.launch(
            consume, len(w.tetrahedra), inputs=[w.tetrahedra, w.tetrahedra_count, w.error, total]
        )
    meshes = []
    for step in [0, 1, -1, 0]:
        phase.fill_(step)
        wp.capture_launch(capture.graph)
        expected = points.copy()
        expected[0, 0] += np.float32(step) * np.float32(0.0001)
        tets = valid(w, expected)
        meshes.append(tet_set(tets))
        assert total.numpy()[0] == tets.sum()
        assert w.symbolic_recovered.numpy()[0] == 1
        assert w.base_attempted.numpy()[0] == 0
        assert w.attempts.numpy()[0] == 0
        assert pointers == (w.points.ptr, w.tetrahedra.ptr, w.tetrahedra_count.ptr)
    assert meshes[0] == meshes[-1]
    assert meshes[1] != meshes[2]
    path = tmp_path / "symbolic.dot"
    wp.capture_debug_dot_print(capture.graph, str(path))
    dot = path.read_text().lower()
    assert "host" not in dot and "mem_alloc" not in dot and "mem_free" not in dot


def test_symbolic_invalid_and_flat_input_then_recovery():
    points = np.array(list(product([-1, 1], repeat=3)), np.float32)
    w = DelaunayWorkspace(points, max_attempts=1)
    w._preferred.fill_(2)
    w.run()
    valid(w, points)
    bad = points.copy()
    bad[0, 0] = np.nan
    w.points.assign(bad)
    w.run()
    assert w.error.numpy()[0] == 1
    assert w.tetrahedra_count.numpy()[0] == 0
    assert w.symbolic_attempted.numpy()[0] == 0
    assert w._preferred.numpy()[0] == 0
    for kind in ["flat", "line", "same"]:
        bad = points.copy()
        bad[:, 2] = 0
        if kind in ("line", "same"):
            bad[:, 1] = 0
        if kind == "same":
            bad[:, 0] = 0
        w.points.assign(bad)
        w._preferred.fill_(2)
        w.run()
        assert w.error.numpy()[0] == 2
        assert w.symbolic_error.numpy()[0] == 1
        assert w.tetrahedra_count.numpy()[0] == 0
    w.points.assign(points)
    w.run()
    valid(w, points)


def test_symbolic_capacity_failure_publishes_zero_count():
    points = np.array(list(product([-1, 1], repeat=3)), np.float32)
    w = DelaunayWorkspace(points, tetrahedra_capacity=1, max_attempts=1)
    w._preferred.fill_(2)
    w.run()
    assert w.error.numpy()[0] == 2
    assert w.symbolic_error.numpy()[0] == 8
    assert w.tetrahedra_count.numpy()[0] == 0
    assert w.symbolic_attempted.numpy()[0] == 1
    assert w.symbolic_recovered.numpy()[0] == 0


def test_symbolic_expansion_has_a_finite_budget_and_reuses_successful_bounds():
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.2, 0.2, 1.0e-4], [0.8, 0.2, 1.05e-4]], np.float32
    )
    w = DelaunayWorkspace(points, symbolic_max_expansions=1)
    w._base.run()  # Build current hierarchy and original coincident ownership.
    s = w._symbolic
    with wp.ScopedCapture() as capture:
        s.run(w._base._hierarchy, w._base._bounds)
    wp.capture_launch(capture.graph)
    assert s.error.numpy()[0] == 32
    assert w.tetrahedra_count.numpy()[0] == 0
    s.max_expansions = 16
    with wp.ScopedCapture() as capture:
        s.run(w._base._hierarchy, w._base._bounds)
    wp.capture_launch(capture.graph)
    assert s.error.numpy()[0] == 0
    assert s.expansions.numpy()[0] > 0
    assert w.certificate_error.numpy()[0] == 0
    tets = w.tetrahedra.numpy()[: w.tetrahedra_count.numpy()[0]]
    assert tet_set(tets) == reference(points)
    exact_empty(points, tets)
    wp.capture_launch(capture.graph)
    assert s.error.numpy()[0] == 0
    assert s.expansions.numpy()[0] == 0


@pytest.mark.parametrize("budget", [0, 33])
def test_invalid_symbolic_expansion_budget_rejected(budget):
    with pytest.raises(ValueError, match="symbolic_max_expansions"):
        DelaunayWorkspace(np.zeros((4, 3), np.float32), symbolic_max_expansions=budget)


def test_failed_preferred_symbolic_strategy_falls_back_to_current_base():
    from tests.test_delaunay_recovery import fail_candidate

    points = np.array(list(product([-1, 1], repeat=3)), np.float32)
    w = DelaunayWorkspace(points)
    w._preferred.fill_(2)
    w.run()
    valid(w, points)
    changed = np.random.default_rng(413).normal(size=points.shape).astype(np.float32)
    w.points.assign(changed)
    real_run = w._symbolic.run

    def failing_run(tree, bounds):
        real_run(tree, bounds)
        wp.launch(fail_candidate, 1, inputs=[w.symbolic_error, w.tetrahedra_count])

    w._symbolic.run = failing_run
    wp.load_module(module="tests.test_delaunay_recovery", device=w.device)
    with wp.ScopedCapture() as capture:
        w.run()
    wp.capture_launch(capture.graph)
    assert tet_set(valid(w, changed)) == reference(changed)
    assert w.recovery_first.numpy()[0] == 1
    assert w.symbolic_attempted.numpy()[0] == 1
    assert w.symbolic_recovered.numpy()[0] == 0
    assert w.base_attempted.numpy()[0] == 1
    assert w.attempts.numpy()[0] == 0
    assert w.recovered.numpy()[0] == 0
    assert w._preferred.numpy()[0] == 0
    wp.capture_launch(capture.graph)
    assert tet_set(valid(w, changed)) == reference(changed)
    assert w.recovery_first.numpy()[0] == 0
    assert w.symbolic_attempted.numpy()[0] == 0
