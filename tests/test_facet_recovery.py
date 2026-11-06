import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import warp as wp

from tests.test_delaunay_recovery import valid
from tests.test_domain import torus
from tests.test_facet_completion import canonical
from warpower import DelaunayWorkspace


def hand():
    path = Path(__file__).parent / "data/hand-refined384.npz"
    return np.load(path)["points"], json.loads(path.with_suffix(".json").read_text())


def check_hand(workspace, points, expected):
    assert workspace.error.numpy()[0] == 0
    cells = workspace.tetrahedra.numpy()[: workspace.tetrahedra_count.numpy()[0]]
    assert len(cells) == expected["expected_tetrahedra"]
    assert (
        hashlib.sha256(canonical(cells).tobytes()).hexdigest()
        == expected["expected_canonical_connectivity_sha256"]
    )
    np.testing.assert_array_equal(workspace.points.numpy(), points)
    assert workspace.facet_recovered.numpy()[0] == 1
    assert workspace.symbolic_attempted.numpy()[0] == 0
    assert workspace.facet_error.numpy()[0] == 0
    assert workspace.facet_rounds.numpy()[0] > 0


def test_refined_hand_completes_without_symbolic_cells_and_reuses_certified_strategy():
    points, expected = hand()
    w = DelaunayWorkspace(
        points, facet_recovery=True, symbolic_recovery=False, max_attempts=1
    ).run()
    check_hand(w, points, expected)
    assert w.base_attempted.numpy()[0] == 1
    assert w._preferred.numpy()[0] == 3
    assert w.facet_attempted.numpy()[0] == 1
    assert w.attempts.numpy()[0] == 1
    w.run()
    check_hand(w, points, expected)
    assert w.base_attempted.numpy()[0] == 0
    assert w.recovery_first.numpy()[0] == 1
    assert w._preferred_level.numpy()[0] == 0
    w.points[0:1].fill_(wp.vec3(float("nan")))
    w.run()
    assert w.error.numpy()[0] == 1
    assert w.tetrahedra_count.numpy()[0] == 0
    assert w.facet_attempted.numpy()[0] == w.facet_recovered.numpy()[0] == 0
    w.points.assign(points)
    w.run()
    check_hand(w, points, expected)


def test_structured_torus_uses_filtered_facets_and_matches_complete_symbolic_mesh():
    points, _ = torus(16, 8)
    reference = DelaunayWorkspace(points).run()
    expected = valid(reference, points)
    w = DelaunayWorkspace(
        points, facet_recovery=True, symbolic_recovery=False, max_attempts=1
    ).run()
    actual = valid(w, points)
    np.testing.assert_array_equal(canonical(actual), canonical(expected))
    assert w.facet_recovered.numpy()[0] == 1
    assert w.symbolic_attempted.numpy()[0] == 0

    # Automatic sizing consumes the retained face index. Check its incidence
    # against an independent enumeration of the published tetrahedra.
    cert = w._certificate
    faces = [tuple(sorted(np.delete(tet, k))) for tet in actual for k in range(4)]
    multiplicities = Counter(faces)
    table, occurrences = cert._table.numpy(), cert._occurrences.numpy()
    indexed = {}
    for slot in np.flatnonzero(table >= 0):
        face = int(table[slot])
        key = tuple(sorted(np.delete(actual[face // 4], face % 4)))
        indexed[key] = int(occurrences[slot])
    assert indexed == multiplicities
    # Its saved candidate input must also describe the new result.
    with wp.ScopedCapture() as capture:
        cert.run()
    wp.capture_launch(capture.graph)
    assert cert.error.numpy()[0] == 0
    np.testing.assert_array_equal(canonical(valid(w, points)), canonical(expected))


def test_exhausted_completion_returns_zero_count_then_accepts_changed_input():
    points, expected = hand()
    w = DelaunayWorkspace(
        points, facet_recovery=True, facet_max_rounds=0, symbolic_recovery=False, max_attempts=1
    ).run()
    assert w.error.numpy()[0] == 2
    assert w.tetrahedra_count.numpy()[0] == 0
    assert w.facet_error.numpy()[0] & 16
    assert w.facet_recovered.numpy()[0] == 0
    changed = np.random.default_rng(91).normal(size=points.shape).astype(np.float32)
    w.points.assign(changed)
    w.run()
    valid(w, changed)
    assert w.facet_attempted.numpy()[0] == w.facet_error.numpy()[0] == 0


@pytest.mark.parametrize("budget", [-1, 65])
def test_invalid_facet_budget_rejected_before_allocation(budget):
    with pytest.raises(ValueError, match="facet_max_rounds"):
        DelaunayWorkspace(np.zeros((4, 3), np.float32), facet_max_rounds=budget)


@wp.kernel
def discard_candidates(error: wp.array[int], count: wp.array[int], raw_count: wp.array[int]):
    error[0] = 2
    count[0] = 0
    raw_count[0] = 0


def test_failed_cached_completion_falls_back_to_current_base_mesh():
    points, expected = hand()
    w = DelaunayWorkspace(
        points, facet_recovery=True, symbolic_recovery=False, max_attempts=1
    ).run()
    check_hand(w, points, expected)
    changed = np.random.default_rng(28).normal(size=points.shape).astype(np.float32)
    w.points.assign(changed)
    real_run = w._auxiliary.run

    def fail_run(stream=None):
        real_run(stream)
        wp.launch(
            discard_candidates,
            1,
            inputs=[
                w._auxiliary.error,
                w._auxiliary.tetrahedra_count,
                w._auxiliary._candidate_count,
            ],
        )

    w._auxiliary.run = fail_run
    wp.load_module(module=__name__, device=w.device)
    with wp.ScopedCapture() as capture:
        w.run()
    wp.capture_launch(capture.graph)
    valid(w, changed)
    assert w.recovery_first.numpy()[0] == w.base_attempted.numpy()[0] == 1
    assert w.facet_attempted.numpy()[0] == 1
    assert w.facet_recovered.numpy()[0] == 0
    assert w.facet_error.numpy()[0] != 0
    assert w._preferred.numpy()[0] == 0


def test_integrated_resident_loop_changes_inputs_invalidates_and_recovers():
    from benchmarks.facet_recovery import audit

    audit()
