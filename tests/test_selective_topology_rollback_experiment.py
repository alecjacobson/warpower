"""Current witnesses, replay reset, pinned sites and fresh rejection contracts."""

import numpy as np
import pytest
import warp as wp

from benchmarks.selective_topology_rollback import block_moved, eligible, expand_star, flag_edges
from warpower._mesh_topology import MeshTopology


@pytest.mark.parametrize(
    "changed,expected",
    [
        ({}, 1),
        ({"before": 4}, 0),
        ({"mode": 0}, 0),
        ({"recovered": 0}, 0),
        ({"reason": 4}, 0),
        ({"mesh_error": 8}, 0),
        ({"boundary_error": 1}, 0),
        ({"extraction_error": 16}, 0),
        ({"extraction_valid": 1}, 0),
        ({"diagnostics": 6}, 0),
        ({"used": 1}, 0),
    ],
)
def test_only_fresh_isolated_edge_failure_can_retry(changed, expected):
    values = dict(
        mode=1,
        recovered=1,
        accepted=0,
        rejections=4,
        before=3,
        reason=3,
        selection_error=16,
        mesh_error=0,
        boundary_error=0,
        extraction_error=0,
        extraction_valid=0,
        diagnostics=2,
        used=0,
    )
    values.update(changed)
    arrays = [wp.array([v], dtype=int, device="cuda:0") for v in values.values()]
    ready = wp.ones(1, dtype=int, device="cuda:0")
    stats = wp.zeros(4, dtype=int, device="cuda:0")
    wp.launch(eligible, 1, inputs=[*arrays, 1, ready, stats])
    assert ready.numpy()[0] == expected


@pytest.mark.parametrize("neighborhood,expected", [(False, [0, 1]), (True, [0, 1, 2, 3])])
def test_current_nonmanifold_star_changed_count_and_pinned_vertices(neighborhood, expected):
    # Two positive tetrahedra meet only at edge (0, 1): four incident boundary
    # triangles. This is a nonmanifold material subcomplex, although each tet
    # is valid. Padding contains invalid indices and must never be inspected.
    p = np.array(
        [[0, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [99, 99, 99]],
        dtype=np.float32,
    )
    cells = np.array([[0, 1, 2, 3], [0, 1, 4, 5]], dtype=np.int32)
    for row in cells:
        if np.linalg.det((p[row[1:]] - p[row[0]]).T) < 0:
            row[2], row[3] = row[3], row[2]
    with wp.ScopedDevice("cuda:0"):
        original, points = wp.array(p, dtype=wp.vec3), wp.array(p + 0.1, dtype=wp.vec3)
        count = wp.array([2], dtype=int)
        active = wp.array([6], dtype=int)
        tets = wp.array(np.vstack([cells, np.full((3, 4), -999, dtype=np.int32)]), dtype=int)
        topology = MeshTopology(points, tets, count)
        bad, star, blocked = [wp.zeros(len(p), dtype=int) for _ in range(3)]
        any_blocked = wp.zeros(1, dtype=int)
        statistics = wp.zeros(4, dtype=int)
        roles = wp.array([1, 1, 0, 0, 3, 0, 0], dtype=int)
        mask = wp.array([1, 1, 1, 1, 1, 0, 1], dtype=int)
        wp.load_module(module="benchmarks.selective_topology_rollback", device="cuda:0")
        with wp.ScopedCapture() as capture:
            for a in [bad, star, blocked, any_blocked, statistics]:
                a.zero_()
            topology.run()
            wp.launch(
                flag_edges,
                len(topology.boundary_indices),
                inputs=[
                    topology.boundary_indices,
                    topology.boundary_count,
                    topology._edge_slots,
                    topology._edge_counts,
                    topology._edge_balance,
                    bad,
                ],
            )
            wp.launch(expand_star, len(tets), inputs=[tets, count, bad, star])
            wp.launch(
                block_moved,
                len(p),
                inputs=[
                    active,
                    star if neighborhood else bad,
                    roles,
                    mask,
                    original,
                    points,
                    blocked,
                    any_blocked,
                    statistics,
                ],
            )
        for n in [2, 1, 2]:
            count.assign(np.array([n], dtype=np.int32))
            wp.capture_launch(capture.graph)
            np.testing.assert_array_equal(
                np.flatnonzero(blocked.numpy()), expected if n == 2 else []
            )
            np.testing.assert_array_equal(np.flatnonzero(bad.numpy()), [0, 1] if n == 2 else [])
            assert statistics.numpy()[1] == (len(expected) if n == 2 else 0)
