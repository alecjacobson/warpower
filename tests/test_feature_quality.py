import numpy as np

from benchmarks.feature_quality import feature_graph, measure, project_chain
from examples.vtm_inputs import counterbored_plate, cube


def test_feature_audit_excludes_planar_diagonals_and_preserves_cube_corners():
    p, f = cube()
    edges, corners, chains = feature_graph(p, f)
    assert len(edges) == len(chains) == 12
    assert len(corners) == 8
    result = measure(p, f, np.full(len(p), 3), np.arange(len(p)), p, f, samples_per_edge=2)
    assert all(row["exact_coordinate_match"] for row in result["corners"])
    assert result["feature_to_boundary"]["maximum_sample_distance"] == 0
    assert all(row["pairs_connected_by_boundary_edge"] == 1 for row in result["chains"])


def test_closed_feature_rims_and_open_edges_cover_every_input_segment():
    p, f = counterbored_plate(16)
    edges, corners, chains = feature_graph(p, f)
    assert len(chains) == 16
    assert sum(chain[0] == chain[-1] for chain in chains) == 4
    assert sum(len(chain) - 1 for chain in chains) == len(edges)
    roles = np.full(len(p), 2)
    roles[corners] = 3
    result = measure(p, f, roles, np.arange(len(p)), p, f, samples_per_edge=2)
    assert result["crease_sites_on_boundary"] == len(p) - 8
    assert result["feature_to_boundary"]["maximum_sample_distance"] < 1e-14
    assert result["crease_site_distance_percentiles"]["100"] < 1e-14
    assert sum(row["assigned_boundary_crease_sites"] for row in result["chains"]) == len(p) - 8
    assert all(not row["missing_boundary_pairs"] for row in result["chains"])
    assert sum(row["consecutive_pairs"] for row in result["chains"]) == len(edges)


def test_audit_reports_off_curve_sites_lost_corner_and_missing_boundary_edge():
    ip, f = counterbored_plate(16)
    _, corners, _ = feature_graph(ip, f)
    p = ip.astype(float)
    roles = np.full(len(p), 2)
    roles[corners] = 3
    # The lower bore rim has 20 vertices and begins at vertex 40. Moving one
    # vertex into the bore wall measures deviation from the feature, not merely
    # the distance to some point on the input surface (which remains zero).
    p[40, 2] += 0.05
    p[corners[0], 2] += 0.01
    # Remove both faces incident to rim edge (45,46); preserve its endpoints.
    boundary = f[~(np.any(f == 45, axis=1) & np.any(f == 46, axis=1))]
    result = measure(p, boundary, roles, np.arange(len(p)), ip, f, samples_per_edge=2)
    assert np.isclose(result["crease_site_distance_percentiles"]["100"], 0.05)
    assert sum(row["exact_coordinate_match"] for row in result["corners"]) == 7
    assert result["feature_to_boundary"]["rms"] > 0
    assert any(
        set(pair) == {45, 46} for row in result["chains"] for pair in row["missing_boundary_pairs"]
    )


def test_projection_clamps_to_endpoints_and_no_features_has_explicit_empty_report():
    d, s, length = project_chain(
        np.array([[-1, 0, 0], [0.5, 3, 0], [3, 0, 0]]),
        np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]]),
    )
    np.testing.assert_allclose(d, [1, 3, 1])
    np.testing.assert_allclose(s, [0, 0.5, 2])
    assert length == 2
    p, f = cube()
    result = measure(p, f, np.ones(len(p)), np.arange(len(p)), p, f, feature_angle=179)
    assert result["input_feature_edges"] == result["input_corners"] == 0
    assert result["feature_to_boundary"] is None
    assert result["crease_site_distance_percentiles"] is None
    assert result["chains"] == []
