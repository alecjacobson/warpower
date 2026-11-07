"""Validate the optional offline CGAL distance oracle against independent geometry."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from benchmarks.mesh_quality import measure
from benchmarks.surface_distance import NativeSurfaceDistance
from tests.test_domain import brute_distances


def audit(binary, scratch):
    oracle = NativeSurfaceDistance(binary, scratch)
    rng = np.random.default_rng(140)
    p = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    known = np.array(
        [[0.25, 0.25, 2], [2, 0, 0], [-1, -1, 1], [0.8, 0.8, 0], [0, 0.5, 0], [-0.25, 0.5, -1]]
    )
    q = np.vstack([known, rng.uniform(-2, 2, (100, 3))])
    random_points = rng.uniform(-2, 2, (27, 3))
    random_faces = np.arange(27, dtype=np.int32).reshape(-1, 3)
    random_queries = rng.uniform(-3, 3, (200, 3))
    rows = []
    for name, points, faces, queries in [
        ("triangle-projections", p, np.array([[0, 1, 2]], np.int32), q),
        ("multiple-triangles", random_points, random_faces, random_queries),
    ]:
        for scale in [2.0**-60, 1.0, 2.0**60]:
            target = (points + [3, -4, 2]) * scale
            samples = (queries + [3, -4, 2]) * scale
            actual = oracle(target, faces, samples)
            expected = brute_distances(target, faces, samples)
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-13 * scale)
            if name == "triangle-projections":
                np.testing.assert_allclose(
                    (actual[:6] / scale) ** 2, [4, 1, 3, 0.18, 0, 1.0625], rtol=1e-12, atol=1e-13
                )
            rows.append(
                dict(
                    case=name,
                    scale=scale,
                    queries=len(queries),
                    maximum_error_divided_by_scale=float(np.max(np.abs(actual - expected)) / scale),
                )
            )
    np.testing.assert_array_equal(oracle(p, [[0, 1, 2]], np.empty((0, 3))), [])
    rejections = {}
    for name, points, faces, queries, exception in [
        ("index", p, [[0, 1, 9]], q, ValueError),
        ("noninteger-index", p, [[0.0, 1.0, 2.0]], q, ValueError),
        ("nonfinite-point", p * np.nan, [[0, 1, 2]], q, ValueError),
        ("nonfinite-query", p, [[0, 1, 2]], q * np.nan, ValueError),
        (
            "degenerate-triangle",
            np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]]),
            [[0, 1, 2]],
            q,
            subprocess.CalledProcessError,
        ),
    ]:
        try:
            oracle(points, faces, queries)
        except exception:
            rejections[name] = exception.__name__
        else:
            raise AssertionError(f"Invalid {name} fixture was accepted")
    # Both distance directions participate in the complete material-quality report.
    points = np.vstack([p, [[0, 0, 1]]])
    faces = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]], np.int32)
    tets = np.array([[0, 1, 2, 3]], np.int32)
    reference = measure(points, tets, points * 1.1, faces)
    native = measure(points, tets, points * 1.1, faces, distance_query=oracle)

    def compare(a, b):
        if isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                if key != "scope":
                    compare(a[key], b[key])
        else:
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-13)

    compare(reference, native)
    return dict(
        scope="Offline validator tests only; no runtime or meshing speed claim.",
        distance_oracle=oracle.metadata,
        comparisons=rows,
        rejected_inputs=rejections,
        two_sided_quality_report_matches_brute_reference=True,
        empty_queries_supported=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.binary, args.scratch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
