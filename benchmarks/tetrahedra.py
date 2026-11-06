"""Compare the complete optional mesh with independent SciPy references."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp
from scipy.spatial import ConvexHull, Delaunay

import warpower


def canonical(tets):
    rows = np.sort(tets, axis=1)
    return rows[np.lexsort(rows.T[::-1])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000, 100000])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    wp.set_mempool_release_threshold("cuda:0", 32 * 1024**3)
    records = []
    for n in args.sizes:
        for weighted in (False, True):
            rng = np.random.default_rng(args.seed)
            points = rng.uniform(-1, 1, (n, 3)).astype(np.float32)
            weight_bound = 0.0002 if n >= 100000 else 0.002
            weights = (
                rng.uniform(-weight_bound, weight_bound, n).astype(np.float32) if weighted else None
            )
            start = time.perf_counter()
            result = (
                warpower.power_diagram(points, weights, return_tetrahedra=True)
                if weighted
                else warpower.voronoi_diagram(points, return_tetrahedra=True)
            )
            elapsed = time.perf_counter() - start
            actual = canonical(result.tetrahedra.numpy())
            if weighted:
                p64 = points.astype(np.float64)
                lifted = np.column_stack(
                    (p64, (p64 * p64).sum(axis=1) - weights.astype(np.float64))
                )
                hull = ConvexHull(lifted)
                expected = canonical(hull.simplices[hull.equations[:, 3] < 0])
            else:
                expected = canonical(Delaunay(points).simplices)
            assert np.array_equal(actual, expected), (n, weighted, actual.shape, expected.shape)
            record = {
                "n": n,
                "weighted": weighted,
                "tetrahedra": len(actual),
                "seconds_including_first_JIT": elapsed,
                "expansions": result.diagnostics["tetrahedra_expansions"],
                "missing": 0,
                "extra": 0,
            }
            records.append(record)
            print(json.dumps(record), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"seed": args.seed, "records": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
