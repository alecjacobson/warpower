"""Audit disagreeing CSR rows against all input halfspaces in double precision.

Run from the repository checkout with the dev/bench dependencies installed.
This audits disagreement rows, not all rows in the full diagram.
"""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import scipy

from benchmarks.compare import edges
from benchmarks.datasets import generate
from tests.oracle import reference_cells

_inputs = None


def initialize(points, weights, a, ao, b, bo):
    global _inputs
    _inputs = points, weights, a, ao, b, bo


def audit_cell(index):
    p, w, a, ao, b, bo = _inputs
    i = int(index)
    truth = reference_cells(p, w, indices=[i])[0][0]
    ra = set(map(int, a[ao[i] : ao[i + 1]]))
    rb = set(map(int, b[bo[i] : bo[i + 1]]))
    return {
        "site": i,
        "oracle_degree": len(truth),
        "a_extra": sorted(ra - truth),
        "a_missing": sorted(truth - ra),
        "b_extra": sorted(rb - truth),
        "b_missing": sorted(truth - rb),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("--a-label", default="a")
    parser.add_argument("--b-label", default="b")
    parser.add_argument("--distribution", default="uniform")
    parser.add_argument("--n", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weighted", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    points, weights = generate(args.n, args.distribution, args.seed, args.weighted)
    a, b = np.load(args.a), np.load(args.b)
    if len(a["status"]) != args.n or len(b["status"]) != args.n:
        raise ValueError("diagram sizes do not match the generated point set")
    ea, _ = edges(a)
    eb, _ = edges(b)
    affected = np.unique(np.setxor1d(ea, eb, assume_unique=True) >> np.uint64(32))
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=initialize,
        initargs=(points, weights, a["adjacency"], a["offsets"], b["adjacency"], b["offsets"]),
    ) as pool:
        cells = list(pool.map(audit_cell, affected))
    summary = {
        key: sum(len(c[key]) for c in cells)
        for key in ("a_extra", "a_missing", "b_extra", "b_missing")
    }
    record = {
        "a": args.a_label,
        "b": args.b_label,
        "n": args.n,
        "distribution": args.distribution,
        "seed": args.seed,
        "weighted": args.weighted,
        "scipy": scipy.__version__,
        "points_sha256": hashlib.sha256(points.tobytes()).hexdigest(),
        "weights_sha256": hashlib.sha256(weights.tobytes()).hexdigest()
        if weights is not None
        else None,
        "oracle": "float64 all-site halfspaces, LP interior, Qhull dual vertices",
        "scope": "only rows where the two diagrams disagree; shared rows are not audited",
        "audited_rows": len(cells),
        "summary": summary,
        "cells": cells,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"audited_rows": len(cells), **summary}))


if __name__ == "__main__":
    main()
