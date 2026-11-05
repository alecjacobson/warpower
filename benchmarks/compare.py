"""Compare unordered CSR graphs saved by benchmarks.run (neither is an oracle)."""

import argparse
import json
from pathlib import Path

import numpy as np


def edges(data):
    adjacency, offsets = data["adjacency"], data["offsets"]
    n = len(offsets) - 1
    if offsets[0] != 0 or offsets[-1] != len(adjacency) or np.any(np.diff(offsets) < 0):
        raise ValueError("invalid CSR offsets")
    if np.any(adjacency < 0) or np.any(adjacency >= n):
        raise ValueError("invalid neighbor index")
    rows = np.repeat(np.arange(n, dtype=np.uint64), np.diff(offsets))
    packed = (rows << np.uint64(32)) | adjacency.astype(np.uint64)
    unique = np.unique(packed)
    return unique, len(packed) - len(unique)


def unpack(values):
    return [[int(v) >> 32, int(v) & 0xFFFFFFFF] for v in values]


def compare(a, b):
    if len(a["status"]) != len(b["status"]):
        raise ValueError("diagrams have different numbers of sites")
    ea, da = edges(a)
    eb, db = edges(b)
    only_a = np.setdiff1d(ea, eb, assume_unique=True)
    only_b = np.setdiff1d(eb, ea, assume_unique=True)
    return {
        "n": len(a["status"]),
        "a_edges": len(ea),
        "b_edges": len(eb),
        "a_duplicate_edges": da,
        "b_duplicate_edges": db,
        "shared_edges": len(ea) - len(only_a),
        "only_a_count": len(only_a),
        "only_b_count": len(only_b),
        "only_a_examples": unpack(only_a[:16]),
        "only_b_examples": unpack(only_b[:16]),
        "status_disagreements": int(np.count_nonzero(a["status"] != b["status"])),
        "a_failed_cells": int(np.count_nonzero(~np.isin(a["status"], [0, 3]))),
        "b_failed_cells": int(np.count_nonzero(~np.isin(b["status"], [0, 3]))),
        "interpretation": "unordered adjacency comparison; neither implementation is ground truth",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = compare(np.load(args.a), np.load(args.b))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
