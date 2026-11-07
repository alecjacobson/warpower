"""Exact expression-tree bounds must match the current executable filters."""

import json
import subprocess
import sys
from pathlib import Path


def test_bounded_predicate_proof(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "bound.json"
    subprocess.run(
        [
            sys.executable,
            str(root / "benchmarks/bounded_predicate_proof.py"),
            "--output",
            str(output),
        ],
        cwd=root,
        check=True,
    )
    proof = json.loads(output.read_text())
    assert proof["formula_range_constants_verified"]
    for row in proof["filters"].values():
        assert row["coefficient_inequality"] and row["floor_inequality"]
        assert row["permanent_matches_absolute_monomials"]
