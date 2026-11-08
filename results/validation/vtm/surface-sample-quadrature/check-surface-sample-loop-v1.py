"""Verify current parallel-quadrature loop output against independently certified reference geometry."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
assert not args.output.exists()
w = Path(".work/vtm")
helper = w / "check-tangential-production-memory-v3.py"
function = next(n for n in ast.parse(helper.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "oriented_rows")
exec(compile(ast.Module(body=[function], type_ignores=[]), str(helper), "exec"))
source = json.loads((w / "surface-sample-production-source-v2.json").read_text())
proof = w / "tangential-production-loop-v2-independent.json"
assert json.loads(proof.read_text())["complete"]
baseline = w / "tangential-production-loop-v2"
before = json.loads((baseline / "result.json").read_text())
after = json.loads((args.input / "result.json").read_text())
for name, digest in after["production_source_sha256"].items():
    assert source["source_sha256"][name] == digest
assert after["source_sha256"] == before["source_sha256"]
assert after["stable_pointers"]
np.testing.assert_array_equal(after["rows"], before["rows"])
np.testing.assert_allclose(after["values"], before["values"], rtol=1e-11, atol=0)
report = dict(scope=__doc__, runtime=source["runtime"], input=str(args.input), reference_proof_sha256=hashlib.sha256(proof.read_bytes()).hexdigest(), oriented_comparison_source_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(), production_source_sha256=after["production_source_sha256"], rows_equal=True, stable_pointers=True, frames={}, complete=False)
for number in [0, 1, 4]:
    a, b = baseline / f"frame{number}.npz", args.input / f"frame{number}.npz"
    fields = {}
    with np.load(a) as old, np.load(b) as new:
        assert old.files == new.files
        for name in old.files:
            x, y = old[name], new[name]
            assert x.shape == y.shape and x.dtype == y.dtype
            if name in ["full_tetrahedra", "baseline_full_tetrahedra", "tetrahedra", "boundary"]:
                equal = np.array_equal(oriented_rows(x), oriented_rows(y))
                kind = "oriented simplex connectivity, independent of row order"
            else:
                equal = x.tobytes() == y.tobytes()
                kind = "bitwise"
            assert equal, (number, name)
            fields[name] = dict(equal=True, comparison=kind)
    report["frames"][number] = dict(fields=fields, reference_sha256=hashlib.sha256(a.read_bytes()).hexdigest(), measured_sha256=hashlib.sha256(b.read_bytes()).hexdigest())
report["complete"] = True
args.output.write_text(json.dumps(report, indent=2) + "\n")
print("Current source bindings, diagnostic resets, exhaustion and both success frames match the certified reference.")
