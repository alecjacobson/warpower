"""Bind sanitizer loop frames to independently validated ordinary-loop geometry."""
import hashlib
import json
from pathlib import Path

import numpy as np

w = Path(".work/vtm")
dest = w / "tangential-production-memory-v3-comparison.json"
assert not dest.exists()
commands = json.loads((w / "tangential-production-memory-v3-commands.json").read_text())
assert len(commands) == 10 and all(r["exit_code"] == 0 and r["zero_memcheck_errors"] for r in commands)
proof = w / "tangential-production-loop-v2-independent.json"
assert json.loads(proof.read_text())["complete"]
baseline = w / "tangential-production-loop-v2"
ordinary = json.loads((baseline / "result.json").read_text())


def oriented_rows(data):
    # Sorted IDs plus permutation parity preserve simplex orientation while
    # disregarding row order. The reference has already been certified.
    if not len(data):
        return np.empty((0, data.shape[1]+1), dtype=np.int64)
    parity = np.zeros(len(data), dtype=np.int64)
    for i in range(data.shape[1]):
        for j in range(i+1, data.shape[1]):
            parity ^= data[:, i] > data[:, j]
    result = np.column_stack([np.sort(data, axis=1), parity])
    return result[np.lexsort(result[:, ::-1].T)]


report = dict(scope=__doc__, ordinary_proof_sha256=hashlib.sha256(proof.read_bytes()).hexdigest(), modes={}, complete=False)
for mode in ["padded", "lifetime"]:
    candidate = w / ("tangential-production-memory-v3-loop-" + mode + "-frames")
    run = json.loads((candidate / "result.json").read_text())
    assert run["stable_pointers"]
    assert run["production_source_sha256"] == ordinary["production_source_sha256"]
    assert run["source_sha256"] == ordinary["source_sha256"]
    np.testing.assert_array_equal(run["rows"], ordinary["rows"])
    np.testing.assert_allclose(run["values"], ordinary["values"], rtol=1e-11, atol=0)
    frames = {}
    for number in [0, 1, 4]:
        before, after = baseline / f"frame{number}.npz", candidate / f"frame{number}.npz"
        fields = {}
        with np.load(before) as a, np.load(after) as b:
            assert a.files == b.files
            for key in a.files:
                x, y = a[key], b[key]
                assert x.shape == y.shape and x.dtype == y.dtype
                if key in ["full_tetrahedra", "baseline_full_tetrahedra", "tetrahedra", "boundary"]:
                    equal = np.array_equal(oriented_rows(x), oriented_rows(y))
                    comparison = "oriented simplex rows, order independent"
                else:
                    equal = x.tobytes() == y.tobytes()
                    comparison = "bitwise"
                fields[key] = dict(equal=bool(equal), comparison=comparison)
                assert equal, (mode, number, key)
        frames[number] = dict(reference_sha256=hashlib.sha256(before.read_bytes()).hexdigest(), measured_sha256=hashlib.sha256(after.read_bytes()).hexdigest(), fields=fields)
    report["modes"][mode] = dict(rows_equal=True, consumed_values_match=True, frames=frames)
report["complete"] = True
dest.write_text(json.dumps(report, indent=2) + "\n")
print("Both sanitizer loops match independently validated ordinary-loop geometry, including orientation.")
