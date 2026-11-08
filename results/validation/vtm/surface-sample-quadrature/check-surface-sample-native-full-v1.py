"""Bind native-profiled output to independently validated current timed geometry."""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np

w = Path(".work/vtm")
dest = w / "surface-sample-native-full-v1-comparison.json"
assert not dest.exists()
proof = w / "surface-sample-final-v1-independent.json"
independent = json.loads(proof.read_text())
assert independent["complete"]
name = "armadillo-uniform8192-recovery"
assert independent["cases"][name]["independent"]["after"]["full_hull"]["valid"]
baseline = w / "surface-sample-final-v1" / (name + "-after")
measured = w / "surface-sample-uniform-native-v1.npz"
native = json.loads((w / "surface-sample-uniform-native-v1.json").read_text())
source = json.loads((w / "surface-sample-production-source-v2.json").read_text())
assert native["runtime"] == source["runtime"]
for path, digest in native["source_sha256"].items():
    if path.startswith("warpower/"):
        assert source["source_sha256"][path] == digest
    else:
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
result = json.loads((baseline / "result.json").read_text())
for field in ["counts", "rounds", "steps", "moves", "topology"]:
    assert native["work"][field] == result[field]
helper = w / "check-tangential-production-memory-v3.py"
function = next(n for n in ast.parse(helper.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "oriented_rows")
exec(compile(ast.Module(body=[function], type_ignores=[]), str(helper), "exec"))
fields = {}
with np.load(baseline / "mesh.npz") as a, np.load(measured) as b:
    for name in ["points", "roles", "input_points", "input_faces"]:
        assert a[name].shape == b[name].shape and a[name].dtype == b[name].dtype
        assert a[name].tobytes() == b[name].tobytes(), name
        fields[name] = "bitwise equal"
    for name in ["full_tetrahedra", "tetrahedra", "boundary"]:
        assert np.array_equal(oriented_rows(a[name]), oriented_rows(b[name])), name
        fields[name] = "equal oriented simplex connectivity"
    def quality(data):
        cells = np.sort(data["tetrahedra"], axis=1)
        return data["radius_ratio"][np.lexsort(cells.T[::-1])]
    x, y = quality(a), quality(b)
    delta = float(np.max(np.abs(x-y)))
    assert delta < 1e-7 and abs(float(x.min()-y.min())) < 1e-12
report = dict(scope=__doc__, runtime=source["runtime"], reference_proof_sha256=hashlib.sha256(proof.read_bytes()).hexdigest(), reference_mesh_sha256=hashlib.sha256((baseline / "mesh.npz").read_bytes()).hexdigest(), measured_mesh_sha256=hashlib.sha256(measured.read_bytes()).hexdigest(), comparison=fields, maximum_stored_quality_difference=delta, scheduler_work_equal=True, complete=True)
dest.write_text(json.dumps(report, indent=2) + "\n")
print("Native full-solve geometry, orientation and scheduler work match the independently validated timed mesh.")
