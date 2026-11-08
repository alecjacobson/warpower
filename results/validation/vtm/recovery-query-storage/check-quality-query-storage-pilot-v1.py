"""Independent full-mesh validation and cost accounting for the constructor-only probe."""
import hashlib
import json
from pathlib import Path

import numpy as np

w = Path(".work/vtm")
helper = w / "check-surface-bound-reuse-final-v2.py"
exec(compile(helper.read_text().split("out=w/'surface-bound-reuse-final'")[0], str(helper), "exec"))
out = w / "quality-query-storage-pilot-v1"
dest = w / "quality-query-storage-pilot-v1-independent.json"
assert not dest.exists()
commands = json.loads((out / "commands.json").read_text())
assert len(commands) == 8 and all(r["exit_code"] == 0 and r["quota_clean"] for r in commands)
report = dict(scope=__doc__, cases={}, complete=False)


def work(d):
    return {k: d[k] for k in ["counts", "rounds", "topology", "quality_rejections", "refinement_rejections"]} | dict(steps=d.get("steps", d.get("optimizer_steps")), moves=d.get("moves", d.get("quality_moves")))


for name in ["thin-hand-seed7", "armadillo-graded4096", "armadillo-graded8192-recovery", "armadillo-uniform8192-recovery"]:
    paths = [out / (name + "-" + phase) for phase in ["before", "after"]]
    results = [json.loads((p / "result.json").read_text()) for p in paths]
    observers = [json.loads((out / (name + "-" + phase + "-recovery.json")).read_text()) for phase in ["before", "after"]]
    guards = [json.loads((out / (name + "-" + phase + "-storage.json")).read_text()) for phase in ["before", "after"]]
    for d in results:
        assert d["valid"] == 1 and d["error"] == 0
        quota(d)
    params = lambda d: {k: v for k, v in d["parameters"].items() if k != "output"}
    assert params(results[0]) == params(results[1])
    assert work(results[0]) == work(results[1])
    row = dict(before_ms=results[0]["median_additional_replay_ms"], after_ms=results[1]["median_additional_replay_ms"], parameters_equal=True, work_equal=True, independent={}, guards=guards, source_sha256=hashlib.sha256((w / "quality_query_storage_v1.py").read_bytes()).hexdigest())
    for phase, path in zip(["before", "after"], paths):
        row["independent"][phase] = validate(path / "mesh.npz", path / "mesh.vtu.gz")
    with np.load(paths[0] / "mesh.npz") as a, np.load(paths[1] / "mesh.npz") as b:
        equal = compare(a, b)
        assert all(equal[k] for k in ["roles_equal", "full_tets_equal", "material_tets_equal", "oriented_boundary_equal"]), equal
        equal["changed_scalar_coordinates"] = int(np.count_nonzero(a["points"].view(np.uint32) != b["points"].view(np.uint32)))
        equal["relative_coordinate_difference"] = equal["maximum_coordinate_difference"] / max(1.0, float(np.linalg.norm(np.ptp(a["points"].astype(float), axis=0))))
        assert equal["relative_coordinate_difference"] < 1e-9, equal
        for key in ["input_points", "input_faces"]:
            assert a[key].tobytes() == b[key].tobytes()
        def sorted_quality(data):
            cells = np.sort(data["tetrahedra"], axis=1)
            return data["radius_ratio"][np.lexsort(cells.T[::-1])]
        qa, qb = sorted_quality(a), sorted_quality(b)
        equal["maximum_radius_ratio_difference"] = float(np.max(np.abs(qa-qb)))
        equal["minimum_radius_ratio_before"] = float(qa.min())
        equal["minimum_radius_ratio_after"] = float(qb.min())
        assert equal["maximum_radius_ratio_difference"] < 1e-7 and abs(qa.min()-qb.min()) < 1e-12, equal
        row["comparison"] = equal
    for field in ["chosen", "invoked", "accepted", "attempts", "error", "status", "rejection_reason", "point_count", "full_count", "retained_count", "topology"]:
        assert observers[0]["recovery"][field] == observers[1]["recovery"][field], field
    row["constructor"] = {k: {phase: d[k] for phase, d in zip(["before", "after"], observers)} for k in ["constructor_seconds", "persistent_workspace_pool_bytes"]}
    row["saved_workspace_bytes"] = observers[0]["persistent_workspace_pool_bytes"] - observers[1]["persistent_workspace_pool_bytes"]
    row["predicted_payload_bytes"] = guards[1]["predicted_removed_payload_bytes"]
    assert row["saved_workspace_bytes"] > 0
    row["input_sha256"] = {phase: hashlib.sha256((path / "mesh.npz").read_bytes()).hexdigest() for phase, path in zip(["before", "after"], paths)}
    report["cases"][name] = row
    dest.write_text(json.dumps(report, indent=2) + "\n")
    print(name, row["saved_workspace_bytes"], row["before_ms"], row["after_ms"], equal, flush=True)
report["complete"] = True
dest.write_text(json.dumps(report, indent=2) + "\n")
