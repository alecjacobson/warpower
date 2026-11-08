"""Bind the hand hull difference to independently validated unchanged-baseline variation."""
import hashlib
import json
from pathlib import Path

import numpy as np

w = Path(".work/vtm")
out = w / "quality-query-storage-final-v1"
pilot = w / "quality-query-storage-pilot-v1"
dest = w / "quality-query-storage-hand-variation-v2.json"
assert not dest.exists()
name = "thin-hand-seed7"
paths = {"other_baseline": pilot / (name+"-before") / "mesh.npz", "before": out / (name+"-before") / "mesh.npz", "after": out / (name+"-after") / "mesh.npz"}


def cells(data, key):
    return {tuple(map(int, row)) for row in np.sort(data[key], axis=1)}


def work(d):
    return {k: d[k] for k in ["counts", "rounds", "topology", "quality_rejections", "refinement_rejections", "steps", "moves"]}


commands = {"pilot": json.loads((pilot / "commands.json").read_text()), "final": json.loads((out / "commands.json").read_text())}
for records in commands.values():
    row = next(r for r in records if r["name"] == name and r["phase"] == "before")
    assert row["source"] == "5035fa5895a3bdcc9542bbeebabd75f3d955450b" and row["exit_code"] == 0 and row["quota_clean"]
proof = w / "quality-query-storage-pilot-v1-independent.json"
validation = json.loads(proof.read_text())
assert validation["complete"]
old_proof = validation["cases"][name]["independent"]["before"]
assert old_proof["full_hull"]["valid"] and old_proof["boundary_self_intersection_pairs"] == 0
assert old_proof["mesh_sha256"] == hashlib.sha256(paths["other_baseline"].read_bytes()).hexdigest()
results = {key: json.loads(path.with_name("result.json").read_text()) for key, path in paths.items()}
parameters = lambda d: {k: v for k, v in d["parameters"].items() if k != "output"}
assert parameters(results["other_baseline"]) == parameters(results["before"]) == parameters(results["after"])
assert work(results["other_baseline"]) == work(results["before"]) == work(results["after"])
report = dict(scope=__doc__, baseline="5035fa5895a3bdcc9542bbeebabd75f3d955450b", reference_proof_sha256=hashlib.sha256(proof.read_bytes()).hexdigest(), meshes={key: dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for key, path in paths.items()}, parameters_equal=True, work_equal=True, complete=False, limitations=["The v1 diagnostic JSON incorrectly encoded NumPy integer cell IDs as booleans through a generic serialization fallback. This version writes Python integer IDs; the original v1 diagnostic remains retained and is not a connectivity proof.", "This demonstrates baseline output variation; it does not identify a particular reduction or prove its exact cause.", "The main v2 independent checker must still validate both current before/after meshes; this report does not replace those checks."])
with np.load(paths["other_baseline"]) as old, np.load(paths["before"]) as before, np.load(paths["after"]) as after:
    for key in ["input_points", "input_faces", "roles"]:
        assert old[key].tobytes() == before[key].tobytes() == after[key].tobytes()
    a, b, c = [cells(data, "full_tetrahedra") for data in [old, before, after]]
    assert a != b and a == c
    assert (b-c) == (b-a) and (c-b) == (a-b)
    material = cells(before, "tetrahedra")
    assert material == cells(old, "tetrahedra") == cells(after, "tetrahedra")
    removed, added = sorted(b-c), sorted(c-b)
    assert not (set(removed) | set(added)) & material
    report.update(production_full_hull_matches_other_unchanged_baseline=True, baseline_variation_matches_production_difference=True, removed_full_cells=removed, added_full_cells=added, all_changed_full_cells_are_outside_material=True)
    for label, x, y in [("baseline_variation", old, before), ("production_difference", before, after)]:
        indices = np.argwhere(x["points"].view(np.uint32) != y["points"].view(np.uint32))
        report[label] = [dict(site=int(i), axis=int(j), role=int(x["roles"][i]), before=float(x["points"][i,j]), after=float(y["points"][i,j])) for i,j in indices]
report["complete"] = True
dest.write_text(json.dumps(report, indent=2) + "\n")
print("Production matches a certified unchanged baseline;", len(removed), "full exterior cells replaced by", len(added), "; material connectivity unchanged.")
