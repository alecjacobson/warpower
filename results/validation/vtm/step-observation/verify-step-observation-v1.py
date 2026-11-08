"""Verify immutable observation publication, source bindings, geometry and contracts."""
import ast
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

root=Path.cwd();out=root/"results/validation/vtm/step-observation"
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda n:json.loads((out/n).read_text())
manifest=read("manifest.json")
for r in manifest["files"]:
    a,b=root/r["original"],root/r["path"]
    assert sha(a)==sha(b)==r["sha256"] and a.stat().st_ino==b.stat().st_ino
    assert not b.stat().st_mode&0o222
    if b.suffix==".gz":gzip.decompress(b.read_bytes())
for r in manifest["shared_references"]:assert sha(root/r["path"])==r["sha256"]


def bindings(value):
    if isinstance(value,dict):
        if isinstance(value.get("source_sha256"),dict):
            for name,h in value["source_sha256"].items():
                p=Path(name);p=p if p.is_absolute() else root/p
                assert sha(p)==h,(name,h)
        for v in value.values():bindings(v)
    elif isinstance(value,list):
        for v in value:bindings(v)


for r in manifest["files"]:
    p=root/r["path"]
    if p.suffix==".json":bindings(json.loads(p.read_text()))
summary=read("step-observer-summary-v1.json")
assert summary["complete"] and len(summary["source_sha256"])==267
assert summary["all_267_production_files_unchanged"]
assert not subprocess.check_output(["git","diff",manifest["runtime"],"--","warpower","tests","benchmarks","examples","README.md","pyproject.toml","LICENSE"])
tree=ast.parse((out/"step_observer_v1.py").read_text())
kernel=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="snapshot")
for n in ast.walk(kernel):
    targets=n.targets if isinstance(n,ast.Assign) else [n.target] if isinstance(n,ast.AugAssign) else []
    for t in targets:
        if isinstance(t,ast.Subscript):
            assert isinstance(t.value,ast.Name) and t.value.id in ["ints","floats","count","overflow"]
commands=[]
for folder,count in [("step-observer-smoke-v1",2),("step-observer-matrix-v1",20)]:
    rows=read(folder+"/commands.json");assert len(rows)==count
    for r in rows:
        assert r["exit_code"]==0 and r["valid"]==1 and r["error"]==0
        result=json.loads(Path(r["result"]).read_text());assert result["valid"]==1 and result["error"]==0
        meta=json.loads(Path(r["report"]).read_text());assert meta["completed"]
        if r["mode"]=="observed":
            assert 0<meta["count"]<meta["capacity"] and meta["overflow"]==0
            assert sha(Path(meta["trace"]))==meta["trace_sha256"]
    commands.extend(rows)
smoke=read("step-observer-smoke-v1-analysis.json");matrix=read("step-observer-matrix-analysis-v3.json")
assert smoke["complete"] and matrix["complete"]
cases={**smoke["cases"],**matrix["cases"]};assert len(cases)==22
fresh=0
for key,r in cases.items():
    assert r["work_equal"]
    assert all(v for k,v in r["geometry_equal"].items() if k not in ["points","radius_ratio","full_tetrahedra"])
    if "independent" in r:
        p=r["independent"]
        assert p["full_hull"]["valid"] and p["all_boundary_vertex_links_single_cycles"] and p["boundary_self_intersection_pairs"]==0
        assert p["mesh_sha256"]==r["mesh_sha256"]
        fresh+=int(p.get("fresh_validation",False))
        assert r["minimum_quality_difference"]==0
    else:assert r["exact_geometry_proof_transfer"]
    if not r["exact_geometry_proof_transfer"]:
        assert r["coordinate_variation"]["maximum_coordinate_difference"]<1e-9
        assert r["independent"]["fresh_validation"]
assert fresh==3
assert list(summary["full_hull_alternatives"])==["thin-hand-seed7-observed"]
alternative=summary["full_hull_alternatives"]["thin-hand-seed7-observed"]
assert len(alternative["removed_full_cells"])==len(alternative["added_full_cells"])==4
assert alternative["changed_full_cells_outside_material"]
for name,r in summary["cases"].items():
    assert r["mode_trials"]["0"]["trials"]==r["mode_trials"]["0"]["accepted"]
    assert r["trial_count"]==r["accepted"]+r["rejected"]
assert summary["cases"]["torus16384-fixed32"]["scale_history"]["2"]["rejected_prefix_count"]==99
assert summary["cases"]["armadillo-graded4096"]["scale_history"]["2"]["starts_below_observed_success"]==2
uniform=read("step-observer-uniform-rejections-v1.json")
assert uniform["complete"] and uniform["rows"]==[dict(error=16,retained_count=0,guard_reason=3,count=57)]
for p,h in uniform["inputs"].items():assert sha(root/p)==h
assert read("step-observer-contracts-v1.json")["exit_code"]==0
assert "9 passed" in (out/"step-observer-contracts-v1.log").read_text()
memory=read("step-observer-memory-v4.json")
assert memory["complete"] and len(memory["rows"])==4
for r in memory["rows"]:
    assert r["exit_code"]==0 and r["zero_memcheck_errors"] and r["passed"]
    log=Path(r["log"]).read_text()
    assert "ERROR SUMMARY: 0 errors" in log and f"{r['expected_tests']} passed" in log
assert read("step-observer-memory-v1.json")["exit_code"]!=0
assert read("step-observer-memory-v2.json")["exit_code"]!=0
assert read("step-observer-memory-v3.json")["rows"][-1]["exit_code"]==124
assert not read("step-observer-matrix-prefix-v2.json")["complete"]
assert not read("step-observer-matrix-analysis-v2.json")["complete"]
result=dict(scope=__doc__,artifacts=len(manifest["files"]),manifest_sha256=sha(out/"manifest.json"),
    verifier_sha256=sha(Path(__file__)),source_files_unchanged=267,paired_jobs=22,
    meshes=22,exact_geometry_transfers=19,fresh_validations=3,ordinary_contracts=9,
    padded_memory_contracts=5,isolated_memory_jobs=4,zero_memory_errors=True,runtime_changes=False)
p=out/"premerge-integrity.json";assert not p.exists();p.write_text(json.dumps(result,indent=2)+"\n");p.chmod(0o444)
print(result)
