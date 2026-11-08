"""Independently certify changed trajectories; report quality/fit without a speed-only gate."""
import ast
import gzip
import hashlib
import json
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

SCOPE=__doc__
root=Path.cwd();w=root/".work/vtm";folder=w/"selective-step-history-pilot-v2"
out=w/"selective-step-history-pilot-v2-independent-v3.json";assert not out.exists()
assert not subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],text=True).strip()
commands=json.loads((folder/"commands.json").read_text())
assert len(commands)==8 and all(r["exit_code"]==0 and r["quota_clean"] for r in commands)
helper_path=w/"check-surface-bound-reuse-final-v2.py"
exec(compile(helper_path.read_text().split("out=w/'surface-bound-reuse-final'")[0],str(helper_path),"exec"))
helpers=[helper_path,w/"check-large-reuse-proof-pairs.py",w/"check-tangential-production-memory-v3.py",
         w/"check-surface-query-order-final-v1.py",w/"check-surface-push-pairs-v1.py"]
for p,names in [(helpers[2],["oriented_rows"]),(helpers[3],["exact_arrays"]),(helpers[4],["exports"])]:
    defs=[n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(defs)==len(names);exec(compile(ast.Module(body=defs,type_ignores=[]),str(p),"exec"))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
proof_path=w/"surface-push-final-v1-independent.json";prior=json.loads(proof_path.read_text());assert prior["complete"]
sources=[Path(__file__),*helpers,root/"benchmarks/cpu/check_delaunay.cpp",root/"benchmarks/mesh_quality.py",root/"benchmarks/surface_distance.py"]
report=dict(scope=SCOPE,complete=False,source_sha256={str(p):sha(p) for p in sources},prior_proof=str(proof_path),prior_proof_sha256=sha(proof_path),cases={})


def paths(stem):
    return (stem/"result.json",stem/"mesh.npz",stem/"mesh.vtu.gz") if stem.is_dir() else (stem.with_suffix(".json"),stem.with_suffix(".npz"),None)


def certify(name,path,vtu):
    for phase in ["after","before"]:
        reference=paths(w/"surface-push-final-v1"/(name+"-"+phase))[1]
        proof=prior["cases"][name]["independent"][phase]
        assert sha(reference)==proof["mesh_sha256"] and proof["full_hull"]["valid"]
        checks=exact_arrays(reference,path)
        if checks is not None:
            row=dict(proof,mesh_sha256=sha(path),transferred_by_exact_geometry=True,
                reference_mesh=str(reference),reference_mesh_sha256=sha(reference),reference_proof_sha256=sha(proof_path),exact_comparison=checks)
            row.update(exports(path,vtu));return row
    row=validate(path,vtu)
    if vtu is None:
        with np.load(path) as d,tempfile.TemporaryDirectory(prefix="step-history-boundary-",dir="/opt/warpower-generated-cache") as tmp:
            boundary=Path(tmp)/"boundary.off"
            with boundary.open("w") as f:
                f.write(f'OFF\n{len(d["points"])} {len(d["boundary"])} 0\n')
                np.savetxt(f,d["points"],fmt="%.17g")
                np.savetxt(f,np.column_stack((np.full(len(d["boundary"]),3),d["boundary"])),fmt="%d")
            text=subprocess.run([str(w/"surface-reference"),str(boundary)],capture_output=True,text=True,check=True).stdout
            assert int(text.splitlines()[0].split()[-1])==0,text
            row.update(boundary_self_intersection_pairs=0,boundary_reference_stdout=text)
    row.update(exports(path,vtu),fresh_validation=True)
    return row


def work(d):
    return {k:d[k] for k in ["counts","rounds","refinement_rejections","quality_rejections","topology"]}|{
        "steps":d.get("steps",d.get("optimizer_steps")),"moves":d.get("moves",d.get("quality_moves"))}


start=time.time()
for name in dict.fromkeys(r["name"] for r in commands):
    pair=[paths(folder/(name+"-"+phase)) for phase in ["before","after"]]
    a,b=[json.loads(p[0].read_text()) for p in pair]
    params=lambda d:{k:v for k,v in d["parameters"].items() if k not in ["output","save_mesh","dot"]}
    assert params(a)==params(b)
    row=dict(parameters_equal=True,work={"before":work(a),"after":work(b)},independent={})
    report["cases"][name]=row
    with np.load(pair[0][1]) as x,np.load(pair[1][1]) as y:
        assert all(x[k].tobytes()==y[k].tobytes() for k in ["input_points","input_faces"])
        row["input_geometry_equal"]=True
        row["comparison"]=compare(x,y)
    for phase,d,(_,mesh,vtu) in zip(["before","after"],[a,b],pair):
        assert d["valid"]==1 and d["error"]==0;quota(d)
        report["validating"]=str(mesh);out.write_text(json.dumps(report,indent=2)+"\n")
        row["independent"][phase]=certify(name,mesh,vtu)
        proof=row["independent"][phase]
        assert proof["full_hull"]["valid"] and proof["all_boundary_vertex_links_single_cycles"] and proof["boundary_self_intersection_pairs"]==0
        np.testing.assert_allclose(list(proof["quality"]["radius_ratio_percentiles"].values()),
            list(d["quality"]["radius_ratio_percentiles"].values()),rtol=1e-10,atol=1e-12)
    observers=[json.loads((folder/(name+"-"+phase+"-history.json")).read_text()) for phase in ["before","after"]]
    times=[d.get("median_gpu_milliseconds",d.get("median_additional_replay_ms")) for d in [a,b]]
    row["timing"]=dict(before_ms=times[0],after_ms=times[1],speedup=times[0]/times[1],reduction_fraction=1-times[1]/times[0],additional_pool_bytes=observers[1]["persistent_workspace_pool_bytes"]-observers[0]["persistent_workspace_pool_bytes"])
    row["history_statistics"]=observers[1]["history_statistics"]
    qa,qb=[row["independent"][p]["quality"] for p in ["before","after"]]
    metrics=["mean_radius_ratio","two_sided_area_weighted_rms","two_sided_area_weighted_rms_percent_bbox_diagonal","relative_volume_error","dihedral_degrees_min","dihedral_degrees_max"]
    row["quality_comparison"]={k:dict(before=qa[k],after=qb[k],difference=qb[k]-qa[k]) for k in metrics}
    row["quality_comparison"]["radius_ratio_percentiles"]={k:dict(before=qa["radius_ratio_percentiles"][k],after=qb["radius_ratio_percentiles"][k],difference=qb["radius_ratio_percentiles"][k]-qa["radius_ratio_percentiles"][k]) for k in qa["radius_ratio_percentiles"]}
    print(name,row["timing"],"minimum",row["quality_comparison"]["radius_ratio_percentiles"]["0"],flush=True)
    out.write_text(json.dumps(report,indent=2)+"\n")
report.update(complete=True,wall_seconds=time.time()-start)
report.pop("validating",None);out.write_text(json.dumps(report,indent=2)+"\n")
