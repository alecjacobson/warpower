"""Validate read-only traces and report retrospective trial-count opportunities.

V3 explicitly retains alternate valid full-hull cells in the hand rerun.
Material connectivity/work remain equal; all changed full hulls need exact certification.
These are counts on observed trajectories, not predicted runtime or a proof
that skipping trials preserves solver history. Geometry/control equivalence is
checked separately against the current uninstrumented complete references.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument("--folder", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args=parser.parse_args()
assert not args.output.exists()
w=Path(".work/vtm")
helper=w/"check-tangential-production-memory-v3.py"
fn=next(n for n in ast.parse(helper.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=="oriented_rows")
exec(compile(ast.Module(body=[fn],type_ignores=[]),str(helper),"exec"))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
proof_path=w/"surface-push-final-v1-independent.json"
proof=json.loads(proof_path.read_text())
assert proof["complete"]
commands=json.loads((args.folder/"commands.json").read_text())
report=dict(scope=__doc__,complete=False,source_sha256={str(p):sha(p) for p in [Path(__file__),helper,proof_path]},cases={})


def work(d):
    return {k:d[k] for k in ["counts","rounds","topology","quality_rejections","refinement_rejections"]} | {
        "steps":d.get("steps",d.get("optimizer_steps")), "moves":d.get("moves",d.get("quality_moves"))}


def mesh_path(result):
    return result.parent/"mesh.npz" if result.name=="result.json" else result.with_suffix(".npz")


def geometry(reference, actual):
    checks={}
    with np.load(reference) as x,np.load(actual) as y:
        assert set(x.files)==set(y.files)
        for k in x.files:
            a,b=x[k],y[k]
            assert a.dtype==b.dtype and a.shape==b.shape
            if k in ["tetrahedra","full_tetrahedra","boundary"]:
                checks[k]=bool(np.array_equal(oriented_rows(a),oriented_rows(b)))
            elif k=="radius_ratio":
                # Quality rows follow cell order. Compare paired canonical cells.
                def sorted_quality(z):
                    t=np.sort(z["tetrahedra"],axis=1)
                    return z[k][np.lexsort(t.T[::-1])]
                checks[k]=bool(np.array_equal(sorted_quality(x),sorted_quality(y)))
            else:
                checks[k]=a.tobytes()==b.tobytes()
    return checks


validation_helper=w/"check-surface-bound-reuse-final-v2.py"
exec(compile(validation_helper.read_text().split("out=w/'surface-bound-reuse-final'")[0],str(validation_helper),"exec"))
for p in [validation_helper,w/"check-large-reuse-proof-pairs.py"]:
    report["source_sha256"][str(p)]=sha(p)


def variation(a,b):
    with np.load(a) as x,np.load(b) as y:
        changed=np.argwhere(x["points"].view(np.uint32)!=y["points"].view(np.uint32))
        delta=float(np.max(np.abs(x["points"].astype(float)-y["points"].astype(float))))
        full_x=set(map(tuple,np.sort(x["full_tetrahedra"],axis=1)))
        full_y=set(map(tuple,np.sort(y["full_tetrahedra"],axis=1)))
        material_x=set(map(tuple,np.sort(x["tetrahedra"],axis=1)))
        material_y=set(map(tuple,np.sort(y["tetrahedra"],axis=1)))
        return dict(changed_scalar_coordinates=len(changed),maximum_coordinate_difference=delta,
                    removed_full_cells=[list(map(int,t)) for t in sorted(full_x-full_y)],
                    added_full_cells=[list(map(int,t)) for t in sorted(full_y-full_x)],
                    changed_full_cells_outside_material=not bool((full_x-full_y)&material_x or (full_y-full_x)&material_y),
                    witnesses=[[int(i),int(j),float(x["points"][i,j]),float(y["points"][i,j])] for i,j in changed])


for cmd in commands:
    assert cmd["exit_code"]==0
    name=cmd["name"]
    current_path=Path(cmd["result"])
    reference_path=Path(cmd["reference_result"])
    current=json.loads(current_path.read_text())
    reference=json.loads(reference_path.read_text())
    checks=geometry(mesh_path(reference_path),mesh_path(current_path))
    row=dict(mode=cmd["mode"],geometry_equal=checks,work_equal=work(reference)==work(current),
             reference_mesh_sha256=sha(mesh_path(reference_path)),mesh_sha256=sha(mesh_path(current_path)),
             prior_proof_sha256=sha(proof_path))
    expected=proof["cases"][name]["independent"]["after"]
    assert expected["mesh_sha256"]==row["reference_mesh_sha256"] and expected["full_hull"]["valid"]
    row["exact_geometry_proof_transfer"]=all(checks.values())
    # Preserve a nonidentical result for fresh review rather than silently tolerating it.
    report["cases"][name+"-"+cmd["mode"]]=row
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    assert row["work_equal"] and all(v for k,v in checks.items() if k not in ["points","radius_ratio","full_tetrahedra"]),(name,row)
    parameters=lambda d:{k:v for k,v in d["parameters"].items() if k not in ["output","save_mesh","dot"]}
    assert parameters(reference)==parameters(current)
    row["parameters_equal"]=True
    row["coordinate_variation"]=variation(mesh_path(reference_path),mesh_path(current_path))
    assert row["coordinate_variation"]["maximum_coordinate_difference"]<1e-9
    if row["exact_geometry_proof_transfer"]:
        row["independent"]=dict(expected,transferred_from=str(mesh_path(reference_path)))
        row["independent"]["mesh_sha256"]=row["mesh_sha256"]
    else:
        assert current_path.name=="result.json", "Fresh non-gallery geometry needs explicit surface-intersection validation"
        prior_partial_path=w/"step-observer-matrix-analysis-v2.json"
        prior_partial=json.loads(prior_partial_path.read_text())
        bound=next((r["independent"] for r in prior_partial["cases"].values() if r.get("mesh_sha256")==row["mesh_sha256"] and r.get("independent",{}).get("full_hull",{}).get("valid")),None)
        if bound is not None:
            row["independent"]=dict(bound,validated_in_partial_report=str(prior_partial_path),partial_report_sha256=sha(prior_partial_path))
        else:
            row["independent"]=validate(mesh_path(current_path),current_path.parent/"mesh.vtu.gz")
            row["independent"]["fresh_validation"]=True
    assert row["independent"]["full_hull"]["valid"] and row["independent"]["boundary_self_intersection_pairs"]==0
    row["minimum_quality_difference"]=row["independent"]["quality"]["radius_ratio_percentiles"]["0"]-expected["quality"]["radius_ratio_percentiles"]["0"]
    assert abs(row["minimum_quality_difference"])<1e-10
    if cmd["mode"]=="observed":
        reference_command=next(c for c in commands if c["name"]==name and c["mode"]=="reference")
        pair_reference=Path(reference_command["result"])
        row["paired_geometry_equal"]=geometry(mesh_path(pair_reference),mesh_path(current_path))
        row["paired_coordinate_variation"]=variation(mesh_path(pair_reference),mesh_path(current_path))
        assert all(v for k,v in row["paired_geometry_equal"].items() if k not in ["points","radius_ratio","full_tetrahedra"])
        assert row["paired_coordinate_variation"]["maximum_coordinate_difference"]<1e-9
    if cmd["mode"]!="observed":
        continue
    meta=json.loads(Path(cmd["report"]).read_text())
    assert meta["completed"] and meta["overflow"]==0 and sha(meta["trace"])==meta["trace_sha256"]
    with np.load(meta["trace"]) as d:
        ints=d["ints"];floats=d["floats"]
    assert len(ints)==meta["count"]
    ni={k:i for i,k in enumerate(meta["int_columns"])}
    nf={k:i for i,k in enumerate(meta["float_columns"])}
    events=meta["events"]
    cores=[];steps=[];trials=[];stage_stack=[];core=None;step=None;trial=None;reconstruction=None
    reconstructions=[]
    for index,(iv,fv) in enumerate(zip(ints,floats)):
        i=dict(zip(meta["int_columns"],map(int,iv)))
        f=dict(zip(meta["float_columns"],map(float,fv)))
        e=events[i["event"]]
        if e=="stage_enter":stage_stack.append(i["stage"])
        if e=="stage_exit":assert stage_stack.pop()==i["stage"]
        if e=="core_begin":
            assert core is None
            core=dict(id=len(cores),stage=meta["stages"][i["stage"]],mode=i["mode"],active=i["active"],budget=i["budget"],steps=[])
            cores.append(core)
        if e=="start_attempt":
            assert core is not None and trial is None
            step=dict(id=len(steps),core=core["id"],iteration=i["iterations"],trials=[],start_error=i["core_error"])
            steps.append(step);core["steps"].append(step["id"])
            assert f["alpha"]==meta["core_options"]["damping"] and i["backtracks"]==0 and i["accepted"]==0
        if e=="trial_begin":
            assert trial is None and step is not None and core is not None
            trial=dict(id=len(trials),step=step["id"],core=core["id"],stage=core["stage"],mode=i["mode"],active=i["active"],alpha=f["alpha"],begin_index=index,events={},reconstructions=0)
        if e=="reconstruct_pre":
            assert reconstruction is None
            reconstruction=dict(index=index,stage=meta["stages"][i["stage"]],phase=meta["phases"][i["phase"]],active=i["active"])
            if trial is not None:trial["reconstructions"]+=1
        if e=="reconstruct_post":
            assert reconstruction is not None
            reconstruction["error"]=i["mesh_error"]
            reconstructions.append(reconstruction);reconstruction=None
        if trial is not None and e in ["candidate_ready","selection_pre","selection_post","finish_pre","finish_post","local_retry"]:
            assert e not in trial["events"]
            trial["events"][e]=dict(ints=i,floats=f)
        if e=="trial_end":
            assert trial is not None and trial["reconstructions"]==1 and f["alpha"]>0
            ev=trial["events"]
            c=ev["candidate_ready"]["ints"]
            if c["accepted"]==0:
                reason="objective_guard"
                if c["mesh_error"]:reason="delaunay"
                elif c["boundary_error"]:reason="boundary"
                elif c["invalid"]:reason="odt_geometry"
                elif not all(np.isfinite(ev["candidate_ready"]["floats"][k]) for k in ["energy","surface","crease"]):reason="nonfinite_objective"
            else:reason="accepted"
            if "selection_post" in ev:
                before=ev["selection_pre"]["ints"];after=ev["selection_post"]["ints"]
                assert c["accepted"]==1 and before["accepted"]==1
                if after["accepted"]==0:
                    assert after["selection_rejections"]==before["selection_rejections"]+1
                    reason="selection_guard_"+str(after["selection_last_error"])
            if "local_retry" in ev:
                assert "finish_pre" not in ev and "finish_post" not in ev and reason!="accepted"
                assert f["alpha"]==trial["alpha"]
                trial.update(accepted=False,local_retry=True)
            else:
                before=ev["finish_pre"];after=ev["finish_post"]
                bi,ai=before["ints"],after["ints"]
                assert before["floats"]["alpha"]==trial["alpha"]
                accepted=bool(bi["accepted"])
                assert accepted==(reason=="accepted")
                assert ai["iterations"]==bi["iterations"]+int(accepted)
                assert ai["rejected"]==bi["rejected"]+int(not accepted)
                assert ai["backtracks"]==bi["backtracks"]+int(not accepted)
                assert after["floats"]["alpha"]==trial["alpha"]*(1 if accepted else .5)
                if accepted:
                    assert ai["attempt"]==0
                    assert ai["termination"]==int(before["floats"]["maximum"]<=meta["core_options"]["tolerance"])
                else:
                    assert ai["attempt"]==int(ai["backtracks"]<=meta["core_options"]["max_backtracks"])
                trial.update(accepted=accepted,local_retry=False,maximum=before["floats"]["maximum"])
            trial.update(reason=reason,end_index=index)
            step["trials"].append(trial["id"]);trials.append(trial);trial=None
        if e=="core_end":
            assert core is not None and trial is None
            own=[t for t in trials if t["core"]==core["id"]]
            assert sum(t["accepted"] for t in own)==i["iterations"]
            assert sum(not t["accepted"] for t in own)==i["rejected"]
            core.update(termination=i["termination"],iterations=i["iterations"])
            core=None;step=None
    assert not stage_stack and core is None and trial is None and reconstruction is None
    counted_steps=[sum(t["accepted"] for t in trials if t["mode"]==mode) for mode in [0,1]]
    assert counted_steps==work(current)["steps"],(counted_steps,work(current))
    opportunities=[]
    for c in cores:
        previous=None
        for sid in c["steps"]:
            s=steps[sid];ts=[trials[t] for t in s["trials"]]
            accepted=next((t for t in ts if t["accepted"]),None)
            if previous is not None and ts:
                eligible=[t for t in ts if t["alpha"]>previous]
                unknown=accepted is not None and previous<accepted["alpha"]
                opportunities.append(dict(core=c["id"],step=sid,previous_accepted_alpha=previous,
                    current_accepted_alpha=None if accepted is None else accepted["alpha"],
                    starts_below_observed_success=unknown,
                    rejected_prefix_count=0 if unknown else sum(not t["accepted"] for t in eligible),
                    local_retry_present=any(t["local_retry"] for t in ts)))
            if accepted is not None:previous=accepted["alpha"]
    row.update(trace_rows=len(ints),core_count=len(cores),step_count=len(steps),trial_count=len(trials),
        accepted=sum(t["accepted"] for t in trials),rejected=sum(not t["accepted"] for t in trials),
        reconstruction_count=len(reconstructions),
        reconstruction_phases=dict(Counter(r["phase"] for r in reconstructions)),
        reconstruction_stages=dict(Counter(r["stage"] for r in reconstructions)),
        rejection_reasons=dict(Counter(t["reason"] for t in trials if not t["accepted"])),
        accepted_scales=dict(Counter(str(t["alpha"]) for t in trials if t["accepted"])),
        cores=cores,opportunities=opportunities,
        observed_rejected_prefix_bound=sum(o["rejected_prefix_count"] for o in opportunities),
        starts_below_observed_success=sum(o["starts_below_observed_success"] for o in opportunities),
        trials=[{k:v for k,v in t.items() if k!="events"} for t in trials])
    print(name,{k:row[k] for k in ["trial_count","accepted","rejected","reconstruction_count","rejection_reasons","observed_rejected_prefix_bound","starts_below_observed_success"]},flush=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
report["complete"]=True
args.output.write_text(json.dumps(report,indent=2)+"\n")
