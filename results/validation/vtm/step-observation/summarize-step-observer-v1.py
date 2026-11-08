"""Source binding, observed rejection attribution and cautious scale-history bounds."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

root=Path.cwd();w=root/".work/vtm"
out=w/"step-observer-summary-v1.json";assert not out.exists()
inputs=[w/"step-observer-smoke-v1-analysis.json",w/"step-observer-matrix-analysis-v3.json"]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
previous_source=w/"surface-push-production-source-v1.json"
hashes=json.loads(previous_source.read_text())["source_sha256"]
assert len(hashes)==267
assert all(sha(p)==h for p,h in hashes.items())
reports=[json.loads(p.read_text()) for p in inputs];assert all(r["complete"] for r in reports)
all_cases={k:v for r in reports for k,v in r["cases"].items()}
assert len(all_cases)==22
result=dict(scope=__doc__,runtime="7bd8542d1292512b0594a6ac5e6c75451b77aa50",
    main=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
    source_sha256=hashes,all_267_production_files_unchanged=True,
    input_sha256={str(p):sha(p) for p in [Path(__file__),previous_source,*inputs]},cases={},
    geometry_variations={},full_hull_alternatives={},limits=[
        "Counts come from read-only instrumented trajectories; no instrumented timing is a speed claim.",
        "Rejected-prefix counts assume the observed trajectory only. Skipping trials changes reconstruction reuse history and may change subsequent decisions.",
        "No observation proves rejection monotone in alpha or that an accepted scale remains appropriate after geometry changes.",
        "Within-core history is reset for each core invocation. Cross-stage/active-count reuse is not estimated.",
        "A remembered scale can be smaller than an observed successful scale. Default-scale convergence checks and fallback are required in a candidate policy."])
for key,row in all_cases.items():
    if not all(row["geometry_equal"].values()):
        result["geometry_variations"][key]=row["coordinate_variation"]
    if not row["geometry_equal"]["full_tetrahedra"]:
        result["full_hull_alternatives"][key]=row["coordinate_variation"]
    if row["mode"]!="observed":continue
    name=key.removesuffix("-observed")
    trials=row["trials"]
    assert all(t["mode"] in [0,1] for t in trials)
    assert all(t["accepted"] for t in trials if t["mode"]==0)
    assert not any(t["local_retry"] for t in trials)
    d={k:row[k] for k in ["core_count","step_count","trial_count","accepted","rejected","reconstruction_count","reconstruction_phases","reconstruction_stages","rejection_reasons","accepted_scales"]}
    d["mode_trials"]={str(mode):dict(trials=sum(t["mode"]==mode for t in trials),accepted=sum(t["mode"]==mode and t["accepted"] for t in trials)) for mode in [0,1]}
    d["exhausted_core_calls"]=sum(c["termination"]==2 for c in row["cores"])
    d["scale_history"]={}
    for multiplier in [1,2,4]:
        history=[]
        for c in row["cores"]:
            previous=None
            for sid in c["steps"]:
                ts=[t for t in trials if t["step"]==sid]
                accepted=next((t for t in ts if t["accepted"]),None)
                if previous is not None and ts:
                    initial=min(.5,multiplier*previous)
                    below=accepted is not None and initial<accepted["alpha"]
                    history.append(dict(core=c["id"],step=sid,initial=initial,
                        accepted=None if accepted is None else accepted["alpha"],starts_below_observed_success=below,
                        rejected_prefix=0 if below else sum(t["alpha"]>initial and not t["accepted"] for t in ts)))
                if accepted is not None:previous=accepted["alpha"]
        d["scale_history"][str(multiplier)]=dict(steps_with_history=len(history),rejected_prefix_count=sum(h["rejected_prefix"] for h in history),
            starts_below_observed_success=sum(h["starts_below_observed_success"] for h in history),
            understep_witnesses=[h for h in history if h["starts_below_observed_success"]])
    result["cases"][name]=d
    print(name,"global/selective",d["mode_trials"],"history",d["scale_history"],flush=True)
result["complete"]=True
out.write_text(json.dumps(result,indent=2)+"\n")
