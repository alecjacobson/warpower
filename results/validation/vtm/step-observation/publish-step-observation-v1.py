"""Publish read-only VTM trial observation and all retained failures immutably."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

root=Path.cwd();w=root/".work/vtm";out=root/"results/validation/vtm/step-observation"
assert not out.exists()
read=lambda n:json.loads((w/n).read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
summary=read("step-observer-summary-v1.json")
assert summary["complete"] and len(summary["cases"])==11
assert all(sha(root/p)==h for p,h in summary["source_sha256"].items())
assert len(summary["source_sha256"])==267
assert not subprocess.check_output(["git","diff",summary["runtime"],"--","warpower","tests","benchmarks","examples","README.md","pyproject.toml","LICENSE"])
assert read("step-observer-contracts-v1.json")["exit_code"]==0
memory=read("step-observer-memory-v4.json")
assert memory["complete"] and len(memory["rows"])==4
assert all(r["exit_code"]==0 and r["zero_memcheck_errors"] and r["passed"] for r in memory["rows"])
assert read("step-observer-memory-v1.json")["exit_code"]!=0
assert read("step-observer-memory-v2.json")["exit_code"]!=0
for folder,count in [("step-observer-smoke-v1",2),("step-observer-matrix-v1",20)]:
    commands=read(folder+"/commands.json")
    assert len(commands)==count and all(r["exit_code"]==0 and r["valid"]==1 and r["error"]==0 for r in commands)
assert read("step-observer-smoke-v1-analysis.json")["complete"]
assert read("step-observer-matrix-analysis-v3.json")["complete"]
prefixes=("step-observer-","step_observer_","test_step_observer_","run-step-observer-",
          "analyze-step-observer-","summarize-step-observer-","review-step-observer-","publish-step-observation-","verify-step-observation-")
files=[p for p in w.iterdir() if p.is_file() and p.name.startswith(prefixes) and p.suffix!=".md"
       and not any(s in p.name for s in ["premerge-files","merge-audit","publication-review","git-blobs"])]
for folder in ["step-observer-smoke-v1","step-observer-matrix-v1"]:
    files.extend(p for p in (w/folder).rglob("*") if p.is_file())
shared=[w/n for n in ["surface-push-production-source-v1.json","surface-push-final-v1-independent.json",
        "surface-push-final-v1/commands.json","check-tangential-production-memory-v3.py",
        "check-surface-bound-reuse-final-v2.py","check-large-reuse-proof-pairs.py",
        "large-reuse-isolated-test.py","scanned-armadillo-input/armadillo.off"]]
for row in read("step-observer-smoke-v1/commands.json")+read("step-observer-matrix-v1/commands.json"):
    result=Path(row["reference_result"])
    shared.extend([result,result.parent/"mesh.npz" if result.name=="result.json" else result.with_suffix(".npz")])
shared.append(root/"results/validation/vtm/surface-push-pruning/manifest.json")
meta=dict(scope=__doc__,main=summary["main"],runtime=summary["runtime"],production_unchanged=True,
    files=[],shared_references=[],limits=summary["limits"]+[
        "Tiny coordinate variation and four replaced hand full-hull cells are retained and independently certified. Material connectivity/work/minimum quality match.",
        "Two multi-workspace sanitizer batches were explicitly stopped and the isolated v3 producer reached its 300-second deadline. Only completed isolated buffer/v4 jobs qualify.",
        "The Stanford/libigl scanned input retains its original provenance and is not relicensed under MIT.",
        "The wide comparable-quality CPU goal is not established."],reproduction=[
        "NVIDIA L40, Warp1.19.0+cu12, Python3.10. Restore original .work/vtm paths from this manifest and shared prior evidence, or adapt copies of scripts into fresh output paths.",
        "Use PYTHONDONTWRITEBYTECODE=1 WARP_CACHE_PATH=/opt/warpower-generated-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONPATH=.:.work/vtm and .venv/bin/python. One GPU job at a time.",
        "Run run-step-observer-matrix-v1.py with --paired and requested --name entries from recorded commands; choose a fresh --folder. Final warmed output is retained; observer timings are not speed claims.",
        "Use CPU-only single-threaded analysis after GPU runs. The v3 checker retains explicit variations and fresh validation; it can reuse the SHA-bound partial v2 hand-reference proof. Native validators are documented in benchmarks/cpu/README.md.",
        "Run graph contracts and isolated memory-v4 through the established explicit-stack audit helper with its 1200-second full-workspace allowance. Keep partial failures; do not overwrite existing evidence."])
out.mkdir(parents=True)
for p in sorted(set(files)):
    target=out/p.relative_to(w);target.parent.mkdir(parents=True,exist_ok=True)
    assert not target.exists();os.link(p,target);p.chmod(0o444)
    meta["files"].append(dict(original=str(p.relative_to(root)),path=str(target.relative_to(root)),bytes=p.stat().st_size,sha256=sha(p)))
for p in sorted(set(shared)):
    meta["shared_references"].append(dict(path=str(p.relative_to(root)),sha256=sha(p)))
p=out/"manifest.json";p.write_text(json.dumps(meta,indent=2)+"\n");p.chmod(0o444)
print(dict(files=len(meta["files"]),references=len(meta["shared_references"]),manifest_sha256=sha(p)))
