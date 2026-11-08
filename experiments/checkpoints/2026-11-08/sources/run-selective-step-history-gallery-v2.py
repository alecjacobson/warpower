"""Paired complete pilots; guards and independent geometry are required afterward."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

root=Path.cwd();w=root/".work/vtm";out=w/"selective-step-history-gallery-v2"
assert not out.exists()
assert not subprocess.check_output(["git","status","--porcelain"])
assert not subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],text=True).strip()
tests=json.loads((w/"selective-step-history-cuda-contracts-v2.json").read_text())
assert tests["exit_code"]==0 and tests["passed"]==21
pilot=json.loads((w/"selective-step-history-pilot-v2-independent-v3.json").read_text());assert pilot["complete"]
commands=json.loads((w/"surface-push-final-v1/commands.json").read_text())
sources=[Path(__file__),w/"run-selective-step-history-case-v2.py",w/"selective_step_history_v2.py",*root.glob("warpower/*.py")]
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
out.mkdir();rows=[]
(out/"source.json").write_text(json.dumps(dict(source_sha256=hashes),indent=2)+"\n")
for name in ["torus8192-fixed32","torus8192-progressive","thin-hand-seed7","cavity","disconnected","plate","armadillo-graded8192-recovery"]:
    reference=next(r for r in commands if r["name"]==name and r["phase"]=="after")
    for phase in ["before","after"]:
        label=name+"-"+phase;cmd=reference["command"][:]
        benchmark=cmd[1:3]==["-m","benchmarks.mesher"]
        cmd=[cmd[0],str(w/"run-selective-step-history-case-v2.py"),*cmd[3 if benchmark else 2:]]
        cmd[cmd.index("--output")+1]=str(out/(label+".json" if benchmark else label))
        if benchmark:cmd[cmd.index("--save-mesh")+1]=str(out/(label+".npz"))
        env=dict(os.environ,WARPOWER_HISTORY_PHASE=phase,WARPOWER_HISTORY_REPORT=str(out/(label+"-history.json")),WARPOWER_HISTORY_KIND="benchmark" if benchmark else "gallery")
        start=time.time()
        with (out/(label+".log")).open("w") as stream:
            p=subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=p.wait(timeout=600)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);p.wait();code=124
        result=Path(cmd[cmd.index("--output")+1]);result=result/"result.json" if result.is_dir() else result
        row=dict(name=name,phase=phase,command=cmd,exit_code=code,wall_seconds=time.time()-start,timeout_seconds=600)
        if result.exists():
            d=json.loads(result.read_text());row.update(result=str(result),valid=d["valid"],error=d["error"],
                median_ms=d.get("median_gpu_milliseconds",d.get("median_additional_replay_ms")),
                quota_clean=all(v["throttled_periods"]==0 for k in ["cpu_quota_delta","initial_replays_cpu_quota_delta","additional_replays_cpu_quota_delta","first_replay_cpu_quota_delta","initial_cpu_quota_delta"] if k in d for v in d[k].values()))
        rows.append(row);(out/"commands.json").write_text(json.dumps(rows,indent=2)+"\n");print(row,flush=True)
        assert code==0 and row["valid"]==1 and row["error"]==0 and row["quota_clean"],row
        if not benchmark:
            for filename in ["input.off","mesh.vtu","boundary.off"]:
                raw=result.parent/filename;data=raw.read_bytes();packed=raw.with_suffix(raw.suffix+".gz")
                assert not packed.exists();packed.write_bytes(gzip.compress(data,mtime=0))
                assert gzip.decompress(packed.read_bytes())==data;raw.unlink()
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
assert len(rows)==14
