"""Clean private route-extension candidates using unchanged larger-control commands."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

root=Path.cwd()
w=root/'.work/vtm'
out=w/'extended-reuse-pilot-v1'
assert not out.exists()
assert not subprocess.check_output(['git','status','--porcelain'],text=True)
source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
assert source=='b44ab9e7332551a7a8a1c9bf9697f7bf8d7c109c'
assert '8 passed' in (w/'extended-reuse-tests-v1.log').read_text()
controls=json.loads((w/'fixed-site-current-larger-v1/commands.json').read_text())
assert len(controls)==4
out.mkdir()
rows=[]
for control in controls:
    name=control['name']
    stem=out/name
    cmd=control['command'][:]
    assert cmd[1:3]==['-m','benchmarks.mesher']
    cmd[1:3]=[str(w/'run-extended-reuse-mesher-v1.py'),'--candidate']
    cmd[cmd.index('--save-mesh')+1]=str(stem.with_suffix('.npz'))
    cmd[cmd.index('--output')+1]=str(stem.with_suffix('.json'))
    start=time.time()
    with stem.with_suffix('.log').open('w') as f:
        p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            code=p.wait(timeout=600)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM)
            p.wait()
            code=124
    row=dict(name=name,command=cmd,source=source,exit_code=code,timeout_seconds=600,wall_seconds=time.time()-start)
    if stem.with_suffix('.json').exists():
        d=json.loads(stem.with_suffix('.json').read_text())
        row.update(valid=d['valid'],error=d['error'],counts=d['counts'],median_ms=d['median_gpu_milliseconds'],quota_clean=all(v['throttled_periods']==0 for k in ['cpu_quota_delta','initial_replays_cpu_quota_delta'] for v in d[k].values()),result_sha256=hashlib.sha256(stem.with_suffix('.json').read_bytes()).hexdigest())
    rows.append(row)
    (out/'commands.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(row,flush=True)
