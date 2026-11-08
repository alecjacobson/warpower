"""Begin CPU validation only after every clean GPU timing job has finished."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

root=Path.cwd()
w=root/'.work/vtm'
dest=w/'surface-query-order-independent-v1.json'
log=w/'surface-query-order-final-v1-independent.log'
assert not dest.exists() and not log.exists()
while True:
    p=w/'surface-query-order-pipeline-v1.json'
    if p.exists():
        jobs=json.loads(p.read_text())
        timing=next((r for r in jobs if r['name']=='timing'),None)
        if timing is not None:
            assert timing['exit_code']==0,timing
            break
    time.sleep(10)
commands=json.loads((w/'surface-query-order-final-v1/commands.json').read_text())
assert len(commands)==32 and all(r['exit_code']==0 and r['quota_clean'] for r in commands)
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONPATH='.')
cmd=[str(root/'.venv/bin/python'),str(w/'check-surface-query-order-final-v1.py')]
start=time.time()
with log.open('w') as stream:
    p=subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        code=p.wait(timeout=1200)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid,signal.SIGTERM)
        p.wait()
        code=124
row=dict(command=cmd,exit_code=code,wall_seconds=time.time()-start,timeout_seconds=1200,all_32_clean_timings_finished_before_CPU_started=True,log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
dest.write_text(json.dumps(row,indent=2)+'\n')
print(row,flush=True)
assert code==0,row
