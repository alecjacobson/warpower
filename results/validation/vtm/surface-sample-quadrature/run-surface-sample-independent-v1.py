"""Wait for all clean timing windows to end, then validate current geometry and exports."""
import json
import os
import signal
import subprocess
import time
from pathlib import Path

root=Path.cwd();w=root/'.work/vtm';dest=w/'surface-sample-independent-v1.json';assert not dest.exists()
started=time.monotonic()
while True:
    p=w/'surface-sample-pipeline-v1.json'
    if p.exists():
        rows=json.loads(p.read_text())
        row=next((r for r in rows if r['name']=='timing'),None)
        if row is not None:assert row['exit_code']==0;break
    assert time.monotonic()-started<3600,'Timed out waiting for clean timings'
    time.sleep(10)
cmd=[str(root/'.venv/bin/python'),str(w/'check-surface-sample-pairs-v4.py'),'--folder',str(w/'surface-sample-final-v1'),'--output',str(w/'surface-sample-final-v1-independent.json')]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONPATH='.')
log=w/'surface-sample-final-v1-independent.log';assert not log.exists();start=time.time()
with log.open('w') as stream:
    p=subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    try:code=p.wait(timeout=1200)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);p.wait();code=124
row=dict(command=cmd,exit_code=code,wall_seconds=time.time()-start,timeout_seconds=1200,all_clean_timing_jobs_completed_first=True)
dest.write_text(json.dumps(row,indent=2)+'\n');print(row,flush=True);assert code==0
