"""Bind ordinary, native and sanitizer frames after the complete qualification jobs."""
import json
import os
import subprocess
import time
from pathlib import Path

root=Path.cwd();w=root/'.work/vtm';dest=w/'surface-sample-qualified-v1.json';assert not dest.exists()
start=time.monotonic()
while True:
    pipeline=w/'surface-sample-pipeline-v1.json';independent=w/'surface-sample-independent-v1.json'
    if pipeline.exists() and independent.exists():
        rows=json.loads(pipeline.read_text());proof=json.loads(independent.read_text())
        assert proof['exit_code']==0
        qualification=next((r for r in rows if r['name']=='qualification'),None)
        if qualification is not None:assert qualification['exit_code']==0;break
    assert time.monotonic()-start<5400,'Timed out waiting for bounded qualification jobs'
    time.sleep(10)
rows=[]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONPATH='.')
jobs=[('native-full',[str(w/'check-surface-sample-native-full-v1.py')])]
for name,folder in [('ordinary-loop','surface-sample-production-loop-v2'),('native-loop','surface-sample-loop-native-v1-frames'),('padded-loop','surface-sample-memory-v1-loop-padded-frames'),('lifetime-loop','surface-sample-memory-v1-loop-lifetime-frames')]:
    jobs.append((name,[str(w/'check-surface-sample-loop-v1.py'),'--input',str(w/folder),'--output',str(w/('surface-sample-'+name+'-comparison-v1.json'))]))
for name,args in jobs:
    cmd=[str(root/'.venv/bin/python'),*args];log=w/('surface-sample-qualified-v1-'+name+'.log');assert not log.exists();start=time.time()
    with log.open('w') as stream:run=subprocess.run(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT)
    row=dict(name=name,command=cmd,exit_code=run.returncode,wall_seconds=time.time()-start)
    rows.append(row);dest.write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True);assert run.returncode==0,row
