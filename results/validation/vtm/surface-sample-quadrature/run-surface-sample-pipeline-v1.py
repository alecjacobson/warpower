"""Continue production qualification only after the running committed-source tests pass."""
import json
import os
import subprocess
import time
from pathlib import Path

root=Path.cwd();w=root/'.work/vtm';dest=w/'surface-sample-pipeline-v1.json';assert not dest.exists()
started=time.monotonic()
while True:
    p=w/'surface-sample-production-tests-v2.json'
    if p.exists():
        records=json.loads(p.read_text());assert all(r['exit_code']==0 and r['passed'] for r in records),records
        if len(records)==2:break
    assert time.monotonic()-started<1800,'Timed out waiting for required tests'
    time.sleep(10)
rows=[dict(name='tests',reused_report=str(p),exit_code=0,passed=True)]
dest.write_text(json.dumps(rows,indent=2)+'\n')
for name,script in [('timing','run-surface-sample-final-v1.py'),('qualification','run-surface-sample-qualification-v1.py')]:
    cmd=[str(root/'.venv/bin/python'),str(w/script)];log=w/('surface-sample-pipeline-v1-'+name+'.log');assert not log.exists()
    start=time.time()
    with log.open('w') as stream:
        # Children bound each measured process and preserve partial failures.
        run=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT)
    row=dict(name=name,command=cmd,exit_code=run.returncode,wall_seconds=time.time()-start,child_timeouts=True)
    rows.append(row);dest.write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True);assert run.returncode==0,row
