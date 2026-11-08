"""Run exact decision checks before the filtered complete-solve pilot."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

root=Path.cwd();w=root/'.work/vtm';python=root/'.venv/bin/python';rows=[]
assert '14 passed' in (w/'proven-exterior-tests-v2.log').read_text()
assert (w/'hand-proven-exterior-device-v2.json').exists()
jobs=[('ray-oracle',[python,w/'check-centroid-ray-filter-v3.py']),
      ('full-seed7',[python,w/'run-full-proven-recovery-v2.py','--case','thin-hand','--n','1000','--initial','256','--seed','7','--subdivisions','20','--crease-subdivisions','20','--quality-rounds','4','--repeats','3','--output',w/'hand-full-proven-recovery-v2/seed7'])]
for name,cmd in jobs:
    cmd=list(map(str,cmd));log=w/('proven-filter-v3-'+name+'.log');assert not log.exists();start=time.time()
    with log.open('w') as stream:
        p=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=p.wait(timeout=300)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM);p.wait();code=124
    row=dict(name=name,command=cmd,exit_code=code,wall_seconds=time.time()-start,timeout_seconds=300)
    rows.append(row);(w/'proven-filter-gates-v3-commands.json').write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True)
    assert code==0,row
