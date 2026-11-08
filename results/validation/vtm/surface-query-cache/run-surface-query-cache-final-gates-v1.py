"""After clean timing, overlap independent CPU geometry with sequential GPU audit jobs."""
import json
import os
from pathlib import Path
import subprocess
import time

root=Path.cwd();w=root/'.work/vtm';python=root/'.venv/bin/python'
commands=json.loads((w/'surface-query-cache-final/commands.json').read_text())
assert len(commands)==36 and all(r['quota_clean'] for r in commands)
output=w/'surface-query-cache-final-gates-v1.json';assert not output.exists()
env=dict(os.environ,PYTHONPATH='.:.work/vtm',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4')
records=[]
def save():output.write_text(json.dumps(records,indent=2)+'\n')
checker=[str(python),str(w/'check-surface-query-cache-final-v1.py')]
cpu_log=w/'surface-query-cache-final-independent-v1.log';assert not cpu_log.exists()
start=time.time()
with cpu_log.open('w') as log:
    cpu=subprocess.Popen(checker,env=env,stdout=log,stderr=subprocess.STDOUT)
    try:
        jobs=[('memory',[str(python),str(w/'run-surface-query-cache-memory-v1.py')]),
              ('native-full',[str(python),str(w/'run-surface-query-cache-native-v1.py')]),
              ('native-loop',[str(python),str(w/'run-surface-query-cache-native-v1.py'),'--loop']),
              ('torus-profile',[str(python),str(w/'run-surface-query-cache-torus-profile-v1.py')])]
        for name,command in jobs:
            dest=w/('gate-surface-query-cache-'+name+'-v1.log');assert not dest.exists()
            began=time.time()
            with dest.open('w') as stream:r=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
            row=dict(name=name,command=command,exit_code=r.returncode,wall_seconds=time.time()-began)
            records.append(row);save();print(row,flush=True);assert r.returncode==0
        code=cpu.wait()
        records.append(dict(name='independent-geometry',command=checker,exit_code=code,wall_seconds=time.time()-start));save();print(records[-1],flush=True);assert code==0
    finally:
        if cpu.poll() is None:cpu.terminate();cpu.wait()
