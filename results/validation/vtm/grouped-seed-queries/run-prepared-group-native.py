"""Native full solve and device-controlled producer/consumer capture evidence."""
import json
import os
import signal
import subprocess
import time
from pathlib import Path

root=Path.cwd();w=root/'.work/vtm';python=root/'.venv/bin/python';rows=[]
audit=json.loads((w/'prepared-group-memory-commands.json').read_text())
assert len(audit)==6 and all(r['exit_code']==0 and r['zero_memcheck_errors'] for r in audit)
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.')
assert env.get('WARP_CACHE_PATH')=='/opt/warpower-generated-cache'
def run(name,cmd,timeout=300):
    cmd=list(map(str,cmd));path=w/('prepared-group-'+name+'.log');assert not path.exists();start=time.time()
    with path.open('w') as log:
        p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            code=124
    rows.append(dict(name=name,command=cmd,exit_code=code,wall_seconds=time.time()-start,timeout_seconds=timeout))
    (w/'prepared-group-native-commands.json').write_text(json.dumps(rows,indent=2)+'\n');print(rows[-1],flush=True);assert code==0,rows[-1]
for name,script in [('full','prepared-group-full-native.py'),('loop','prepared-group-loop.py')]:
    stem=w/('prepared-group-'+name+'-native')
    cmd=['nsys','profile','--trace=cuda','--cuda-graph-trace=node','--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=true','-o',stem,python,w/script,'--dot',stem.with_suffix('.dot'),'--output',stem.with_suffix('.json')]
    if name=='loop':cmd.append('--profile')
    run(name+'-native',cmd)
run('native-analysis',[python,w/'analyze-prepared-group-audit.py'],180)
