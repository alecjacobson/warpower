"""Native traces after final timing/geometry and the focused memory gates."""
import json,os,signal,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';python=root/'.venv/bin/python';rows=[]
audit=json.loads((w/'analytic-audit-commands.json').read_text())
assert len(audit)==19 and all(r['exit_code']==0 and r['zero_memcheck_errors'] for r in audit[:18])
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.')
assert env.get('WARP_CACHE_PATH')=='/opt/warpower-generated-cache'
def run(name,cmd,timeout=300):
    cmd=list(map(str,cmd));path=w/('analytic-'+name+'.log');assert not path.exists()
    start=time.time()
    with path.open('w') as log:
        p=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            code=124
    rows.append(dict(name=name,command=cmd,exit_code=code,wall_seconds=time.time()-start,timeout_seconds=timeout));(w/'analytic-native-commands.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(rows[-1],flush=True);assert code==0,rows[-1]
for name,script in [('full','analytic-full-native.py'),('loop','analytic-loop.py')]:
    stem=w/('analytic-'+name+'-native')
    cmd=['nsys','profile','--trace=cuda','--cuda-graph-trace=node','--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=true','-o',stem,python,w/script,'--dot',stem.with_suffix('.dot'),'--output',stem.with_suffix('.json')]
    if name=='loop':cmd.append('--profile')
    run(name+'-native',cmd)
run('native-analysis',[python,w/'analyze-analytic-audit.py'],180)
