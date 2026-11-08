"""Run private register-cap contracts and static profiles, one GPU process at a time."""
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

root=Path.cwd();w=root/'.work/vtm';out=w/'surface-register-limits-v2';assert not out.exists()
assert not subprocess.check_output(['git','status','--porcelain'],text=True)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert os.environ['WARP_CACHE_PATH']=='/opt/warpower-generated-cache'
assert json.loads((w/'cpu-surface-policies-merge-audit-v1.json').read_text())['root_clean']
out.mkdir();python=root/'.venv/bin/python';rows=[]
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.'+os.pathsep+str(w))
def run(name,cmd):
    start=time.time();log=out/(name+'.log')
    with log.open('w') as stream:
        p=subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=p.wait(timeout=600)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM);p.wait();code=124
    row=dict(name=name,command=cmd,exit_code=code,wall_seconds=time.time()-start,timeout_seconds=600,log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
    rows.append(row);(out/'commands.json').write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True);return code
passed=[]
for limit in [96,128,192]:
    name=f'contracts-r{limit}'
    reference=w/'surface-register-limits-v1'/(name+'.json')
    contract=json.loads(reference.read_text());assert contract['complete'] and contract['passed'] and contract['exit_code']==0
    for path,digest in contract['source_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    assert '19 passed' in (reference.with_suffix('.log')).read_text()
    rows.append(dict(name=name,reused_from=str(reference),sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),exit_code=0))
    passed.append(limit)
(out/'commands.json').write_text(json.dumps(rows,indent=2)+'\n')
base=w/'surface-query-order-final-v1'
for name,path in [('torus1000',base/'torus1000-after.npz'),('torus16384',base/'torus16384-fixed32-after.npz'),('armadillo-uniform8192',base/'armadillo-uniform8192-recovery-after/mesh.npz'),('armadillo-graded8192',base/'armadillo-graded8192-recovery-after/mesh.npz')]:
    code=run(name,[str(python),str(w/'profile-surface-register-limits-v2.py'),'--input',str(path),'--output',str(out/(name+'.json')),'--limits',*[str(x) for x in passed]])
    assert code==0,name
print('Private static qualification complete; no production adoption or full-solve speedup claim.')
