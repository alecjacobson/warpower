"""Qualify committed sample-parallel quadrature with the full suite and unchanged full recovery loop."""
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

root=Path.cwd()
w=root/'.work/vtm'
dest=w/'surface-sample-production-tests-v2.json'
assert not dest.exists()
source=json.loads((w/'surface-sample-production-source-v2.json').read_text())
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==source['runtime']
assert not subprocess.check_output(['git','status','--porcelain'])
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.')
assert env['WARP_CACHE_PATH']=='/opt/warpower-generated-cache'
rows=[]
for name,tests,expected,timeout in [('full-suite',[],1387,1200),('loop',[str(w/'test_tangential_production_loop_v2.py')],1,600)]:
    cmd=[str(root/'.venv/bin/python'),'-m','pytest','-q',*(['-s'] if name=='loop' else []),*tests]
    runenv=dict(env)
    if name=='loop':
        runenv['WARPOWER_TOPOLOGY_LOOP_OUTPUT']=str(w/'surface-sample-production-loop-v2')
    log=w/('surface-sample-production-'+name+'-v2.log')
    assert not log.exists()
    start=time.time()
    with log.open('w') as stream:
        p=subprocess.Popen(cmd,env=runenv,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            code=p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM)
            p.wait()
            code=124
    row=dict(name=name,command=cmd,runtime=source['runtime'],exit_code=code,wall_seconds=time.time()-start,timeout_seconds=timeout,expected_tests=expected,passed=code==0 and f'{expected} passed' in log.read_text(),log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
    rows.append(row)
    dest.write_text(json.dumps(rows,indent=2)+'\n')
    (w/('surface-sample-production-'+name+'-v2.json')).write_text(json.dumps(row,indent=2)+'\n')
    print(row,flush=True)
    assert row['passed'],row
