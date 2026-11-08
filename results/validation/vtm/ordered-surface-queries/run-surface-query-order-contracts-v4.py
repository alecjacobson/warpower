"""Run private size-gated ordering contracts with immutable source bindings."""
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

root=Path.cwd()
w=root/'.work/vtm'
dest=w/'surface-query-order-contracts-v4.json'
log=w/'surface-query-order-contracts-v4.log'
assert not dest.exists() and not log.exists()
assert not subprocess.check_output(['git','status','--porcelain'],text=True)
cmd=[str(root/'.venv/bin/python'),str(w/'test-surface-query-order-contracts-v4.py')]
paths=[w/n for n in ['surface_query_order_v1.py','surface_query_order_v2.py','surface_query_order_v4.py','test-surface-query-order-contracts-v4.py','test_surface_query_order_gate_v2.py']]+[root/'warpower/_surface_query_cache.py',root/'tests/test_surface_query_cache.py',Path(__file__).resolve()]
hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.'+os.pathsep+str(w))
assert env['WARP_CACHE_PATH']=='/opt/warpower-generated-cache'
start=time.time()
with log.open('w') as stream:
    p=subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        code=p.wait(timeout=600)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid,signal.SIGTERM)
        p.wait()
        code=124
row=dict(scope=__doc__,command=cmd,source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),exit_code=code,wall_seconds=time.time()-start,timeout_seconds=600,passed=code==0 and '19 passed' in log.read_text(),source_sha256=hashes,log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
dest.write_text(json.dumps(row,indent=2)+'\n')
print(row,flush=True)
assert row['passed'],row
