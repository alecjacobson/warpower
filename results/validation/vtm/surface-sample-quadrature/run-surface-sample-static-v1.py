"""One GPU process at a time; contracts must pass before static quadrature timing."""
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

root=Path.cwd();w=root/'.work/vtm';out=w/'surface-sample-static-v1';assert not out.exists()
contract=json.loads((w/'surface-sample-parallel-contracts-v1.json').read_text());assert contract['complete'] and contract['passed']
for p,h in contract['source_sha256'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
assert not subprocess.check_output(['git','status','--porcelain'],text=True)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
out.mkdir();rows=[];base=w/'surface-query-order-final-v1'
cases=[('torus1000',base/'torus1000-after.npz'),('torus16384',base/'torus16384-fixed32-after.npz'),('armadillo-uniform8192',base/'armadillo-uniform8192-recovery-after/mesh.npz'),('armadillo-graded8192',base/'armadillo-graded8192-recovery-after/mesh.npz')]
for name,path in cases:
    cmd=[str(root/'.venv/bin/python'),str(w/'profile-surface-sample-parallel-v1.py'),'--input',str(path),'--output',str(out/(name+'.json'))]
    start=time.time()
    with (out/(name+'.log')).open('w') as stream:
        p=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=p.wait(timeout=600)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);p.wait();code=124
    row=dict(name=name,command=cmd,exit_code=code,wall_seconds=time.time()-start,timeout_seconds=600)
    rows.append(row);(out/'commands.json').write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True);assert code==0
