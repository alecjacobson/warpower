"""Paired static profiles after contracts, with one GPU process at a time."""
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

root=Path.cwd(); w=root/'.work/vtm'; out=w/'surface-push-pruning-static-v1'
assert not out.exists()
contract=json.loads((w/'surface-push-pruning-contracts-v1.json').read_text())
assert contract['complete'] and contract['passed']
for p,h in contract['source_sha256'].items(): assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
assert not subprocess.check_output(['git','status','--porcelain'],text=True)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
out.mkdir(); rows=[]; base=w/'surface-sample-final-v1'
cases=[('torus1000',base/'torus1000-after.npz'),('torus16384',base/'torus16384-fixed32-after.npz'),('armadillo-uniform8192',base/'armadillo-uniform8192-recovery-after/mesh.npz'),('armadillo-graded8192',base/'armadillo-graded8192-recovery-after/mesh.npz')]
for name,path in cases:
    pairs={}
    for phase in ['before','after']:
        target=out/(name+'-'+phase+'.json')
        cmd=[str(root/'.venv/bin/python'),str(w/'profile-surface-push-pruning-v1.py'),'--input',str(path),'--output',str(target),'--phase',phase]
        start=time.time()
        with target.with_suffix('.log').open('w') as stream:
            process=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try: code=process.wait(timeout=600)
            except subprocess.TimeoutExpired: os.killpg(process.pid,signal.SIGTERM); process.wait(); code=124
        row=dict(name=name,phase=phase,command=cmd,exit_code=code,wall_seconds=time.time()-start,timeout_seconds=600)
        rows.append(row); (out/'commands.json').write_text(json.dumps(rows,indent=2)+'\n'); print(row,flush=True); assert code==0
        pairs[phase]=json.loads(target.read_text()); assert pairs[phase]['complete']
    for a,b in zip(pairs['before']['rows'],pairs['after']['rows']):
        assert a['name']==b['name']
        if a['name']=='quadrature':
            assert a['values']['error']==b['values']['error']==0
            assert a['output_sha256']['maximum']==b['output_sha256']['maximum']
            assert abs(a['values']['integral']-b['values']['integral'])<=2e-12*abs(a['values']['integral'])
        else: assert a['output_sha256']==b['output_sha256'],(name,a['name'])
    print(name,'paired outputs match',flush=True)
