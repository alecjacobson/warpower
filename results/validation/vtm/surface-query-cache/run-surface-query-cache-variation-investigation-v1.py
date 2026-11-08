"""Sequential diagnostic repeat trajectories and paired-query audit after all GPU gates."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root=Path.cwd();w=root/'.work/vtm';python=root/'.venv/bin/python'
output=w/'surface-query-cache-variation-investigation-v1.json';assert not output.exists()
memory=json.loads((w/'surface-query-cache-memory-v1-commands.json').read_text())
assert len(memory)==6 and all(r['exit_code']==0 for r in memory)
for name in ['full','loop']:
    assert json.loads((w/f'surface-query-cache-{name}-native-v1-command.json').read_text())['exit_code']==0
assert json.loads((w/'surface-query-cache-torus-profile-v1-command.json').read_text())['exit_code']==0
jobs=[]
for phase,cwd in [('before',w/'surface-query-cache-baseline'),('after',root)]:
    out=w/f'surface-query-cache-repeat-{phase}-v1'
    jobs.append((phase,[str(python),str(w/'run-surface-query-cache-repeat-v1.py'),'--output',str(out)],cwd))
audit=w/'surface-query-cache-paired-audit-hand1-v1'
jobs.append(('paired-hand1',[str(python),str(w/'run-surface-query-cache-paired-audit-v1.py'),
                            '--case','thin-hand','--n','1000','--initial','256','--seed','1',
                            '--quality-rounds','4','--repeats','1','--output',str(audit)],root))
rows=[]
for name,command,cwd in jobs:
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH=str(cwd)+':'+str(w))
    log=w/f'surface-query-cache-variation-{name}-v1.log';assert not log.exists();began=time.time()
    with log.open('w') as stream:r=subprocess.run(command,cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT)
    row=dict(name=name,command=command,cwd=str(cwd),exit_code=r.returncode,wall_seconds=time.time()-began)
    rows.append(row);output.write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True);assert r.returncode==0
archives=[]
for name in ['mesh.vtu','boundary.off']:
    p=audit/name;raw=p.read_bytes();packed=p.with_suffix(p.suffix+'.gz')
    assert not packed.exists();packed.write_bytes(gzip.compress(raw,mtime=0));assert gzip.decompress(packed.read_bytes())==raw
    archives.append(dict(path=str(packed),sha256=hashlib.sha256(packed.read_bytes()).hexdigest(),raw_sha256=hashlib.sha256(raw).hexdigest()))
    (audit/'lossless-archives.json').write_text(json.dumps(archives,indent=2)+'\n');p.unlink()
