"""Native full solve and device-controlled producer/consumer capture evidence."""
import gzip,hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

root=Path.cwd();w=root/'.work/vtm';python=root/'.venv/bin/python';rows=[]
audit=json.loads((w/'local-backtracking-memory-v2-commands.json').read_text())
assert len(audit)==6 and all(r['exit_code']==0 and r['zero_memcheck_errors'] for r in audit)
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.')
assert env.get('WARP_CACHE_PATH')=='/opt/warpower-generated-cache'
def run(name,cmd,timeout=300):
    cmd=list(map(str,cmd));path=w/('local-backtracking-'+name+'.log');assert not path.exists();start=time.time()
    with path.open('w') as log:
        p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            code=124
    rows.append(dict(name=name,command=cmd,exit_code=code,wall_seconds=time.time()-start,timeout_seconds=timeout))
    (w/'local-backtracking-native-commands.json').write_text(json.dumps(rows,indent=2)+'\n');print(rows[-1],flush=True);assert code==0,rows[-1]
for name,script in [('full','local-backtracking-full-native.py'),('loop','local-backtracking-loop-native.py')]:
    stem=w/('local-backtracking-'+name+'-native')
    derived=Path('/opt/warpower-generated-cache/graph-derived');derived.mkdir(parents=True,exist_ok=True)
    dot=derived/(stem.name+'.dot');assert not dot.exists()
    cmd=['nsys','profile','--trace=cuda','--cuda-graph-trace=node','--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=true','-o',stem,python,w/script,'--dot',dot,'--output',stem.with_suffix('.json')]
    if name=='loop':cmd.append('--profile')
    run(name+'-native',cmd)
    raw=stem.with_suffix('.nsys-rep');data=raw.read_bytes();dest=raw.with_suffix('.nsys-rep.gz');assert not dest.exists()
    dest.write_bytes(gzip.compress(data,mtime=0));assert gzip.decompress(dest.read_bytes())==data
    (w/('local-backtracking-'+name+'-trace-archive.json')).write_text(json.dumps(dict(raw_sha256=hashlib.sha256(data).hexdigest(),gzip_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),raw_bytes=len(data),bytes=dest.stat().st_size),indent=2)+'\n')
    raw.unlink()
    data=dot.read_bytes();dest=stem.with_suffix('.dot.gz');assert not dest.exists()
    dest.write_bytes(gzip.compress(data,mtime=0));assert gzip.decompress(dest.read_bytes())==data
run('native-analysis',[python,w/'analyze-local-backtracking-audit.py'],180)
