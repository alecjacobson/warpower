"""Fresh pair after an unexpectedly slow unchanged diagram control; retain both."""
import json
import os
import subprocess
import time
from pathlib import Path

root=Path.cwd();w=root/'.work/vtm';out=w/'prepared-query-diagram-repeat'
assert not out.exists();out.mkdir()
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.')
assert env.get('WARP_CACHE_PATH')=='/opt/warpower-generated-cache'
old=json.loads((w/'prepared-query-final/commands.json').read_text());rows=[]
for phase in ['before','after']:
    ref=next(r for r in old if r['name']=='diagram100000' and r['phase']==phase)
    cmd=ref['command'][:];path=out/(phase+'.json');cmd[cmd.index('--output')+1]=str(path)
    start=time.time()
    with (out/(phase+'.log')).open('w') as log:
        p=subprocess.run(cmd,cwd=ref['cwd'],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
    row=dict(command=cmd,cwd=ref['cwd'],source=ref['source'],phase=phase,exit_code=p.returncode,wall_seconds=time.time()-start)
    if path.exists():
        d=json.loads(path.read_text());row.update(median_ms=1000*d['median_seconds'],quota_clean=not d['cpu_throttled'])
    rows.append(row);(out/'commands.json').write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True)
    assert p.returncode==0 and row.get('quota_clean'),row
