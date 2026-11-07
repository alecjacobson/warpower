"""One reverse-order confirmation pair; pool every original and new sample."""
import json,os,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=w/'prepared-group-hand-repeat'
assert not out.exists();out.mkdir()
assert len(json.loads((w/'prepared-group-final/commands.json').read_text()))==18
assert len(json.loads((w/'prepared-group-final-independent.json').read_text())['cases'])==8
rows=[]
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.')
assert env.get('WARP_CACHE_PATH')=='/opt/warpower-generated-cache'
for phase,cwd in [('after',root),('before',w/'extraction-geometry')]:
    command=[str(root/'.venv/bin/python'),'-m','benchmarks.delaunay_recovery','--case','fitted-hand','--repeats','9','--output',str(out/(phase+'.json'))]
    start=time.time()
    with (out/(phase+'.log')).open('w') as f:r=subprocess.run(command,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180)
    d=json.loads((out/(phase+'.json')).read_text()) if (out/(phase+'.json')).exists() else {}
    row=dict(phase=phase,command=command,cwd=str(cwd),source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=cwd,text=True).strip(),exit_code=r.returncode,wall_seconds=time.time()-start,median_ms=d.get('full',{}).get('median_ms'))
    rows.append(row);(out/'commands.json').write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True)
    assert r.returncode==0 and d['error']==0 and all(v['throttled_periods']==0 and v['throttled_usec']==0 for v in d['full']['cpu'].values())
