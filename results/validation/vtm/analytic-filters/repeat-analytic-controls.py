"""Investigate observed one-ulp coordinate variation without changing runtime."""
import json,os,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';rows=[]
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.')
assert env.get('WARP_CACHE_PATH')=='/opt/warpower-generated-cache'
for name in ['cavity','disconnected']:
    stem=w/('analytic-'+name+'-control-repeat')
    cmd=[str(root/'.venv/bin/python'),'-m','examples.vtm_gallery','--case',name,'--n','1000','--subdivisions','20','--crease-subdivisions','20','--repeats','3','--output',str(stem)]
    start=time.time()
    with stem.with_suffix('.log').open('w') as log:
        p=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
    rows.append(dict(name=name,command=cmd,exit_code=p.returncode,wall_seconds=time.time()-start))
    (w/'analytic-coordinate-repeat-commands.json').write_text(json.dumps(rows,indent=2)+'\n')
    assert p.returncode==0
