"""Sequential unchanged-runtime diagnostics for issue130."""
import json
import os
from pathlib import Path
import subprocess
import time

root=Path.cwd();w=root/'.work/vtm';rows=[]
for case,quad in [('graded-torus',4),('thin-hand',20),('disconnected',20),('counterbored-plate',16)]:
    stem=w/f'reuse-inputs-{case}'
    command=[str(root/'.venv/bin/python'),str(w/'profile-reuse-inputs.py'),'--case',case,'--n','1000','--subdivisions',str(quad),'--crease-subdivisions',str(quad),'--repeats','1','--output',str(stem)]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.',WARPOWER_PROBE_GALLERY='1',WARPOWER_PROBE_OUTPUT=str(stem.with_suffix('.json')))
    start=time.time()
    with stem.with_suffix('.log').open('w') as log:result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
    rows.append(dict(case=case,command=command,exit_code=result.returncode,wall_seconds=time.time()-start))
    (w/'reuse-input-profile-commands.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(case,result.returncode,time.time()-start,flush=True)
    if result.returncode:raise SystemExit(result.returncode)
