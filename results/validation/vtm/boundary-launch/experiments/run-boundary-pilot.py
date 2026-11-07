"""Sequential clean full-solve gates before any production boundary change."""
import json
import os
from pathlib import Path
import subprocess
import time

root=Path.cwd();w=root/'.work/vtm';rows=[]
for case in ('torus','plate'):
    for phase in ('before','scalar32'):
        stem=w/f'boundary-pilot-{phase}-{case}'
        cmd=[str(root/'.venv/bin/python')]
        if phase=='scalar32':cmd += [str(w/'boundary-variant-run.py')]
        else:cmd += ['-m']
        if case=='torus':
            cmd += ['benchmarks.mesher','--n','1000','--repeats','5','--save-mesh',str(stem.with_suffix('.npz')),'--output',str(stem.with_suffix('.json'))]
        else:
            cmd += ['examples.vtm_gallery','--case','counterbored-plate','--n','1000','--subdivisions','16','--crease-subdivisions','16','--repeats','5','--output',str(stem)]
        start=time.time()
        with stem.with_suffix('.log').open('w') as f:
            result=subprocess.run(cmd,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.'),stdout=f,stderr=subprocess.STDOUT)
        rows.append(dict(case=case,phase=phase,command=cmd,exit_code=result.returncode,wall_seconds=time.time()-start,source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
        (w/'boundary-pilot-commands.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(case,phase,result.returncode,time.time()-start,flush=True)
        if result.returncode:raise SystemExit(result.returncode)
