import os,json,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';baseline=w/'extraction-geometry';rows=[];env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.')
def run(phase,name,module,args):
 cwd=baseline if phase=='before' else root;stem=w/f'radix-{phase}-{name}';cmd=([str(root/'.venv/bin/python'),str(root/'benchmarks/domain.py')] if module=='benchmarks.domain' else [str(root/'.venv/bin/python'),'-m',module])+list(map(str,args));start=time.time()
 with stem.with_suffix('.log').open('w') as f:r=subprocess.run(cmd,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT)
 rows.append(dict(phase=phase,name=name,command=cmd,cwd=str(cwd),seconds=time.time()-start,exit_code=r.returncode,source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=cwd,text=True).strip()));(w/'radix-pilot-commands.json').write_text(json.dumps(rows,indent=2)+'\n');print(phase,name,r.returncode,round(time.time()-start,2),flush=True)
 if r.returncode:raise SystemExit(r.returncode)
for phase in ['before','after']:run(phase,'domain','benchmarks.domain',['--repeats',15,'--output',w/f'radix-{phase}-domain.json'])
for phase in ['before','after']:run(phase,'torus1000','benchmarks.mesher',['--n',1000,'--repeats',5,'--save-mesh',w/f'radix-{phase}-torus1000.npz','--output',w/f'radix-{phase}-torus1000.json'])
for phase in ['before','after']:run(phase,'counterbored-plate','examples.vtm_gallery',['--case','counterbored-plate','--n',1000,'--subdivisions',16,'--crease-subdivisions',16,'--repeats',5,'--output',w/f'radix-{phase}-counterbored-plate'])
