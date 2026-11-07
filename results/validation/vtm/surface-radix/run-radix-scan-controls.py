import os,json,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';baseline=w/'extraction-geometry';rows=[];env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.')
def run(phase,name,module,args):
 cwd=baseline if phase=='before' else root;stem=w/f'radix-{phase}-{name}';cmd=[str(root/'.venv/bin/python'),'-m',module]+list(map(str,args));start=time.time()
 with stem.with_suffix('.log').open('w') as f:r=subprocess.run(cmd,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT)
 rows.append(dict(phase=phase,name=name,command=cmd,cwd=str(cwd),seconds=time.time()-start,exit_code=r.returncode));(w/'radix-scan-control-commands.json').write_text(json.dumps(rows,indent=2)+'\n');print(phase,name,r.returncode,round(time.time()-start,2),flush=True)
 if r.returncode and name!='coarse-thin-hand':raise SystemExit(r.returncode)
for case in ['surface','graded-surface']:
 name='armadillo-uniform' if case=='surface' else 'armadillo-graded'
 run('before',name,'examples.vtm_gallery',['--case',case,'--input',w/'scanned-armadillo-input/armadillo.off','--n',4096,'--initial',256,'--subdivisions',1,'--crease-subdivisions',1,'--feature-angle',120,'--corner-angle',45,'--repeats',3,'--output',w/f'radix-before-{name}'])
