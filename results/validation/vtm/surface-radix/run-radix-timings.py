import os,json,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';baseline=w/'extraction-geometry';rows=[];env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.')
def run(phase,name,module,args):
 cwd=baseline if phase=='before' else root;stem=w/f'radix-{phase}-{name}';cmd=[str(root/'.venv/bin/python'),'-m',module]+list(map(str,args));start=time.time()
 with stem.with_suffix('.log').open('w') as f:r=subprocess.run(cmd,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT)
 rows.append(dict(phase=phase,name=name,command=cmd,cwd=str(cwd),seconds=time.time()-start,exit_code=r.returncode));(w/'radix-remaining-timing-commands.json').write_text(json.dumps(rows,indent=2)+'\n');print(phase,name,r.returncode,round(time.time()-start,2),flush=True)
 if r.returncode and name!='coarse-thin-hand':raise SystemExit(r.returncode)
for n in [4096]:
 for phase in ['before','after']:
  stem=w/f'radix-{phase}-torus{n}';run(phase,f'torus{n}','benchmarks.mesher',['--n',n,'--repeats',3,'--save-mesh',stem.with_suffix('.npz'),'--output',stem.with_suffix('.json')])
for case,quad in [('graded-torus',4),('thin-hand',20),('cavity',20),('disconnected',20)]:
 for phase in ['before','after']:run(phase,case,'examples.vtm_gallery',['--case',case,'--n',1000,'--subdivisions',quad,'--crease-subdivisions',quad,'--repeats',3,'--output',w/f'radix-{phase}-{case}'])
for name,args in [('delaunay1000',['--n',1000]),('delaunay100000',['--n',100000]),('delaunay-grid',['--case','grid','--grid-side',4]),('delaunay-fitted-hand',['--case','fitted-hand'])]:
 for phase in ['before','after']:run(phase,name,'benchmarks.delaunay_recovery',args+['--repeats',9,'--output',w/f'radix-{phase}-{name}.json'])
for phase in ['before','after']:run(phase,'diagram100000','benchmarks.run',['--backend','warpower','--n',100000,'--warmup',5,'--repeats',30,'--output',w/f'radix-{phase}-diagram100000.json'])
for case in ['surface','graded-surface']:
 name='armadillo-uniform' if case=='surface' else 'armadillo-graded'
 run('after',name,'examples.vtm_gallery',['--case',case,'--input',w/'scanned-armadillo-input/armadillo.off','--n',4096,'--initial',256,'--subdivisions',1,'--crease-subdivisions',1,'--feature-angle',120,'--corner-angle',45,'--repeats',3,'--output',w/f'radix-after-{name}'])
for case in ['thin-hand','cavity','disconnected']:
 run('after','coarse-'+case,'examples.vtm_gallery',['--case',case,'--n',1000,'--subdivisions',4,'--crease-subdivisions',4,'--repeats',0,'--output',w/f'radix-after-coarse-{case}'])

run('after','warm-setup','benchmarks.mesher',['--n',1000,'--repeats',1,'--output',w/'radix-after-warm-setup.json'])
