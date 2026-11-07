import os,json,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';python=root/'.venv/bin/python';env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.');rows=[]
def run(name,cmd):
 cmd=list(map(str,cmd));start=time.time()
 with (w/f'boundary-{name}.log').open('w') as f:r=subprocess.run(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
 rows.append(dict(name=name,command=cmd,exit_code=r.returncode,seconds=time.time()-start));(w/'boundary-audit-commands.json').write_text(json.dumps(rows,indent=2)+'\n');print(name,r.returncode,round(time.time()-start,3),flush=True)
 if r.returncode:raise SystemExit(r.returncode)
run('full-suite',[python,'-m','pytest','-q'])
mem=['/usr/local/cuda-12.9/bin/compute-sanitizer','--tool','memcheck','--padding','32','--error-exitcode','99']
run('memcheck',mem+[python,'-m','benchmarks.memcheck','--stack-bytes','131072','--','-q','tests/test_boundary.py'])
run('lifetime',mem+['--check-api-memory-access','yes','--track-stream-ordered-races','all',python,'-m','benchmarks.memcheck','--stack-bytes','131072','--disable-mempool','--','-q','tests/test_boundary.py'])
name='full-native';stem=w/f'boundary-{name}'
run(name,['nsys','profile','--trace=cuda','--cuda-graph-trace=node','--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=true','-o',stem,python,'-m','benchmarks.mesher','--n','1000','--repeats','3','--profile','--dot',stem.with_suffix('.dot'),'--output',stem.with_suffix('.json')])
run(name+'-export',['nsys','export','--type','sqlite','--force-overwrite=true','--output',stem.with_suffix('.sqlite'),stem.with_suffix('.nsys-rep')])
run(name+'-audit',[python,'-m','benchmarks.graph_capture','--analyze',stem.with_suffix('.sqlite'),'--dot',stem.with_suffix('.dot'),'--scope','Three complete fresh torus1000 solves with 32-thread boundary assignment; one final synchronization, setup and readback excluded.','--output',w/f'boundary-{name}-audit.json'])
run('mesher-loop',[python,'-m','benchmarks.mesher_loop','--output',w/'boundary-mesher-loop.json'])
run('independent',[python,w/'check-boundary-results.py'])
