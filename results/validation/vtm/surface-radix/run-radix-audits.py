import os,json,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';python=root/'.venv/bin/python';env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.');rows=[]
def run(name,cmd):
 cmd=list(map(str,cmd));start=time.time()
 with (w/f'radix-{name}.log').open('w') as f:r=subprocess.run(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
 rows.append(dict(name=name,command=cmd,exit_code=r.returncode,seconds=time.time()-start));(w/'radix-audit-commands.json').write_text(json.dumps(rows,indent=2)+'\n');print(name,r.returncode,round(time.time()-start,3),flush=True)
 if r.returncode:raise SystemExit(r.returncode)
run('full-suite',[python,'-m','pytest','-q'])
mem=['/usr/local/cuda-12.9/bin/compute-sanitizer','--tool','memcheck','--padding','32','--error-exitcode','99']
run('memcheck-surface',mem+[python,'-m','benchmarks.memcheck','--stack-bytes','131072','--','-q','tests/test_surface_radix.py','tests/test_domain.py','tests/test_sampling.py'])
run('memcheck-mesher',mem+[python,'-m','benchmarks.memcheck','--stack-bytes','131072','--','-q','tests/test_mesher.py::test_complete_pipeline_samples_grows_fits_and_preserves_domain_topology','tests/test_mesher.py::test_full_device_producer_mesher_consumer_with_changing_input_and_recovery','tests/test_mesher.py::test_mesher_composes_inside_an_outer_device_controlled_loop'])
run('domain-lifetime',mem+['--check-api-memory-access','yes','--track-stream-ordered-races','all',python,w/'radix-domain-lifetime.py','--queries','4096','--repeats','3','--output',w/'radix-domain-lifetime.json'])
for name,module,args,scope in [
 ('full-native','benchmarks.mesher',['--n','1000','--repeats','3'],'Three complete fresh resident torus1000 solves with spatial radix surface traversal. Setup, warmup and readback excluded; one final synchronization.'),
 ('domain-native','benchmarks.domain',['--queries','100000','--repeats','3'],'Three changing-input surface deformation, refit, query and consumer replays with spatial radix topology; one final synchronization.')]:
 stem=w/f'radix-{name}'
 run(name,['nsys','profile','--trace=cuda','--cuda-graph-trace=node','--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=true','-o',stem,python,'-m',module]+args+['--profile','--dot',stem.with_suffix('.dot'),'--output',stem.with_suffix('.json')])
 run(name+'-export',['nsys','export','--type','sqlite','--force-overwrite=true','--output',stem.with_suffix('.sqlite'),stem.with_suffix('.nsys-rep')])
 run(name+'-audit',[python,'-m','benchmarks.graph_capture','--analyze',stem.with_suffix('.sqlite'),'--dot',stem.with_suffix('.dot'),'--scope',scope,'--output',w/f'radix-{name}-audit.json'])
run('mesher-loop',[python,'-m','benchmarks.mesher_loop','--output',w/'radix-mesher-loop.json'])
