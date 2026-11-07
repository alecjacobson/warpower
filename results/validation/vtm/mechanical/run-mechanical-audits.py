import os,json,subprocess,time
from pathlib import Path
w=Path('.work/vtm');python=Path('.venv/bin/python');env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.');rows=[]
def run(name,cmd,expect=0):
 cmd=list(map(str,cmd));start=time.time()
 with (w/f'mechanical-{name}.log').open('w') as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
 rows.append(dict(name=name,command=cmd,exit_code=r.returncode,expected_exit_code=expect,seconds=time.time()-start));(w/'mechanical-audit-commands.json').write_text(json.dumps(rows,indent=2)+'\n');print(name,r.returncode,flush=True)
 if r.returncode!=expect:raise SystemExit(r.returncode or 98)
mem=['/usr/local/cuda-12.9/bin/compute-sanitizer','--tool','memcheck','--padding','32','--error-exitcode','99']
run('memcheck-ordinary',mem+[python,'-m','benchmarks.memcheck','--stack-bytes','131072','--','-q','tests/test_vtm_examples.py','tests/test_feature_quality.py'])
run('memcheck-afterfree',mem+['--check-api-memory-access','yes','--track-stream-ordered-races','use-after-free',python,'-m','benchmarks.memcheck','--stack-bytes','131072','--disable-mempool','--','-q','tests/test_vtm_examples.py','tests/test_feature_quality.py'])
for name,n,depth,tracking,expect in [('radix-conditional',8192,1,'all',99),('radix-top',8192,0,'all',0),('radix-afterfree',8192,1,'use-after-free',0)]:
 run(name,mem+['--track-stream-ordered-races',tracking,python,'-m','benchmarks.radix_capture','--disable-mempool','--n',str(n),'--depth',str(depth)],expect)
run('native',['nsys','profile','--trace=cuda','--cuda-graph-trace=node','--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=true','-o',w/'mechanical-native',python,w/'mechanical-native.py'])
run('native-export',['nsys','export','--type','sqlite','--force-overwrite=true','--output',w/'mechanical-native.sqlite',w/'mechanical-native.nsys-rep'])
run('native-audit',[python,'-m','benchmarks.graph_capture','--analyze',w/'mechanical-native.sqlite','--dot',w/'mechanical-native.dot','--scope','Three complete fresh resident counterbored-plate1000 solves, 16/16 quadrature and actual 45-degree features. Setup, warmup and readback excluded; one final synchronization.','--output',w/'mechanical-native-audit.json'])
