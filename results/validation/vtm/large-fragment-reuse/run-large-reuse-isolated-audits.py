"""Fresh-process memory gates after the incomplete combined sanitizer run."""
import json,os,signal,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';python=root/'.venv/bin/python'
assert '650 passed' in (w/'large-reuse-full-suite.log').read_text()
assert len(json.loads((w/'large-reuse-final-independent.json').read_text())['cases'])==10
assert (w/'large-reuse-combined-interruption.json').exists()
nodes=[line for line in (w/'large-reuse-collected-tests.log').read_text().splitlines() if line.startswith('tests/test_large_reuse.py::')]
assert len(nodes)==7
base_env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.')
rows=[]
def run(name,cmd,env=base_env,timeout=600):
 cmd=list(map(str,cmd));logpath=w/f'large-reuse-{name}.log';assert not logpath.exists(),logpath
 start=time.time();timed_out=False
 with logpath.open('w') as log:
  process=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=process.wait(timeout=timeout)
  except subprocess.TimeoutExpired:
   timed_out=True;os.killpg(process.pid,signal.SIGTERM)
   try:process.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
   code=124
 row=dict(name=name,command=cmd,exit_code=code,timed_out=timed_out,timeout_seconds=timeout,wall_seconds=time.time()-start,disable_pool=env.get('WARPOWER_AUDIT_DISABLE_POOL')=='1')
 rows.append(row);(w/'large-reuse-isolated-audit-commands.json').write_text(json.dumps(rows,indent=2)+'\n')
 print(name,code,row['wall_seconds'],flush=True);assert code==0,row
 return logpath
mem=['/usr/local/cuda-12.9/bin/compute-sanitizer','--tool','memcheck','--padding','32','--error-exitcode','99']
for kind in ['memcheck','lifetime']:
 paths=[]
 # Start with the case at which the combined run stopped making progress.
 for i in [3,0,1,2,4,5,6]:
  env=dict(base_env)
  args=mem[:]
  if kind=='lifetime':
   env['WARPOWER_AUDIT_DISABLE_POOL']='1';args+=['--check-api-memory-access','yes','--track-stream-ordered-races','all']
  path=run(f'{kind}-isolated-{i}',args+[python,w/'large-reuse-isolated-test.py',nodes[i]],env)
  s=path.read_text();assert '1 passed' in s and 'ERROR SUMMARY: 0 errors' in s,(kind,i)
  paths.append(path)
 aggregate=w/f'large-reuse-{kind}.log'
 aggregate.write_text('Seven unchanged tests, each in a fresh instrumented process. All seven report1passed andzeroerrors; this is not a combined-process pass.\n\n'+'\n\n'.join(str(p.name)+'\n'+p.read_text() for p in paths))
for name,script in [('full','large-reuse-full-native.py'),('loop','large-reuse-loop.py')]:
 stem=w/f'large-reuse-{name}-native'
 cmd=['nsys','profile','--trace=cuda','--cuda-graph-trace=node','--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=true','-o',stem,python,w/script,'--dot',stem.with_suffix('.dot'),'--output',stem.with_suffix('.json')]
 if name=='loop':cmd.append('--profile')
 run(name+'-native',cmd,timeout=300)
run('native-adoption-analysis',[python,w/'analyze-large-reuse-audit.py'],timeout=180)
assert len(rows)==17
