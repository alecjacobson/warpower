"""Native audits after independent and instrumented checks and retained controls."""
import json,os,signal,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';python=root/'.venv/bin/python';rows=[]
assert (w/'large-reuse-lifetime-analysis.json').exists()
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.')
def run(name,cmd,timeout=300):
 cmd=list(map(str,cmd));path=w/f'large-reuse-{name}.log';assert not path.exists(),path
 start=time.time()
 with path.open('w') as log:
  process=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=process.wait(timeout=timeout)
  except subprocess.TimeoutExpired:
   os.killpg(process.pid,signal.SIGTERM)
   try:process.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
   code=124
 rows.append(dict(name=name,command=cmd,exit_code=code,wall_seconds=time.time()-start,timeout_seconds=timeout));(w/'large-reuse-native-adoption-commands.json').write_text(json.dumps(rows,indent=2)+'\n')
 print(name,code,rows[-1]['wall_seconds'],flush=True);assert code==0,rows[-1]
for name,script in [('full','large-reuse-full-native.py'),('loop','large-reuse-loop.py')]:
 stem=w/f'large-reuse-{name}-native'
 cmd=['nsys','profile','--trace=cuda','--cuda-graph-trace=node','--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=true','-o',stem,python,w/script,'--dot',stem.with_suffix('.dot'),'--output',stem.with_suffix('.json')]
 if name=='loop':cmd.append('--profile')
 run(name+'-native',cmd)
run('native-adoption-analysis',[python,w/'analyze-large-reuse-audit.py'],180)
