"""Start CPU validation only after all clean GPU timing jobs have terminated."""
import json,os,signal,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';python=root/'.venv/bin/python';dest=w/'tangential-production-independent-v1.json';assert not dest.exists()
pipeline=w/'tangential-production-pipeline-v1.json';start=time.time()
while not pipeline.exists():
 assert time.time()-start<2000
 time.sleep(2)
d=json.loads(pipeline.read_text());assert d[0]['name']=='timing' and d[0]['exit_code']==0,d
rows=[]
for name,script,timeout in [('full','check-tangential-production-final-v1.py',1200),('loop','check-tangential-production-loop-native-v1.py',600)]:
 if name=='loop':
  start=time.time();ready=w/'tangential-production-loop-native-v1-command.json'
  while not ready.exists():
   assert time.time()-start<1000
   p=json.loads(pipeline.read_text());assert all(r['exit_code']==0 for r in p),p
   time.sleep(2)
  assert json.loads(ready.read_text())['exit_code']==0
 cmd=[str(python),str(w/script)];log=w/f'tangential-production-independent-v1-{name}.log';assert not log.exists();start=time.time()
 with log.open('w') as f:
  p=subprocess.Popen(cmd,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONPATH='.'),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=p.wait(timeout=timeout)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);p.wait();code=124
 row=dict(name=name,command=cmd,exit_code=code,wall_seconds=time.time()-start,timeout_seconds=timeout)
 rows.append(row);dest.write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True);assert code==0,row
