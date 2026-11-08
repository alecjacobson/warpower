"""Wait until clean GPU timing is terminal, then independently check all saved outputs."""
import json,os,signal,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';python=root/'.venv/bin/python';dest=w/'exterior-bvh-independent-commands-v2.json';assert not dest.exists()
start=time.time();stage=w/'exterior-bvh-pipeline-v2.json'
while not stage.exists():
 assert time.time()-start<2400
 time.sleep(2)
stages=json.loads(stage.read_text());assert stages[0]['name']=='timing' and stages[0]['exit_code']==0
commands=json.loads((w/'exterior-bvh-final-v2/commands.json').read_text());assert len(commands)==28 and all(r['exit_code']==0 and r['quota_clean'] for r in commands)
rows=[]
for name,script in [('final','check-exterior-bvh-final-v2.py'),('loop','check-exterior-bvh-native-loop-v1.py')]:
 if name=='loop':
  ready=w/'exterior-bvh-loop-native-v1-command.json';start=time.time()
  while not ready.exists():
   assert time.time()-start<1500
   time.sleep(2)
  assert json.loads(ready.read_text())['exit_code']==0
 cmd=[str(python),str(w/script)];log=w/f'exterior-bvh-independent-v2-{name}.log';assert not log.exists();start=time.time()
 with log.open('w') as f:
  p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=p.wait(timeout=1500)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);p.wait();code=124
 row=dict(name=name,command=cmd,exit_code=code,wall_seconds=time.time()-start,timeout_seconds=1500)
 rows.append(row);dest.write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True);assert code==0,row
