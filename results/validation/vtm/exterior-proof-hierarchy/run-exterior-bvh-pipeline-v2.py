"""Wait for the full suite; then serial clean timing and device qualification."""
import json,os,signal,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';python=root/'.venv/bin/python';dest=w/'exterior-bvh-pipeline-v2.json';assert not dest.exists()
start=time.time();suite=w/'exterior-bvh-full-suite-v1.json'
while not suite.exists():
 assert time.time()-start<1000
 time.sleep(2)
s=json.loads(suite.read_text());assert s['exit_code']==0 and '1317 passed' in suite.with_suffix('.log').read_text(),s
jobs=[('timing',[str(python),str(w/'run-exterior-bvh-final-v2.py')],1800),('qualification',[str(python),str(w/'run-exterior-bvh-qualification-v2.py')],6300)]
rows=[]
for name,cmd,timeout in jobs:
 log=w/f'exterior-bvh-pipeline-v2-{name}.log';assert not log.exists();start=time.time()
 with log.open('w') as f:
  p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=p.wait(timeout=timeout)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);p.wait();code=124
 row=dict(name=name,command=cmd,exit_code=code,wall_seconds=time.time()-start,timeout_seconds=timeout)
 rows.append(row);dest.write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True);assert code==0,row
