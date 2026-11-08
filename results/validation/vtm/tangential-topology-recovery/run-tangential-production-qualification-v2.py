"""Reuse completed native gates and retry only the incomplete memory gate."""
import json,os,signal,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';python=root/'.venv/bin/python';dest=w/'tangential-production-qualification-v2.json';assert not dest.exists()
old=json.loads((w/'tangential-production-qualification-v1.json').read_text());assert len(old)==3 and old[-1]['name']=='memory' and old[-1]['exit_code']==1
rows=[dict(r,reused_from='tangential-production-qualification-v1.json') for r in old[:2]];assert all(r['exit_code']==0 for r in rows)
dest.write_text(json.dumps(rows,indent=2)+'\n')
cmd=[str(python),str(w/'run-tangential-production-memory-v2.py')];log=w/'tangential-production-qualification-v2-memory.log';assert not log.exists();start=time.time();timeout=6500
with log.open('w') as f:
 p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 try:code=p.wait(timeout=timeout)
 except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);p.wait();code=124
row=dict(name='memory',command=cmd,exit_code=code,wall_seconds=time.time()-start,timeout_seconds=timeout)
rows.append(row);dest.write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True);assert code==0,row
