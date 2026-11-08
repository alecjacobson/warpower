"""Offline CPU geometry checks after every private clean timing job is terminal."""
import subprocess,json,time
from pathlib import Path
w=Path('.work/vtm');dest=w/'prepared-new-cell-independent-v2-commands.json';assert not dest.exists();assert json.loads((w/'prepared-new-cell-simple-grid-retry-v1.json').read_text())['quota_clean'];rows=[]
for name in ['check-prepared-new-cell-pilot-v2.py','check-prepared-new-cell-grid-v2.py','check-prepared-new-cell-simple-guards-v1.py']:
 cmd=['.venv/bin/python',str(w/name)];start=time.time()
 with (w/(name[:-3]+'.log')).open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=900)
 row=dict(command=cmd,exit_code=p.returncode,wall_seconds=time.time()-start);rows.append(row);dest.write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True);assert p.returncode==0
