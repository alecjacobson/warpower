"""CPU validation only after the 26 clean production jobs are complete."""
import json,subprocess,time
from pathlib import Path
w=Path('.work/vtm');dest=w/'prepared-new-cell-final-independent-v1-commands.json';assert not dest.exists();runs=json.loads((w/'prepared-new-cell-final-v1/commands.json').read_text());assert len(runs)==26 and all(r['exit_code']==0 and r['quota_clean'] for r in runs);rows=[]
for name in ['check-prepared-new-cell-final-v1.py','check-prepared-new-cell-grid-v1.py']:
 cmd=['.venv/bin/python',str(w/name)];start=time.time()
 with (w/(name[:-3]+'.log')).open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=900)
 row=dict(command=cmd,exit_code=p.returncode,wall_seconds=time.time()-start);rows.append(row);dest.write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True);assert p.returncode==0
