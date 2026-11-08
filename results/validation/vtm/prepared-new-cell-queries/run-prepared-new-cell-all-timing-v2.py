from pathlib import Path
import subprocess,json,time
w=Path('.work/vtm');record=w/'prepared-new-cell-all-timing-v2.json';assert not record.exists();rows=[]
for name in ['run-prepared-new-cell-pilot-v2.py','run-prepared-new-cell-grid-v2.py']:
 start=time.time();p=subprocess.run(['.venv/bin/python',str(w/name)]);rows.append(dict(driver=name,exit_code=p.returncode,wall_seconds=time.time()-start));record.write_text(json.dumps(rows,indent=2)+'\n');assert p.returncode==0
