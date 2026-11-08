"""Independent final meshes and retained-grid checks after all clean timing completes."""
import json,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';dest=w/'larger-prefix-final-independent-v1-commands.json';assert not dest.exists()
commands=json.loads((w/'larger-prefix-final-v1/commands.json').read_text());assert len(commands)==34 and all(r['quota_clean'] for r in commands)
rows=[]
for script in ['check-larger-prefix-final-v1.py','check-larger-prefix-grid-v1.py']:
 cmd=[str(root/'.venv/bin/python'),str(w/script)];log=w/(Path(script).stem+'.log');assert not log.exists();start=time.time()
 with log.open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=1200)
 row=dict(command=cmd,log=str(log),exit_code=r.returncode,wall_seconds=time.time()-start);rows.append(row);dest.write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True);assert r.returncode==0
