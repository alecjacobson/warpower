"""Run independent CPU qualification only after all eight clean timing jobs."""
import hashlib,json,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';rows=json.loads((w/'larger-prefix-pilot-v1/commands.json').read_text())
assert len(rows)==8 and all(r['exit_code']==0 and r['quota_clean'] for r in rows)
p=w/'larger-prefix-pilot-independent-v1-command.json';log=w/'larger-prefix-pilot-independent-v1.log';assert not p.exists() and not log.exists()
cmd=[str(root/'.venv/bin/python'),str(w/'check-larger-prefix-pilot-v1.py')];start=time.time()
with log.open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=600)
p.write_text(json.dumps(dict(command=cmd,exit_code=r.returncode,wall_seconds=time.time()-start,log_sha256=hashlib.sha256(log.read_bytes()).hexdigest()),indent=2)+'\n');print(r.returncode,flush=True);raise SystemExit(r.returncode)
