"""Production full suite, with pinned source and terminal status."""
import hashlib,json,os,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm'
assert '121 passed' in (w/'large-stackless-production-focused-v1.log').read_text()
log=w/'large-stackless-full-suite-v1.log';assert not log.exists()
cmd=[str(root/'.venv/bin/python'),'-m','pytest','-q']
start=time.time()
with log.open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
row=dict(command=cmd,exit_code=r.returncode,wall_seconds=time.time()-start,source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
(w/'large-stackless-full-suite-v1.json').write_text(json.dumps(row,indent=2)+'\n');print(row,flush=True)
assert r.returncode==0 and '1227 passed' in log.read_text()
