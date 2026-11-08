"""Retry only the grid wrapper after preserving the system-Python import failure."""
from pathlib import Path
import hashlib,json,subprocess,time
w=Path('.work/vtm').absolute();out=w/'prepared-new-cell-simple-guards-v1';dest=w/'prepared-new-cell-simple-grid-retry-v1.json';assert not dest.exists();assert not (out/'retained-grid.json').exists()
cmd=[str(Path.cwd()/'.venv/bin/python'),str(w/'bench-prepared-new-cell-grid-v1.py'),'--candidate','--output',str(out/'retained-grid.json')];start=time.time()
with (out/'retained-grid-retry-v1.log').open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=300)
r=dict(command=cmd,exit_code=p.returncode,wall_seconds=time.time()-start,prior_failure=dict(driver_exit_code=1,cause='Path.resolve followed the venv Python symlink to system Python; NumPy import failed before setup/GPU work. Original child exit status was not recorded by the failed driver.',log=str(out/'retained-grid.log'),log_sha256=hashlib.sha256((out/'retained-grid.log').read_bytes()).hexdigest()),source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
dest.write_text(json.dumps(r,indent=2)+'\n');assert p.returncode==0
p=out/'retained-grid.json';d=json.loads(p.read_text());assert d['error']==0 and d['reuse_accepted']==[1];assert all(v['throttled_periods']==0 for k in ['cpu_quota_delta','initial_cpu_quota_delta'] for v in d[k].values());r.update(result=str(p),result_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),median_ms=d['median_ms'],quota_clean=True);dest.write_text(json.dumps(r,indent=2)+'\n');print(r,flush=True)
