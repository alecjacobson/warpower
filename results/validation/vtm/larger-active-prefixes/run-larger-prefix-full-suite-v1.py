"""Record the complete production suite against an immutable runtime commit."""
import hashlib,json,re,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';log=w/'larger-prefix-full-suite-v1.log';result=w/'larger-prefix-full-suite-v1.json'
assert not log.exists() and not result.exists();assert not subprocess.check_output(['git','status','--porcelain'])
source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();cmd=[str(root/'.venv/bin/python'),'-m','pytest','-q','-p','no:cacheprovider','tests'];start=time.time()
with log.open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=1200)
text=log.read_text();matches=re.findall(r'(\d+) passed',text)
meta=dict(command=cmd,source=source,exit_code=r.returncode,wall_seconds=time.time()-start,tests=int(matches[-1]) if matches else None,log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
result.write_text(json.dumps(meta,indent=2)+'\n');print(meta,flush=True);print(text[-3000:],flush=True);assert r.returncode==0 and meta['tests']==1292
