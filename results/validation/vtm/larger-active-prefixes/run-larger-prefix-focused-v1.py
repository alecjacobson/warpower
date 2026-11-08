"""Record production routing regression collection and focused correctness result."""
import hashlib,json,re,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';python=str(root/'.venv/bin/python')
files=['tests/test_larger_prefixes.py','tests/test_active_delaunay.py','tests/test_active_sites.py','tests/test_large_reuse.py']
collection=w/'larger-prefix-collection-v1.log';log=w/'larger-prefix-focused-v1.log';result=w/'larger-prefix-focused-v1.json'
assert not any(p.exists() for p in [collection,log,result])
with collection.open('w') as f:subprocess.run([python,'-m','pytest','--collect-only','-q','-p','no:cacheprovider',*files],stdout=f,stderr=subprocess.STDOUT,check=True)
cmd=[python,'-m','pytest','-q','-p','no:cacheprovider',*files];start=time.time()
with log.open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=600)
meta=dict(command=cmd,parent=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),exit_code=r.returncode,wall_seconds=time.time()-start,sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*map(Path,files),Path('warpower/_active_delaunay.py'),Path('warpower/_vtm.py'),log,collection]})
result.write_text(json.dumps(meta,indent=2)+'\n');print(log.read_text()[-3500:],flush=True);raise SystemExit(r.returncode)
