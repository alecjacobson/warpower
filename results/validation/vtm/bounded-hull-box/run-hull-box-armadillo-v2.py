"""Repeat only Armadillo with its established quadrature and feature settings."""
import gzip,hashlib,json,os,subprocess
from pathlib import Path
w=Path('.work/vtm');out=w/'hull-box-observer-v2';out.mkdir(exist_ok=False);folder=out/'graded-armadillo'
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',WARP_CACHE_PATH='/opt/warpower-generated-cache',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.:.work/vtm')
cmd=['.venv/bin/python',str(w/'run-hull-box-observer-v1.py'),'--case','graded-surface','--n','4096','--input',str(w/'scanned-armadillo-input/armadillo.off'),'--subdivisions','1','--crease-subdivisions','1','--feature-angle','120','--initial','256','--quality-rounds','4','--repeats','1','--output',str(folder)]
with (w/'hull-box-observer-v2-graded-armadillo.log').open('w') as f:p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
(out/'command.json').write_text(json.dumps(dict(command=cmd,exit_code=p.returncode,scope=__doc__,prior_attempt='v1 used the gallery defaults, instead of established Armadillo settings. It rejected initialization with error1 before any hull call; retained separately.'),indent=2)+'\n');assert p.returncode==0
archives=[]
for name in ['mesh.vtu','boundary.off']:
 raw=folder/name;b=raw.read_bytes();q=raw.with_suffix(raw.suffix+'.gz');assert not q.exists();q.write_bytes(gzip.compress(b,mtime=0));assert gzip.decompress(q.read_bytes())==b;archives.append(dict(path=str(q),raw_sha256=hashlib.sha256(b).hexdigest()));raw.unlink()
(out/'lossless-archives.json').write_text(json.dumps(archives,indent=2)+'\n')
print(json.loads((folder/'observer.json').read_text())['instances'],flush=True)
