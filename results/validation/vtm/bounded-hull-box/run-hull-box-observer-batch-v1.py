"""Sequential actual hull traversal observations; not performance timings."""
import gzip,hashlib,json,os,subprocess
from pathlib import Path
w=Path('.work/vtm');out=w/'hull-box-observer-v1';out.mkdir(exist_ok=False);rows=[];archives=[]
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',WARP_CACHE_PATH='/opt/warpower-generated-cache',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.:.work/vtm')
for name,args in [('torus4096',['--case','torus','--n','4096']),('hand7',['--case','thin-hand','--n','1000','--seed','7']),('plate',['--case','counterbored-plate','--n','1000']),('graded-armadillo',['--case','graded-surface','--n','4096','--input',str(w/'scanned-armadillo-input/armadillo.off')])]:
 folder=out/name;cmd=['.venv/bin/python',str(w/'run-hull-box-observer-v1.py'),*args,'--initial','256','--quality-rounds','4','--repeats','1','--output',str(folder)]
 with (w/('hull-box-observer-v1-'+name+'.log')).open('w') as f:p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
 row=dict(name=name,command=cmd,exit_code=p.returncode);rows.append(row);(out/'commands.json').write_text(json.dumps(rows,indent=2)+'\n');assert p.returncode==0,row
 print(name,json.loads((folder/'observer.json').read_text())['instances'],flush=True)
 for name in ['mesh.vtu','boundary.off']:
  raw=folder/name;b=raw.read_bytes();q=raw.with_suffix(raw.suffix+'.gz');assert not q.exists();q.write_bytes(gzip.compress(b,mtime=0));assert gzip.decompress(q.read_bytes())==b;archives.append(dict(path=str(q),raw_sha256=hashlib.sha256(b).hexdigest()));raw.unlink()
 (out/'lossless-archives.json').write_text(json.dumps(archives,indent=2)+'\n')
