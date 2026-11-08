"""Sequential applicability observations; instrumentation is not timing evidence."""
import gzip,json,os,subprocess
from pathlib import Path
w=Path('.work/vtm');out=w/'axis-hull-observer-v1';dest=out/'batch-commands.json';assert not dest.exists();rows=[]
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',WARP_CACHE_PATH='/opt/warpower-generated-cache',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.:.work/vtm')
for name,case,n in [('plate','counterbored-plate',1000),('cavity','cavity',1000),('torus4096','torus',4096)]:
 folder=out/name;assert not folder.exists();cmd=['.venv/bin/python',str(w/'run-axis-hull-observer-v1.py'),'--case',case,'--n',str(n),'--initial','256','--quality-rounds','4','--repeats','1','--output',str(folder)]
 with (w/('axis-hull-observer-v1-'+name+'.log')).open('w') as f:p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
 row=dict(name=name,command=cmd,exit_code=p.returncode);rows.append(row);dest.write_text(json.dumps(rows,indent=2)+'\n');assert p.returncode==0,row;print(name,json.loads((folder/'observer.json').read_text())['instances'],flush=True)
 for raw in [folder/'mesh.vtu',folder/'boundary.off']:
  b=raw.read_bytes();q=raw.with_suffix(raw.suffix+'.gz');q.write_bytes(gzip.compress(b,mtime=0));assert gzip.decompress(q.read_bytes())==b;raw.unlink()
