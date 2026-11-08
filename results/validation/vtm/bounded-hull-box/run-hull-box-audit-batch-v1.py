"""Old/new support decisions on identical current inputs; timings are diagnostic."""
import gzip,hashlib,json,os,subprocess
from pathlib import Path
w=Path('.work/vtm');out=w/'hull-box-audit-v1';out.mkdir(exist_ok=False);rows=[];archives=[]
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',WARP_CACHE_PATH='/opt/warpower-generated-cache',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.:.work/vtm')
cases=[('torus4096',['--case','torus','--n','4096']),('hand7',['--case','thin-hand','--n','1000','--seed','7']),('plate',['--case','counterbored-plate','--n','1000']),('graded-armadillo',['--case','graded-surface','--n','4096','--input',str(w/'scanned-armadillo-input/armadillo.off'),'--subdivisions','1','--crease-subdivisions','1','--feature-angle','120'])]
for name,args in cases:
 folder=out/name;cmd=['.venv/bin/python',str(w/'run-hull-box-audit-v1.py'),*args,'--initial','256','--quality-rounds','4','--repeats','1','--output',str(folder)]
 with (w/('hull-box-audit-v1-'+name+'.log')).open('w') as f:p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
 row=dict(name=name,command=cmd,exit_code=p.returncode);rows.append(row);(out/'commands.json').write_text(json.dumps(rows,indent=2)+'\n');assert p.returncode==0,row
 audit=json.loads((folder/'decision-audit.json').read_text());assert all(r['statistics'][2:4]==[0,0] for r in audit['instances']);print(name,audit['instances'],flush=True)
 for filename in ['mesh.vtu','boundary.off']:
  raw=folder/filename;b=raw.read_bytes();q=raw.with_suffix(raw.suffix+'.gz');assert not q.exists();q.write_bytes(gzip.compress(b,mtime=0));assert gzip.decompress(q.read_bytes())==b;archives.append(dict(path=str(q),raw_sha256=hashlib.sha256(b).hexdigest()));raw.unlink()
 if name=='graded-armadillo':
  path=folder/'input.off';canonical=w/'scanned-armadillo-input/armadillo.off';assert path.read_bytes()==canonical.read_bytes();path.unlink();os.link(canonical,path)
 (out/'lossless-archives.json').write_text(json.dumps(archives,indent=2)+'\n')
