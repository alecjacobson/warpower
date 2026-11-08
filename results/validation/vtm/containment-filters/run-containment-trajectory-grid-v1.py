"""Untimed exact decision audit over complete recovered solves."""
import gzip,hashlib,json,os,signal,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=w/'containment-trajectory-v1';assert not out.exists();out.mkdir()
rows=[];env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',WARP_CACHE_PATH='/opt/warpower-generated-cache',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.:.work/vtm')
for name,case,extra in [('hand-seed7','thin-hand',['--seed','7']),('hand-seed7-local4','thin-hand',['--seed','7','--local-quality-retries','4']),('hand-seed2','thin-hand',['--seed','2']),('coarse-cavity','cavity',['--subdivisions','4','--crease-subdivisions','4'])]:
 folder=out/name;assert not folder.exists()
 cmd=[str(root/'.venv/bin/python'),str(w/'run-containment-trajectory-oracle-v1.py'),'--case',case,'--n','1000','--initial','256','--quality-rounds','4','--repeats','1',*extra,'--output',str(folder)]
 start=time.time()
 with (out/(name+'.log')).open('w') as log:
  p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=p.wait(timeout=300)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);p.wait();code=124
 row=dict(name=name,command=cmd,exit_code=code,wall_seconds=time.time()-start,timeout_seconds=300)
 for filename in ['result.json','decision-audit.json']:
  path=folder/filename
  if path.exists():row[filename]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),data=json.loads(path.read_text()))
 rows.append(row);(out/'commands.json').write_text(json.dumps(rows,indent=2)+'\n');print(name,code,row.get('decision-audit.json',{}).get('data',{}).get('statistics'),flush=True)
 assert code==0 and row['decision-audit.json']['data']['statistics'][1]==0 and row['result.json']['data']['valid']==1,row
 for filename in ['mesh.vtu','boundary.off']:
  p=folder/filename;raw=p.read_bytes();dest=p.with_suffix(p.suffix+'.gz');assert not dest.exists();dest.write_bytes(gzip.compress(raw,mtime=0));assert gzip.decompress(dest.read_bytes())==raw;p.unlink()
