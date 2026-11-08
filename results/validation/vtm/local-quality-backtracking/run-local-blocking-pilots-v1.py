"""Bounded sequential fresh gallery probes; retain all outcomes and lossless exports."""
import gzip,hashlib,json,os,signal,subprocess,time
from pathlib import Path
w=Path('.work/vtm');out=w/'local-backtracking-pilots-v1';assert not out.exists();out.mkdir();rows=[]
for version,midpoint in [(2,1),(3,1),(3,0)]:
 stem=out/f'dense-hand-v{version}-midpoint{midpoint}'
 cmd=[str(Path('.venv/bin/python').resolve()),str(w/f'local_backtracking_full_v{version}.py'),'--local-midpoint',str(midpoint),'--case','thin-hand','--n','1000','--subdivisions','20','--crease-subdivisions','20','--repeats','1','--output',str(stem)]
 start=time.time()
 with stem.with_suffix('.log').open('w') as log:
  p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=p.wait(timeout=180)
  except subprocess.TimeoutExpired:
   os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=10);code=124
 row=dict(version=version,midpoint=midpoint,command=cmd,exit_code=code,wall_seconds=time.time()-start)
 if (stem/'result.json').exists():
  r=json.loads((stem/'result.json').read_text());row.update({k:r[k] for k in ['valid','error','steps','counts','quality','median_additional_replay_ms']})
 if (stem/'pilot.json').exists():row['pilot']=json.loads((stem/'pilot.json').read_text())
 row['archives']=[]
 for p in [stem/'mesh.vtu',stem/'boundary.off']:
  if p.exists():
   raw=p.read_bytes();archive=p.with_suffix(p.suffix+'.gz');archive.write_bytes(gzip.compress(raw,mtime=0));assert gzip.decompress(archive.read_bytes())==raw
   row['archives'].append(dict(original=str(p),path=str(archive),raw_sha256=hashlib.sha256(raw).hexdigest(),sha256=hashlib.sha256(archive.read_bytes()).hexdigest()));p.unlink()
 rows.append(row);(out/'commands.json').write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True)
 assert code in [0,1] and (stem/'pilot.json').exists()
