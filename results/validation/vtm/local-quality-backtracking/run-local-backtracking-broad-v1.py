"""Paired broad pilots for direct blocking without midpoint correction."""
import gzip,hashlib,json,os,signal,subprocess,time
from pathlib import Path
w=Path('.work/vtm');out=w/'local-backtracking-broad-v1';assert not out.exists();out.mkdir();rows=[]
cases=[('torus1000','torus',1000,[]),('torus4096','torus',4096,[]),('plate','counterbored-plate',1000,[]),('cavity','cavity',1000,['--subdivisions','20','--crease-subdivisions','20']),('disconnected','disconnected',1000,['--subdivisions','20','--crease-subdivisions','20'])]
for name,case,n,extra in cases:
 for mode in ['baseline','candidate']:
  folder=out/(name+'-'+mode)
  entry=['-m','examples.vtm_gallery'] if mode=='baseline' else [str(w/'local_backtracking_full_v2.py'),'--local-midpoint','0']
  cmd=[str(Path.cwd()/'.venv/bin/python'),*entry,'--case',case,'--n',str(n),'--repeats','2',*extra,'--output',str(folder)]
  start=time.time()
  with folder.with_suffix('.log').open('w') as log:
   p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   try:code=p.wait(timeout=180)
   except subprocess.TimeoutExpired:
    os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=10);code=124
  row=dict(case=name,mode=mode,command=cmd,exit_code=code,wall_seconds=time.time()-start,archives=[])
  if (folder/'result.json').exists():
   d=json.loads((folder/'result.json').read_text());row.update({k:d[k] for k in ['valid','error','counts','steps','moves','quality','median_additional_replay_ms']})
  if (folder/'pilot.json').exists():row['pilot']=json.loads((folder/'pilot.json').read_text())
  for p in [folder/'mesh.vtu',folder/'boundary.off']:
   if p.exists():
    raw=p.read_bytes();dest=p.with_suffix(p.suffix+'.gz');dest.write_bytes(gzip.compress(raw,mtime=0));assert gzip.decompress(dest.read_bytes())==raw
    row['archives'].append(dict(original=str(p),path=str(dest),raw_sha256=hashlib.sha256(raw).hexdigest(),sha256=hashlib.sha256(dest.read_bytes()).hexdigest()));p.unlink()
  rows.append(row);(out/'commands.json').write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True)
  assert code in [0,1] and (folder/'result.json').exists()
