"""Pinned complete-solve pairs; retain geometry and every timing attempt."""
import gzip,hashlib,json,os,signal,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=w/'containment-filter-pilot-v1';base=w/'containment-baseline';python=root/'.venv/bin/python'
assert not out.exists();assert '26 passed' in (w/'containment-filter-focused-v2.log').read_text()
assert not subprocess.check_output(['git','status','--porcelain'])
sources={p:subprocess.check_output(['git','rev-parse','HEAD'],cwd=c,text=True).strip() for p,c in [('before',base),('after',root)]}
assert sources['before']=='96027fbb07625b144106c2c36d92870b3a7402db'
out.mkdir();rows=[];archives=[]
env=dict(os.environ,PYTHONPATH='.',PYTHONDONTWRITEBYTECODE='1',WARP_CACHE_PATH='/opt/warpower-generated-cache',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4')
cases=[('hand-seed7','thin-hand',1000,['--seed','7']),('hand-seed7-local4','thin-hand',1000,['--seed','7','--local-quality-retries','4']),('hand-seed2','thin-hand',1000,['--seed','2']),('coarse-cavity','cavity',1000,['--subdivisions','4','--crease-subdivisions','4']),('torus4096','torus',4096,[])]
for name,case,n,extra in cases:
 for phase,cwd in [('before',base),('after',root)]:
  folder=out/(name+'-'+phase);cmd=[str(python),'-m','examples.vtm_gallery','--case',case,'--n',str(n),'--initial','256','--quality-rounds','4','--repeats','3',*extra,'--output',str(folder)]
  started=time.time()
  with (out/(name+'-'+phase+'.log')).open('w') as stream:
   p=subprocess.Popen(cmd,cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
   try:code=p.wait(timeout=300)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);p.wait();code=124
  row=dict(name=name,phase=phase,source=sources[phase],command=cmd,cwd=str(cwd),exit_code=code,wall_seconds=time.time()-started,timeout_seconds=300)
  path=folder/'result.json'
  if path.exists():
   d=json.loads(path.read_text());row.update(result=str(path),result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),valid=d['valid'],error=d['error'],counts=d['counts'],median_ms=d['median_additional_replay_ms'],quota_clean=all(v['throttled_periods']==0 for k in ['first_replay_cpu_quota_delta','additional_replays_cpu_quota_delta'] for v in d[k].values()))
  rows.append(row);(out/'commands.json').write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True)
  assert code==0 and row['valid']==1 and row['error']==0 and row['quota_clean'],row
  for name in ['mesh.vtu','boundary.off']:
   raw=folder/name;data=raw.read_bytes();packed=raw.with_suffix(raw.suffix+'.gz');packed.write_bytes(gzip.compress(data,mtime=0));assert gzip.decompress(packed.read_bytes())==data
   archives.append(dict(original=str(raw),path=str(packed),raw_sha256=hashlib.sha256(data).hexdigest(),sha256=hashlib.sha256(packed.read_bytes()).hexdigest()))
   (out/'lossless-archives.json').write_text(json.dumps(archives,indent=2)+'\n');raw.unlink()
assert len(rows)==10
