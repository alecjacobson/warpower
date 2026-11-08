"""Sequential before/after complete VTM timings; failures retained, no filtering."""
import gzip,hashlib,json,os,subprocess
from pathlib import Path
w=Path('.work/vtm');out=w/'bounded-hull-box-pilot-v3';out.mkdir(exist_ok=False);rows=[]
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',WARP_CACHE_PATH='/opt/warpower-generated-cache',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.:.work/vtm')
source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
assert not subprocess.check_output(['git','diff','--name-only'],text=True).strip()
for name,args in [('plate',['--case','counterbored-plate','--n','1000']),('cavity',['--case','cavity','--n','1000']),('graded-armadillo',['--case','graded-surface','--n','4096','--input',str(w/'scanned-armadillo-input/armadillo.off'),'--subdivisions','1','--crease-subdivisions','1','--feature-angle','120'])]:
 for phase in ['before','after']:
  folder=out/(name+'-'+phase);script=['-m','examples.vtm_gallery'] if phase=='before' else [str(w/'run-bounded-hull-box-gallery-v2.py')]
  cmd=['.venv/bin/python',*script,*args,'--initial','256','--quality-rounds','4','--repeats','3','--output',str(folder)]
  log=out/(name+'-'+phase+'.log')
  with log.open('w') as f:proc=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
  row=dict(name=name,phase=phase,command=cmd,source=source,exit_code=proc.returncode,log=str(log))
  p=folder/'result.json'
  if p.exists():
   d=json.loads(p.read_text());row.update(result=str(p),result_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),valid=d['valid'],error=d['error'],counts=d['counts'],median_ms=d['median_additional_replay_ms'],quota_clean=all(v['throttled_periods']==0 for k in ['first_replay_cpu_quota_delta','additional_replays_cpu_quota_delta'] for v in d[k].values()))
  rows.append(row);(out/'commands.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
  for filename in ['mesh.vtu','boundary.off']:
   raw=folder/filename
   if raw.exists():
    b=raw.read_bytes();gz=raw.with_suffix(raw.suffix+'.gz');assert not gz.exists();gz.write_bytes(gzip.compress(b,mtime=0));assert gzip.decompress(gz.read_bytes())==b;raw.unlink()
  if name=='graded-armadillo' and (folder/'input.off').exists():
   original=folder/'input.off';canonical=w/'scanned-armadillo-input/armadillo.off';assert original.read_bytes()==canonical.read_bytes();original.unlink();os.link(canonical,original)
  assert proc.returncode==0,row
