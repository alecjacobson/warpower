"""Observe identical material evaluations without skipping any production work."""
import gzip,hashlib,json,os,signal,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=w/'current-selection-observers-v3';assert not out.exists();out.mkdir()
old=json.loads((w/'shared-wall-final/commands.json').read_text());rows=[]
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.:.work/vtm',PYTHONDONTWRITEBYTECODE='1',WARP_CACHE_PATH='/opt/warpower-generated-cache')
source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
for name in ['torus1000','graded-torus','thin-hand','cavity','disconnected']:
 ref=next(r for r in old if r['name']==name and r['phase']=='after');cmd=ref['command'][:];stem=out/name
 cmd[cmd.index('--repeats')+1]='1';gallery=not name.startswith('torus')
 cmd[1:3]=[str(w/('run-current-selection-gallery-v3.py' if gallery else 'current_selection_observer_v3.py'))]
 cmd[cmd.index('--output')+1]=str(stem if gallery else stem.with_suffix('.json'))
 if '--save-mesh' in cmd:cmd[cmd.index('--save-mesh')+1]=str(stem.with_suffix('.npz'))
 observer=out/(name+'-observer.json');runenv=dict(env,WARPOWER_SELECTION_OBSERVER=str(observer));start=time.time()
 with stem.with_suffix('.log').open('w') as log:
  p=subprocess.Popen(cmd,env=runenv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=p.wait(timeout=300)
  except subprocess.TimeoutExpired:
   os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=10);code=124
 row=dict(name=name,command=cmd,source=source,exit_code=code,wall_seconds=time.time()-start,scope='Instrumentation only; no speed claim.')
 if code==0:
  d=json.loads((stem/'result.json' if gallery else stem.with_suffix('.json')).read_text());assert d['valid']==1 and d['error']==0
  report=json.loads(observer.read_text());row['instances']=[{k:v for k,v in r.items() if k!='records'} for r in report['instances']]
 rows.append(row);(out/'commands.json').write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True);assert code==0
 if gallery:
  for path in [stem/'mesh.vtu',stem/'boundary.off']:
   data=path.read_bytes();packed=path.with_suffix(path.suffix+'.gz');packed.write_bytes(gzip.compress(data,mtime=0));assert gzip.decompress(packed.read_bytes())==data
   path.unlink()
