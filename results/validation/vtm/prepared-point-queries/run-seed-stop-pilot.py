"""Serial control/private pairs after correctness probes, no production change."""
import hashlib,json,os,signal,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=w/'seed-stop-pilot';python=root/'.venv/bin/python'
assert '69 passed' in (w/'seed-stop-existing-tests.log').read_text()
assert not out.exists();out.mkdir()
assert not subprocess.check_output(['git','status','--short'],text=True)
source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();assert source.startswith('6469ab6')
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.:.work/vtm');rows=[]
for n in [1000,4096]:
 for variant in ['control','stop']:
  stem=out/f'{n}-{variant}'
  cmd=[str(python)]+(['-m','benchmarks.mesher'] if variant=='control' else [str(w/'run-seed-stop.py')])
  cmd+=['--n',str(n),'--repeats','5' if n==1000 else '3','--output',str(stem.with_suffix('.json')),'--save-mesh',str(stem.with_suffix('.npz'))]
  start=time.time()
  with stem.with_suffix('.log').open('w') as log:
   p=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   try:code=p.wait(timeout=180)
   except subprocess.TimeoutExpired:
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    code=124
  row=dict(n=n,variant=variant,command=cmd,source=source,exit_code=code,wall_seconds=time.time()-start,timeout_seconds=180)
  if stem.with_suffix('.json').exists():
   d=json.loads(stem.with_suffix('.json').read_text())
   row.update(median_ms=d['median_gpu_milliseconds'],valid=d['valid'],error=d['error'],quota_clean=all(v['throttled_periods']==0 for key in ['cpu_quota_delta','initial_replays_cpu_quota_delta'] for v in d[key].values()))
  rows.append(row);(out/'commands.json').write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True)
  assert code==0 and row.get('valid')==1 and row.get('error')==0 and row.get('quota_clean'),row
