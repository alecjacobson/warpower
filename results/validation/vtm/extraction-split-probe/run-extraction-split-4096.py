import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
root=Path.cwd();w=root/'.work/vtm';out=w/'extraction-split-4096';out.mkdir(exist_ok=False)
revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.')
rows=[]
for name,runner in [('control',['-m','benchmarks.mesher']),('split',[str(w/'run-extraction-split.py')])]:
 stem=out/name;cmd=[str(root/'.venv/bin/python'),*runner,'--n','4096','--repeats','3','--save-mesh',str(stem.with_suffix('.npz')),'--output',str(stem.with_suffix('.json'))]
 start=time.time()
 with stem.with_suffix('.log').open('w') as log:
  code=subprocess.run(cmd,env=env,cwd=root,stdout=log,stderr=subprocess.STDOUT,timeout=180).returncode
 row=dict(name=name,command=cmd,source=revision,exit_code=code,wall_seconds=time.time()-start)
 rows.append(row);(out/'commands.json').write_text(json.dumps(rows,indent=2)+'\n')
 assert code==0
 d=json.loads(stem.with_suffix('.json').read_text())
 assert d['valid']==1 and d['error']==0
 assert all(v['throttled_periods']==0 for k in ['cpu_quota_delta','initial_replays_cpu_quota_delta'] for v in d[k].values())
 print(name,d['median_gpu_milliseconds'],flush=True)
import numpy as np
files=[np.load(out/(n+'.npz')) for n in ['control','split']]
checks={}
for key in files[0].files:
 a,b=[f[key] for f in files]
 if key in ['tetrahedra','full_tetrahedra']:
  a,b=[np.sort(v,axis=1) for v in [a,b]]
  a,b=[v[np.lexsort(v.T[::-1])] for v in [a,b]]
 elif key=='boundary':
  a,b=[np.asarray([np.roll(r,-int(np.argmin(r))) for r in v]) for v in [a,b]]
  a,b=[v[np.lexsort(v.T[::-1])] for v in [a,b]]
 checks[key]=bool(np.array_equal(a,b))
assert all(checks.values()),checks
before,after=[json.loads((out/(n+'.json')).read_text()) for n in ['control','split']]
for k in ['rounds','optimizer_steps','quality_moves','refinement_rejections','quality_rejections','counts']:assert before[k]==after[k],k
(out/'comparison.json').write_text(json.dumps(dict(arrays_equal=checks,work_equal=True,all_initial_and_timed_quotas_clean=True,before_ms=before['median_gpu_milliseconds'],after_ms=after['median_gpu_milliseconds'],source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [w/'extraction_split.py',w/'run-extraction-split.py']}),indent=2)+'\n')
print('same complete mesh and work',flush=True)
