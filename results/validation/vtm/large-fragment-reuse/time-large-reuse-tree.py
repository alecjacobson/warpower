import json,os,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm'
cmd=[str(root/'.venv/bin/python'),str(w/'run-large-reuse-tree.py'),'--n','4096','--repeats','3','--save-mesh',str(w/'large-reuse-tree4096.npz'),'--output',str(w/'large-reuse-tree4096.json')]
start=time.time();timed_out=False
with (w/'large-reuse-tree4096.log').open('w') as log:
 try:r=subprocess.run(cmd,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.'),stdout=log,stderr=subprocess.STDOUT,timeout=180);code=r.returncode
 except subprocess.TimeoutExpired:code=124;timed_out=True
row=dict(command=cmd,exit_code=code,timed_out=timed_out,timeout_seconds=180,wall_seconds=time.time()-start,source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
(w/'large-reuse-tree4096-command.json').write_text(json.dumps(row,indent=2)+'\n')
if code==0:
 d=json.loads((w/'large-reuse-tree4096.json').read_text()); print('valid',d['valid'],'median ms',d['median_gpu_milliseconds'],'quota',d['cpu_quota_delta'],flush=True)
raise SystemExit(code)
