import json,os,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';rows=[]
for name in ['control','reuse']:
 stem=w/f'large-reuse-native-{name}'
 cmd=['nsys','profile','--trace=cuda','--cuda-graph-trace=node','--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=true','-o',str(stem),str(root/'.venv/bin/python'),str(w/'profile-large-reuse-native.py'),'--output',str(stem.with_suffix('.json'))]
 if name=='reuse':cmd.append('--reuse')
 start=time.time()
 with stem.with_suffix('.log').open('w') as log:
  try:code=subprocess.run(cmd,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.'),stdout=log,stderr=subprocess.STDOUT,timeout=180).returncode
  except subprocess.TimeoutExpired:code=124
 rows.append(dict(name=name,command=cmd,exit_code=code,wall_seconds=time.time()-start));(w/'large-reuse-native-commands.json').write_text(json.dumps(rows,indent=2)+'\n')
 print(name,code,flush=True)
 assert code==0
