"""Integrated #161 timing gates. Run only after the full-suite GPU job ends."""
import json,os,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=w/'large-reuse-final';baseline=w/'extraction-geometry';python=root/'.venv/bin/python'
assert not out.exists();out.mkdir()
source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();base_source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=baseline,text=True).strip()
assert base_source.startswith('cca7a1d')
assert not subprocess.check_output(['git','diff','--','warpower','benchmarks','tests'],text=True)
assert not subprocess.check_output(['git','status','--short'],cwd=baseline,text=True)
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.')
rows=[]

def run(name,phase,cmd,cwd=root):
 stem=out/(name+'-'+phase);start=time.time();timed_out=False
 with stem.with_suffix('.log').open('w') as log:
  try:code=subprocess.run(cmd,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300).returncode
  except subprocess.TimeoutExpired:code=124;timed_out=True
 row=dict(name=name,phase=phase,command=cmd,cwd=str(cwd),source=base_source if cwd==baseline else source,exit_code=code,timed_out=timed_out,timeout_seconds=300,wall_seconds=time.time()-start)
 p=Path(cmd[cmd.index('--output')+1]);p=p/'result.json' if p.is_dir() else p
 if p.exists():
  d=json.loads(p.read_text());row['result']=str(p)
  for key in ['valid','error']:row[key]=d.get(key)
  row['median_ms']=d.get('median_gpu_milliseconds',d.get('median_additional_replay_ms'))
  row['quota_clean']=all(v['throttled_periods']==0 for k in ['cpu_quota_delta','initial_replays_cpu_quota_delta','additional_replays_cpu_quota_delta','first_replay_cpu_quota_delta'] if k in d for v in d[k].values())
  if 'full' in d:
   row['median_ms']=d['full']['median_ms'];row['quota_clean']=all(v['throttled_periods']==0 for v in d['full']['cpu'].values())
  if 'median_seconds' in d:row['median_ms']=1000*d['median_seconds'];row['quota_clean']=not d['cpu_throttled']
 rows.append(row);(out/'commands.json').write_text(json.dumps(rows,indent=2)+'\n');print(name,phase,code,row.get('median_ms'),row.get('valid'),row.get('quota_clean'),flush=True)
 assert code==0 and row.get('quota_clean') and row.get('valid') in [None,1] and row.get('error') in [None,0],row

# Fresh integrated results against the already recorded current-runtime controls.
old=json.loads((w/'boundary-timing-commands.json').read_text())
for name,old_name in [('torus4096','torus4096'),('graded-torus4096','graded-torus'),('armadillo-uniform','armadillo-uniform'),('armadillo-graded','armadillo-graded')]:
 ref=next(r for r in old if r['phase']=='after' and r['name']==old_name)
 stem=out/(name+'-after');cmd=ref['command'][:];cmd[cmd.index('--n')+1]='4096'
 cmd[cmd.index('--output')+1]=str(stem.with_suffix('.json')) if name=='torus4096' else str(stem)
 if '--save-mesh' in cmd:cmd[cmd.index('--save-mesh')+1]=str(stem.with_suffix('.npz'))
 run(name,'after',cmd)

# Threshold, unused full route, and unchanged larger-reservation checks.
for name,n,capacity in [('torus1000',1000,1000),('torus2050',2050,2050),('torus3072',3072,3072),('reserved4096-target1000',1000,4096),('reserved4096-target2048',2048,4096),('reserved8192-target4096',4096,8192)]:
 for phase,cwd in [('before',baseline),('after',root)]:
  stem=out/(name+'-'+phase)
  cmd=[str(python),'-m','benchmarks.mesher','--n',str(n),'--capacity',str(capacity),'--repeats',str(5 if name=='torus1000' else 3),'--save-mesh',str(stem.with_suffix('.npz')),'--output',str(stem.with_suffix('.json'))]
  run(name,phase,cmd,cwd)

# Independent basic geometry APIs; ordinary defaults and total-operation timings.
for ref in old:
 name=ref['name']
 if ref['phase']!='after' or not name.startswith(('delaunay','diagram')):continue
 for phase,cwd in [('before',baseline),('after',root)]:
  stem=out/(name+'-'+phase);cmd=ref['command'][:];cmd[cmd.index('--output')+1]=str(stem.with_suffix('.json'))
  run(name,phase,cmd,cwd)
assert len(rows)==26,len(rows)
