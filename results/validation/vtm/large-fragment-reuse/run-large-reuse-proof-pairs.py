"""Broader #161 control/prototype pairs before any production default change."""
import json,os,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=w/'large-reuse-proof-pairs';assert not out.exists();out.mkdir()
old=json.loads((w/'boundary-timing-commands.json').read_text());source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();rows=[]
for name,old_name in [('graded-torus4096','graded-torus'),('armadillo-uniform','armadillo-uniform'),('armadillo-graded','armadillo-graded')]:
 ref=next(r for r in old if r['phase']=='after' and r['name']==old_name)
 for phase in ['control','proofs']:
  stem=out/(name+'-'+phase)
  cmd=ref['command'][:];cmd[cmd.index('--output')+1]=str(stem);cmd[cmd.index('--n')+1]='4096'
  env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.')
  if phase=='proofs':
   assert cmd[1:3]==['-m','examples.vtm_gallery'];cmd[1:3]=[str(w/'run-large-reuse-proofs.py')];env['WARPOWER_LARGE_REUSE_GALLERY']='1'
  start=time.time();timed_out=False
  with stem.with_suffix('.log').open('w') as log:
   try:code=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=240).returncode
   except subprocess.TimeoutExpired:code=124;timed_out=True
  row=dict(name=name,phase=phase,command=cmd,source=source,exit_code=code,timed_out=timed_out,timeout_seconds=240,wall_seconds=time.time()-start)
  result=stem/'result.json'
  if result.exists():
   d=json.loads(result.read_text());row.update(valid=d['valid'],error=d['error'],median_ms=d.get('median_additional_replay_ms'),quota_clean=all(v['throttled_periods']==0 for k in ['cpu_quota_delta','initial_replays_cpu_quota_delta','additional_replays_cpu_quota_delta','first_replay_cpu_quota_delta'] if k in d for v in d[k].values()))
  rows.append(row);(out/'commands.json').write_text(json.dumps(rows,indent=2)+'\n');print(name,phase,row,flush=True)
  assert code==0 and row.get('valid')==1 and row.get('quota_clean'),row
