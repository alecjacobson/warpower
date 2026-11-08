"""Qualify all original37 contracts in isolated groups after the mixed batch deadline."""
import hashlib,json,os,signal,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=w/'selective-step-history-memory-v2.json';assert not out.exists()
prior=w/'selective-step-history-memory-v1-commands.json';p=json.loads(prior.read_text());assert len(p)==1 and p[0]['exit_code']==124
start=time.monotonic()
while not (w/'cpu-quality-optimizers-pilot-v1/complete.json').exists():
 assert time.monotonic()-start<4200
 time.sleep(10)
assert json.loads((w/'cpu-quality-optimizers-pilot-v1/complete.json').read_text())['complete']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
source=json.loads((w/'selective-step-history-production-source-v1.json').read_text())
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==source['runtime']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for p,h in source['source_sha256'].items():assert sha(root/p)==h
prefix='tests/test_selective_step_history.py::test_complete_mesher_eligibility_and_fresh_replay_resets'
groups=[('state',['tests/test_selective_step_history.py','-k','not complete_mesher'],17,300),('fresh-global',[prefix+'[0-0]'],1,1200),('fresh-history',[prefix+'[1-0]'],1,1200),('fresh-local',[prefix+'[1-2]'],1,1200),('local',['tests/test_local_backtracking.py'],7,1200),('controls',['tests/test_vtm_controls.py'],10,1200),('loop',[str(w/'test_tangential_production_loop_v2.py')],1,1200)]
report=dict(scope=__doc__,runtime=source['runtime'],prior_failure_sha256=sha(prior),original_contracts_preserved=37,complete=False,rows=[])
for name,tests,count,timeout in groups:
 for mode in ['padded','lifetime']:
  label='selective-step-history-memory-v2-'+name+'-'+mode;log=w/(label+'.log');assert not log.exists()
  env=dict(os.environ,PYTHONPATH='.',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',WARPOWER_TOPOLOGY_LOOP_OUTPUT=str(w/(label+'-frames')))
  options=[]
  if mode=='lifetime':env['WARPOWER_AUDIT_DISABLE_POOL']='1';options=['--check-api-memory-access','yes','--track-stream-ordered-races','use-after-free']
  cmd=['/usr/local/cuda-12.9/bin/compute-sanitizer','--tool','memcheck','--padding','32','--error-exitcode','99',*options,str(root/'.venv/bin/python'),str(w/'large-reuse-isolated-test.py'),*tests]
  start=time.time()
  with log.open('w') as stream:
   p=subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
   try:code=p.wait(timeout=timeout)
   except subprocess.TimeoutExpired:
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    code=124
  text=log.read_text();row=dict(name=name,mode=mode,command=cmd,exit_code=code,wall_seconds=time.time()-start,timeout_seconds=timeout,disable_pool=mode=='lifetime',zero_memcheck_errors='ERROR SUMMARY: 0 errors' in text,expected_tests=count,passed=f'{count} passed' in text,log_sha256=sha(log))
  report['rows'].append(row);out.write_text(json.dumps(report,indent=2)+'\n');print(row,flush=True)
  assert code==0 and row['zero_memcheck_errors'] and row['passed'],row
assert len(report['rows'])==14
for mode in ['padded','lifetime']:assert sum(r['expected_tests'] for r in report['rows'] if r['mode']==mode and r['name']!='loop')==37
for p,h in source['source_sha256'].items():assert sha(root/p)==h
report['complete']=True;out.write_text(json.dumps(report,indent=2)+'\n')
