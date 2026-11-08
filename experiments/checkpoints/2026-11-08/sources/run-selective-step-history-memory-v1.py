"""Audit selective step history, original controls and local rollback and the unchanged complete recovery loop in isolated processes."""
import json,os,signal,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';python=root/'.venv/bin/python';dest=w/'selective-step-history-memory-v1-commands.json';assert not dest.exists()
assert '1407 passed' in (w/'selective-step-history-production-full-suite-v1.log').read_text()
groups=[('kernels',['tests/test_selective_step_history.py','tests/test_local_backtracking.py','tests/test_vtm_controls.py'],37,600),('loop',[str(w/'test_tangential_production_loop_v2.py')],1,1200)]
rows=[]
for name,tests,count,timeout in groups:
 for mode in ['padded','lifetime']:
  label=f'selective-step-history-memory-v1-{name}-{mode}'
  env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.',WARPOWER_TOPOLOGY_LOOP_OUTPUT=str(w/(label+'-frames')))
  options=[]
  if mode=='lifetime':env['WARPOWER_AUDIT_DISABLE_POOL']='1';options=['--check-api-memory-access','yes','--track-stream-ordered-races','use-after-free']
  cmd=['/usr/local/cuda-12.9/bin/compute-sanitizer','--tool','memcheck','--padding','32','--error-exitcode','99',*options,str(python),str(w/'large-reuse-isolated-test.py'),*tests]
  log=w/(label+'.log');assert not log.exists();start=time.time()
  with log.open('w') as stream:
   p=subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
   try:code=p.wait(timeout=timeout)
   except subprocess.TimeoutExpired:
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    code=124
  text=log.read_text();row=dict(name=name,mode=mode,command=cmd,exit_code=code,wall_seconds=time.time()-start,timeout_seconds=timeout,disable_pool=mode=='lifetime',zero_memcheck_errors='ERROR SUMMARY: 0 errors' in text,expected_tests=count)
  rows.append(row);dest.write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True)
  assert code==0 and row['zero_memcheck_errors'] and f'{count} passed' in text,row
