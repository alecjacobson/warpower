"""Padded and pool-disabled memory checks of geometric and captured transactions."""
import json,os,signal,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';python=root/'.venv/bin/python';rows=[]
assert '1084 passed' in (w/'proven-exterior-full-suite-v2.log').read_text()
assert not (w/'proven-exterior-memory-v1-commands.json').exists()
groups=[('primitives',['tests/test_exterior_proof.py','tests/test_centroid_ray_filter.py','tests/test_boundary_rounding.py'],33),
        ('transactions',['tests/test_proven_exterior.py'],8),
        ('full-local4',['tests/test_proven_mesher.py::test_complete_hand_recovery_preserves_corners_and_runs_quality[7-4]'],1),
        ('loop',['tests/test_proven_mesher.py::test_one_captured_loop_failure_healthy_seed_and_recovery'],1)]
for name,tests,count in groups:
 for mode in ['padded','lifetime']:
  env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.')
  options=[]
  if mode=='lifetime':env['WARPOWER_AUDIT_DISABLE_POOL']='1';options=['--check-api-memory-access','yes','--track-stream-ordered-races','use-after-free']
  cmd=['/usr/local/cuda-12.9/bin/compute-sanitizer','--tool','memcheck','--padding','32','--error-exitcode','99',*options,str(python),str(w/'large-reuse-isolated-test.py'),*tests]
  log=w/f'proven-exterior-memory-v1-{name}-{mode}.log';assert not log.exists();start=time.time()
  with log.open('w') as stream:
   p=subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
   try:code=p.wait(timeout=1200)
   except subprocess.TimeoutExpired:
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    code=124
  text=log.read_text();row=dict(name=name,mode=mode,command=cmd,exit_code=code,wall_seconds=time.time()-start,timeout_seconds=1200,disable_pool=mode=='lifetime',zero_memcheck_errors='ERROR SUMMARY: 0 errors' in text,expected_tests=count)
  rows.append(row);(w/'proven-exterior-memory-v1-commands.json').write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True)
  assert code==0 and row['zero_memcheck_errors'] and f'{count} passed' in text,row
