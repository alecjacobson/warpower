"""Complete every new test in isolated groups after the combined job hit its wall limit."""
import json,os,signal,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';python=root/'.venv/bin/python';rows=[]
assert '1034 passed' in (w/'local-backtracking-production-full-suite-v1.log').read_text()
assert not (w/'local-backtracking-memory-v2-commands.json').exists()
groups=[('core',['tests/test_local_backtracking.py','-k','not optional_public_mesher']),
        ('mesher2026',['tests/test_local_backtracking.py::test_optional_public_mesher_preserves_topology_and_corners[2026]']),
        ('mesher7',['tests/test_local_backtracking.py::test_optional_public_mesher_preserves_topology_and_corners[7]'])]
for name,tests in groups:
 for mode in ['padded','lifetime']:
  env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.')
  options=[]
  if mode=='lifetime':
   env['WARPOWER_AUDIT_DISABLE_POOL']='1';options=['--check-api-memory-access','yes','--track-stream-ordered-races','use-after-free']
  cmd=['/usr/local/cuda-12.9/bin/compute-sanitizer','--tool','memcheck','--padding','32','--error-exitcode','99',*options,str(python),str(w/'large-reuse-isolated-test.py'),*tests]
  log=w/f'local-backtracking-v2-{name}-{mode}.log';assert not log.exists();start=time.time()
  with log.open('w') as stream:
   p=subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
   try:code=p.wait(timeout=900)
   except subprocess.TimeoutExpired:
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    code=124
  text=log.read_text();row=dict(name=name,mode=mode,command=cmd,exit_code=code,wall_seconds=time.time()-start,timeout_seconds=900,disable_pool=mode=='lifetime',zero_memcheck_errors='ERROR SUMMARY: 0 errors' in text,expected_tests=5 if name=='core' else 1)
  rows.append(row);(w/'local-backtracking-memory-v2-commands.json').write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True)
  assert code==0 and row['zero_memcheck_errors'] and f"{row['expected_tests']} passed" in text,row
