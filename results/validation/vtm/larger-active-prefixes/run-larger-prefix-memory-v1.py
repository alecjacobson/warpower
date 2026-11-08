"""Padded and pool-disabled memory checks of geometric and captured transactions."""
import json,os,signal,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';python=root/'.venv/bin/python';rows=[]
assert '1292 passed' in (w/'larger-prefix-full-suite-v1.log').read_text()
assert not (w/'larger-prefix-memory-v1-commands.json').exists()
collected=(w/'larger-prefix-collection-v1.log').read_text().splitlines()
tests=[line.strip() for line in collected if line.startswith('tests/test_larger_prefixes.py::') and 'oversized_explicit' not in line]
assert len(tests)==12,tests
tests += [
 'tests/test_active_delaunay.py::test_changing_prefixes_producers_consumers_and_failures_in_one_graph',
 'tests/test_active_delaunay.py::test_vtm_rollback_restores_active_mesh_across_routed_construction',
]
assert len(tests)==14

groups=[(f'case{i:02d}',[test],1) for i,test in enumerate(tests)]
for name,tests,count in groups:
 for mode in ['padded','lifetime']:
  env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.')
  options=[]
  if mode=='lifetime':env['WARPOWER_AUDIT_DISABLE_POOL']='1';options=['--check-api-memory-access','yes','--track-stream-ordered-races','use-after-free']
  cmd=['/usr/local/cuda-12.9/bin/compute-sanitizer','--tool','memcheck','--padding','32','--error-exitcode','99',*options,str(python),str(w/'large-reuse-isolated-test.py'),*tests]
  log=w/f'larger-prefix-memory-v1-{name}-{mode}.log';assert not log.exists();start=time.time()
  with log.open('w') as stream:
   p=subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
   try:code=p.wait(timeout=600)
   except subprocess.TimeoutExpired:
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    code=124
  text=log.read_text();row=dict(name=name,mode=mode,command=cmd,exit_code=code,wall_seconds=time.time()-start,timeout_seconds=600,disable_pool=mode=='lifetime',zero_memcheck_errors='ERROR SUMMARY: 0 errors' in text,expected_tests=count)
  rows.append(row);(w/'larger-prefix-memory-v1-commands.json').write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True)
  assert code==0 and row['zero_memcheck_errors'] and f'{count} passed' in text,row
