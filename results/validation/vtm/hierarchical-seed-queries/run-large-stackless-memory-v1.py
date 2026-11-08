"""Padded and pool-disabled memory checks of geometric and captured transactions."""
import json,os,signal,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';python=root/'.venv/bin/python';rows=[]
assert '1227 passed' in (w/'large-stackless-full-suite-v1.log').read_text()
assert not (w/'large-stackless-memory-v1-commands.json').exists()
tests = [
 'tests/test_stackless_seeds.py::test_captured_producers_preserve_exhaustive_and_rational_masks[32-0-random4097]',
 'tests/test_stackless_seeds.py::test_captured_producers_preserve_exhaustive_and_rational_masks[256-60-torus16384]',
 'tests/test_stackless_traversal.py::test_stackless_coverage_matches_recursive_pruning[32-16385]',
 'tests/test_stackless_traversal.py::test_segment_ancestors_enclose_current_groups_and_extreme_points[256-16385]',
]
collected=(w/'extended-reuse-collection-v1.log').read_text().splitlines()
for line in collected:
 if line.startswith('tests/test_large_reuse.py::') and any(x in line for x in ['[4097]','[8192]','[16384]','-16384]']):
  tests.append(line.strip())
assert len(tests)==12, tests

groups=[(f'case{i:02d}',[test],1) for i,test in enumerate(tests)]
for name,tests,count in groups:
 for mode in ['padded','lifetime']:
  env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.')
  options=[]
  if mode=='lifetime':env['WARPOWER_AUDIT_DISABLE_POOL']='1';options=['--check-api-memory-access','yes','--track-stream-ordered-races','use-after-free']
  cmd=['/usr/local/cuda-12.9/bin/compute-sanitizer','--tool','memcheck','--padding','32','--error-exitcode','99',*options,str(python),str(w/'large-reuse-isolated-test.py'),*tests]
  log=w/f'large-stackless-memory-v1-{name}-{mode}.log';assert not log.exists();start=time.time()
  with log.open('w') as stream:
   p=subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
   try:code=p.wait(timeout=600)
   except subprocess.TimeoutExpired:
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    code=124
  text=log.read_text();row=dict(name=name,mode=mode,command=cmd,exit_code=code,wall_seconds=time.time()-start,timeout_seconds=600,disable_pool=mode=='lifetime',zero_memcheck_errors='ERROR SUMMARY: 0 errors' in text,expected_tests=count)
  rows.append(row);(w/'large-stackless-memory-v1-commands.json').write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True)
  assert code==0 and row['zero_memcheck_errors'] and f'{count} passed' in text,row
