"""Investigate allocation-order failure without filtering reported errors."""
import json,os,signal,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';python=root/'.venv/bin/python';base=w/'extraction-geometry';rows=[]
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.',WARPOWER_AUDIT_DISABLE_POOL='1')
nodes=[s for s in (w/'large-reuse-collected-tests.log').read_text().splitlines() if s.startswith('tests/')]
def run(name,cmd,cwd=root,allowed=(0,),timeout=600):
 cmd=list(map(str,cmd));path=w/f'large-reuse-{name}.log';assert not path.exists(),path
 start=time.time()
 with path.open('w') as log:
  process=subprocess.Popen(cmd,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=process.wait(timeout=timeout)
  except subprocess.TimeoutExpired:
   os.killpg(process.pid,signal.SIGTERM)
   try:process.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
   code=124
 row=dict(name=name,command=cmd,cwd=str(cwd),source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=cwd,text=True).strip(),exit_code=code,wall_seconds=time.time()-start,timeout_seconds=timeout,allowed_codes=list(allowed))
 rows.append(row);(w/'large-reuse-lifetime-controls.json').write_text(json.dumps(rows,indent=2)+'\n')
 print(name,code,row['wall_seconds'],flush=True);assert code in allowed,row
 return path
mem=['/usr/local/cuda-12.9/bin/compute-sanitizer','--tool','memcheck','--padding','32','--error-exitcode','99','--check-api-memory-access','yes']
for name,depth,track in [('radix4096-conditional-all',1,'all'),('radix4096-top-all',0,'all'),('radix4096-conditional-normal',1,None),('radix4096-conditional-afterfree',1,'use-after-free')]:
 cmd=mem+(['--track-stream-ordered-races',track] if track else [])
 run(name,cmd+[python,'-m','benchmarks.radix_capture','--disable-mempool','--stack-bytes','131072','--n','4096','--depth',str(depth),'--dot',w/f'large-reuse-{name}.dot'],allowed=(0,99) if name.endswith('conditional-all') else (0,))
for phase,cwd in [('before',base),('after',root)]:
 run(f'route-{phase}-all',mem+['--track-stream-ordered-races','all',python,w/'large-reuse-route-control.py','--output',w/f'large-reuse-route-{phase}-all.json'],cwd,allowed=(0,99))
# Complete every remaining strict audit; retain any additional failures.
for i in [2,4,5,6]:
 run(f'lifetime-isolated-{i}',mem+['--track-stream-ordered-races','all',python,w/'large-reuse-isolated-test.py',nodes[i]],allowed=(0,99))
strict=json.loads((w/'large-reuse-isolated-audit-commands.json').read_text())+rows
failed=[i for i in range(7) if next(r for r in strict if r['name']==f'lifetime-isolated-{i}')['exit_code']!=0]
for i in failed:
 path=run(f'afterfree-isolated-{i}',mem+['--track-stream-ordered-races','use-after-free',python,w/'large-reuse-isolated-test.py',nodes[i]])
 assert '1 passed' in path.read_text() and 'ERROR SUMMARY: 0 errors' in path.read_text()
(w/'large-reuse-lifetime-coverage.json').write_text(json.dumps(dict(full_tracking_failed_indices=failed,all_seven_have_zero_error_api_and_afterfree_coverage=True,full_tracking_passed_indices=[i for i in range(7) if i not in failed],scope='Strict full allocation-order failures remain failures; affected cases separately tested with use-after-free tracking. No production allocation changes or error filtering.'),indent=2)+'\n')
