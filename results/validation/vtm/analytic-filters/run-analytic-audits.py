"""Focused unfiltered instrumentation; preserve known full-order failures."""
import json,os,signal,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';python=root/'.venv/bin/python';rows=[]
assert '694 passed' in (w/'analytic-full-suite.log').read_text()
assert len(json.loads((w/'analytic-final/commands.json').read_text()))==18
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.')
assert env.get('WARP_CACHE_PATH')=='/opt/warpower-generated-cache'
nodes=[s for s in (w/'large-reuse-collected-tests.log').read_text().splitlines() if s.startswith('tests/test_large_reuse.py::')]
assert len(nodes)==7
def run(name,cmd,environment=env,timeout=600,expected_tests=None,allow_failure=False):
    cmd=list(map(str,cmd));path=w/('analytic-'+name+'.log');assert not path.exists()
    start=time.time()
    with path.open('w') as log:
        p=subprocess.Popen(cmd,cwd=root,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            code=124
    row=dict(name=name,command=cmd,exit_code=code,timeout_seconds=timeout,wall_seconds=time.time()-start,disable_pool=environment.get('WARPOWER_AUDIT_DISABLE_POOL')=='1')
    content=path.read_text();row['zero_memcheck_errors']='ERROR SUMMARY: 0 errors' in content
    rows.append(row);(w/'analytic-audit-commands.json').write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True)
    if not allow_failure:
        assert code==0,row
        if expected_tests is not None:assert f'{expected_tests} passed' in content and row['zero_memcheck_errors'],row
    return row

mem=['/usr/local/cuda-12.9/bin/compute-sanitizer','--tool','memcheck','--padding','32','--error-exitcode','99']
for name,path,n in [('predicates','tests/test_analytic_predicates.py',8),('queries','tests/test_analytic_face_search.py',36)]:
    for mode in ['memcheck','lifetime-all']:
        command=mem[:];e=dict(env)
        if mode=='lifetime-all':
            command+=['--check-api-memory-access','yes','--track-stream-ordered-races','all'];e['WARPOWER_AUDIT_DISABLE_POOL']='1'
        run(name+'-'+mode,command+[python,w/'large-reuse-isolated-test.py',path],e,expected_tests=n)
for i,node in enumerate(nodes):
    for mode in ['memcheck','lifetime-after-free']:
        command=mem[:];e=dict(env)
        if mode=='lifetime-after-free':
            command+=['--check-api-memory-access','yes','--track-stream-ordered-races','use-after-free'];e['WARPOWER_AUDIT_DISABLE_POOL']='1'
        run(f'route-{i}-'+mode,command+[python,w/'large-reuse-isolated-test.py',node],e,expected_tests=1)
# Strict full allocation-order control is retained even if it reproduces #91.
# It is not relabeled a pass and does not replace the scopes above.
e=dict(env,WARPOWER_AUDIT_DISABLE_POOL='1')
run('route-5-lifetime-all',mem+['--check-api-memory-access','yes','--track-stream-ordered-races','all',python,w/'large-reuse-isolated-test.py',nodes[5]],e,allow_failure=True)
assert len(rows)==19
