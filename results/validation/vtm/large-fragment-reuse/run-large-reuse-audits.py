"""Adoption gates after all final clean timing jobs. Serial GPU/CPU execution."""
import json,os,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';python=root/'.venv/bin/python'
assert '650 passed' in (w/'large-reuse-full-suite.log').read_text()
timings=json.loads((w/'large-reuse-final/commands.json').read_text());assert len(timings)==26 and all(r['exit_code']==0 and r['quota_clean'] for r in timings)
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.')
rows=[]
def run(name,cmd,timeout=1800):
    cmd=list(map(str,cmd));logpath=w/f'large-reuse-{name}.log';assert not logpath.exists(),logpath
    start=time.time()
    with logpath.open('w') as log:
        try:code=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout).returncode
        except subprocess.TimeoutExpired:code=124
    rows.append(dict(name=name,command=cmd,exit_code=code,timeout_seconds=timeout,wall_seconds=time.time()-start));(w/'large-reuse-audit-commands.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(name,code,rows[-1]['wall_seconds'],flush=True);assert code==0,rows[-1]
mem=['/usr/local/cuda-12.9/bin/compute-sanitizer','--tool','memcheck','--padding','32','--error-exitcode','99']
tests=['-q','tests/test_large_reuse.py']
run('memcheck',mem+[python,'-m','benchmarks.memcheck','--stack-bytes','131072','--']+tests)
run('lifetime',mem+['--check-api-memory-access','yes','--track-stream-ordered-races','all',python,'-m','benchmarks.memcheck','--stack-bytes','131072','--disable-mempool','--']+tests)
for name,script in [('full','large-reuse-full-native.py'),('loop','large-reuse-loop.py')]:
    stem=w/f'large-reuse-{name}-native'
    cmd=['nsys','profile','--trace=cuda','--cuda-graph-trace=node','--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=true','-o',stem,python,w/script,'--dot',stem.with_suffix('.dot'),'--output',stem.with_suffix('.json')]
    if name=='loop':cmd.append('--profile')
    run(name+'-native',cmd,timeout=300)
run('native-adoption-analysis',[python,w/'analyze-large-reuse-audit.py'],timeout=180)
