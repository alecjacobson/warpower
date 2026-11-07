"""Fresh process pairs for current complete torus solves; private kernels."""
import json
import os
import signal
import subprocess
import time
from pathlib import Path

root=Path.cwd();w=root/'.work/vtm';out=w/'cooperative-retained-pilot-v1'
assert not out.exists();out.mkdir()
assert '27 passed' in (w/'cooperative-retained-tests-v2.log').read_text()
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()=='305758fac0575cfb4c7ded604df02a52560905ce'
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.:.work/vtm',PYTHONDONTWRITEBYTECODE='1',WARP_CACHE_PATH='/opt/warpower-generated-cache')
rows=[]
for count in [1000,4096]:
    for phase in ['control','candidate']:
        stem=out/f'torus{count}-{phase}'
        base=[str(root/'.venv/bin/python'),'-m','benchmarks.mesher']
        if phase=='candidate':base=[base[0],str(w/'run_cooperative_retained_v2.py'),'mesher']
        cmd=base+['--n',str(count),'--repeats','5','--save-mesh',str(stem.with_suffix('.npz')),'--output',str(stem.with_suffix('.json'))]
        start=time.time()
        with stem.with_suffix('.log').open('w') as log:
            proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=proc.wait(timeout=300)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=10);code=124
        row=dict(case=f'torus{count}',phase=phase,command=cmd,source='305758fac0575cfb4c7ded604df02a52560905ce',exit_code=code,wall_seconds=time.time()-start,
                 scope='Complete resident replay. Candidate preloaded before setup, so no setup comparison claim.')
        if code==0:
            data=json.loads(stem.with_suffix('.json').read_text())
            row.update(median_ms=data['median_gpu_milliseconds'],valid=data['valid'],error=data['error'],
                       quota_clean=all(v['throttled_periods']==0 for key in ['cpu_quota_delta','initial_replays_cpu_quota_delta'] for v in data[key].values()))
        rows.append(row);(out/'commands.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(row,flush=True)
        assert code==0 and row['valid']==1 and row['error']==0 and row['quota_clean'],row
