"""Fresh paired complete solves; one process at a time, no concurrent oracles."""
import json,os,signal,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=w/'analytic-face-broad-recheck'
assert not out.exists();out.mkdir()
checks=json.loads((w/'analytic-face-pilot-independent.json').read_text())
assert len(checks)==2 and all(r['after_ms']<r['before_ms'] and r['comparison']['points_bitwise_equal'] for r in checks.values())
source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
assert source.startswith('fb795ac') and not subprocess.check_output(['git','status','--short'],text=True)
old=json.loads((w/'boundary-timing-commands.json').read_text());rows=[]
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.:.work/vtm')
for name in ['armadillo-graded']:
    ref=next(r for r in old if r['phase']=='after' and r['name']==name)
    for variant in ['control','analytic']:
        stem=out/(name+'-'+variant);cmd=ref['command'][:]
        cmd[cmd.index('--output')+1]=str(stem)
        if name=='graded-torus':cmd[cmd.index('--n')+1]='4096'
        if variant=='analytic':
            assert cmd[1:3]==['-m','examples.vtm_gallery']
            cmd[1:3]=[str(w/'run-analytic-gallery.py')]
        start=time.time()
        with stem.with_suffix('.log').open('w') as log:
            p=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=p.wait(timeout=240)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                code=124
        row=dict(name=name,variant=variant,command=cmd,source=source,exit_code=code,wall_seconds=time.time()-start,timeout_seconds=240)
        if (stem/'result.json').exists():
            d=json.loads((stem/'result.json').read_text())
            row.update(valid=d['valid'],error=d['error'],median_ms=d['median_additional_replay_ms'],quota_clean=all(v['throttled_periods']==0 for k in ['cpu_quota_delta','initial_replays_cpu_quota_delta','additional_replays_cpu_quota_delta','first_replay_cpu_quota_delta'] if k in d for v in d[k].values()))
        rows.append(row);(out/'commands.json').write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True)
        assert code==0 and row.get('valid')==1 and row.get('error')==0 and row.get('quota_clean'),row
