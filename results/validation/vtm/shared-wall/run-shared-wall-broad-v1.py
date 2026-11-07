"""Serial full-solve private wall probes on existing paper-inspired examples."""
import hashlib,json,os,signal,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=w/'shared-wall-broad-v1';python=root/'.venv/bin/python'
assert not out.exists();out.mkdir();rows=[]
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.:.work/vtm',PYTHONDONTWRITEBYTECODE='1',WARP_CACHE_PATH='/opt/warpower-generated-cache')
source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
old=json.loads((w/'boundary-timing-commands.json').read_text())
for name in ['graded-torus','thin-hand','cavity','disconnected','armadillo-uniform','armadillo-graded']:
    ref=next(r for r in old if r['phase']=='after' and r['name']==name)
    for phase in ['control','candidate']:
        stem=out/(name+'-'+phase);cmd=ref['command'][:]
        if name=='graded-torus':cmd[cmd.index('--n')+1]='4096'
        cmd[cmd.index('--output')+1]=str(stem)
        if phase=='candidate':cmd[1:3]=[str(w/'run-shared-wall-gallery-v1.py')]
        start=time.time()
        with stem.with_suffix('.log').open('w') as log:
            p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=p.wait(timeout=300)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=10);code=124
        row=dict(name=name,phase=phase,command=cmd,source=source,exit_code=code,wall_seconds=time.time()-start)
        result=stem/'result.json'
        if result.exists():
            d=json.loads(result.read_text());row.update(median_ms=d['median_additional_replay_ms'],valid=d['valid'],error=d['error'],quota_clean=all(v['throttled_periods']==0 for key in ['additional_replays_cpu_quota_delta','first_replay_cpu_quota_delta'] if key in d for v in d[key].values()))
        rows.append(row);(out/'commands.json').write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True)
        assert code==0 and row.get('valid')==1 and row.get('error')==0 and row.get('quota_clean'),row
        if name.startswith('armadillo'):
            duplicate=stem/'input.off';original=w/'scanned-armadillo-input/armadillo.off'
            assert hashlib.sha256(duplicate.read_bytes()).digest()==hashlib.sha256(original.read_bytes()).digest()
            duplicate.unlink();os.link(original,duplicate)
