"""Final integrated solves and unchanged standalone API pairs, after full suite."""
import hashlib,json,os,signal,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=w/'prepared-group-final';baseline=w/'extraction-geometry';python=root/'.venv/bin/python'
assert not out.exists()
assert '786 passed' in (w/'prepared-group-full-suite.log').read_text()
assert len(json.loads((w/'prepared-group-broad-independent.json').read_text())['cases'])==6
source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
base_source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=baseline,text=True).strip()
assert base_source.startswith('ce51096')
assert not subprocess.check_output(['git','diff','--','warpower','benchmarks','tests'],text=True)
assert not subprocess.check_output(['git','status','--short'],cwd=baseline,text=True)
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.')
assert env.get('WARP_CACHE_PATH')=='/opt/warpower-generated-cache'
out.mkdir()
rows=[]
def run(name,phase,cmd,cwd=root):
    stem=out/(name+'-'+phase);start=time.time()
    with stem.with_suffix('.log').open('w') as log:
        p=subprocess.Popen(cmd,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=p.wait(timeout=300)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            code=124
    row=dict(name=name,phase=phase,command=cmd,cwd=str(cwd),source=base_source if cwd==baseline else source,exit_code=code,timeout_seconds=300,wall_seconds=time.time()-start)
    path=Path(cmd[cmd.index('--output')+1]);path=path/'result.json' if path.is_dir() else path
    if path.exists():
        d=json.loads(path.read_text());row['result']=str(path)
        for key in ['valid','error']:row[key]=d.get(key)
        row['median_ms']=d.get('median_gpu_milliseconds',d.get('median_additional_replay_ms'))
        row['quota_clean']=all(v['throttled_periods']==0 for k in ['cpu_quota_delta','initial_replays_cpu_quota_delta','additional_replays_cpu_quota_delta','first_replay_cpu_quota_delta'] if k in d for v in d[k].values())
        if 'full' in d:
            row['median_ms']=d['full']['median_ms'];row['quota_clean']=all(v['throttled_periods']==0 for v in d['full']['cpu'].values())
        if 'median_seconds' in d:row['median_ms']=1000*d['median_seconds'];row['quota_clean']=not d['cpu_throttled']
    rows.append(row);(out/'commands.json').write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True)
    assert code==0 and row.get('quota_clean') and row.get('valid') in [None,1] and row.get('error') in [None,0],row
    if name.startswith('armadillo'):
        duplicate=stem/'input.off';canonical=w/'scanned-armadillo-input/armadillo.off'
        digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        assert digest(duplicate)==digest(canonical)
        duplicate.unlink();os.link(canonical,duplicate)


old=json.loads((w/'boundary-timing-commands.json').read_text())
for name in ['torus1000','torus4096','graded-torus','thin-hand','cavity','disconnected','armadillo-uniform','armadillo-graded']:
    ref=next(r for r in old if r['phase']=='after' and r['name']==name)
    stem=out/(name+'-after');cmd=ref['command'][:]
    if name=='graded-torus':cmd[cmd.index('--n')+1]='4096'
    cmd[cmd.index('--output')+1]=str(stem.with_suffix('.json')) if name.startswith('torus') else str(stem)
    if '--save-mesh' in cmd:cmd[cmd.index('--save-mesh')+1]=str(stem.with_suffix('.npz'))
    run(name,'after',cmd)
for ref in old:
    name=ref['name']
    if ref['phase']!='after' or not name.startswith(('delaunay','diagram')):continue
    for phase,cwd in [('before',baseline),('after',root)]:
        stem=out/(name+'-'+phase);cmd=ref['command'][:];cmd[cmd.index('--output')+1]=str(stem.with_suffix('.json'))
        run(name,phase,cmd,cwd)
assert len(rows)==18
