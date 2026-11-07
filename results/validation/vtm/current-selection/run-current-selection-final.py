"""Final integrated solves and unchanged standalone API pairs, after full suite."""
import gzip,hashlib,json,os,signal,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=w/'current-selection-final';baseline=w/'extraction-geometry';python=root/'.venv/bin/python'
assert not out.exists()
assert '851 passed' in (w/'current-selection-full-suite.log').read_text()
assert len(json.loads((w/'current-selection-broad-v1-independent.json').read_text()))==6
source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
base_source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=baseline,text=True).strip()
assert base_source=='80d6ded503ac799048c3a82d17003951aa0f8aa6'
assert not subprocess.check_output(['git','diff','--','warpower','benchmarks','tests'],text=True)
assert not subprocess.check_output(['git','status','--short'],cwd=baseline,text=True)
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.')
assert env.get('WARP_CACHE_PATH')=='/opt/warpower-generated-cache'
out.mkdir()
rows=[]
archives=[]
def archive(path):
    data=path.read_bytes();dest=path.with_suffix(path.suffix+'.gz');assert not dest.exists()
    dest.write_bytes(gzip.compress(data,mtime=0));assert gzip.decompress(dest.read_bytes())==data
    archives.append(dict(original=str(path.relative_to(w)),path=str(dest.relative_to(w)),raw_bytes=len(data),bytes=dest.stat().st_size,raw_sha256=hashlib.sha256(data).hexdigest(),sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
    (w/'current-selection-final-lossless-archives.json').write_text(json.dumps(archives,indent=2)+'\n')
    path.unlink()

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
    if stem.is_dir():
        for item in [stem/'mesh.vtu',stem/'boundary.off']:
            if item.exists():archive(item)


old=json.loads((w/'boundary-timing-commands.json').read_text())
for name in ['torus1000','torus4096','graded-torus','thin-hand','cavity','disconnected','armadillo-uniform','armadillo-graded']:
    ref=next(r for r in old if r['phase']=='after' and r['name']==name)
    for phase,cwd in [('before',baseline),('after',root)]:
        stem=out/(name+'-'+phase);cmd=ref['command'][:]
        if name=='graded-torus':cmd[cmd.index('--n')+1]='4096'
        cmd[cmd.index('--output')+1]=str(stem.with_suffix('.json')) if name.startswith('torus') else str(stem)
        if '--save-mesh' in cmd:cmd[cmd.index('--save-mesh')+1]=str(stem.with_suffix('.npz'))
        run(name,phase,cmd,cwd)
for ref in old:
    name=ref['name']
    if ref['phase']!='after' or not name.startswith(('delaunay','diagram')):continue
    for phase,cwd in [('before',baseline),('after',root)]:
        stem=out/(name+'-'+phase);cmd=ref['command'][:];cmd[cmd.index('--output')+1]=str(stem.with_suffix('.json'))
        run(name,phase,cmd,cwd)
assert len(rows)==26
