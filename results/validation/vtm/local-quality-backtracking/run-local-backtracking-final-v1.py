"""Final integrated solves and unchanged standalone API pairs, after full suite."""
import gzip,hashlib,json,os,signal,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=w/'local-backtracking-final';baseline=w/'local-backtracking-baseline';python=root/'.venv/bin/python'
assert not out.exists()
assert '1034 passed' in (w/'local-backtracking-production-full-suite-v1.log').read_text()
assert '46 passed' in (w/'local-backtracking-production-focused-v2.log').read_text()
source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
base_source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=baseline,text=True).strip()
assert base_source=='39f8f54fa5cc410b46cd9b057e617fc92f7e4c28'
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
    (w/'local-backtracking-final-lossless-archives.json').write_text(json.dumps(archives,indent=2)+'\n')
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
    expected_failure=name=='coarse-hand'
    assert code==(1 if expected_failure else 0) and row.get('quota_clean') and row.get('valid') in ([0] if expected_failure else [None,1]) and row.get('error') in ([64] if expected_failure else [None,0]),row
    if name.startswith('armadillo'):
        duplicate=stem/'input.off';canonical=w/'scanned-armadillo-input/armadillo.off'
        digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        assert digest(duplicate)==digest(canonical)
        duplicate.unlink();os.link(canonical,duplicate)
    if stem.is_dir():
        for item in [stem/'mesh.vtu',stem/'boundary.off']:
            if item.exists():archive(item)


old=json.loads((w/'boundary-timing-commands.json').read_text())
cases=[('torus1000','torus',1000,[]),('torus4096','torus',4096,[]),('graded-torus','graded-torus',4096,[]),
       ('thin-hand','thin-hand',1000,[]),('plate','counterbored-plate',1000,[]),('cavity','cavity',1000,[]),('disconnected','disconnected',1000,[]),
       ('armadillo-uniform','surface',4096,['--input',str(w/'scanned-armadillo-input/armadillo.off'),'--subdivisions','1','--crease-subdivisions','1','--feature-angle','120']),
       ('armadillo-graded','graded-surface',4096,['--input',str(w/'scanned-armadillo-input/armadillo.off'),'--subdivisions','1','--crease-subdivisions','1','--feature-angle','120']),
       ('hand-seed7','thin-hand',1000,['--seed','7']),('plate-seed7','counterbored-plate',1000,['--seed','7']),
       ('coarse-hand','thin-hand',1000,['--subdivisions','4','--crease-subdivisions','4']),
       ('torus-round1','torus',1000,['--quality-rounds','1']),('torus-round2','torus',1000,['--quality-rounds','2'])]
for name,case,n,extra in cases:
    phases=[('before',baseline),('enabled',root)]
    if name in ['torus1000','thin-hand','plate']:phases.insert(1,('disabled',root))
    for phase,cwd in phases:
        stem=out/(name+'-'+phase)
        cmd=[str(python),'-m','examples.vtm_gallery','--case',case,'--n',str(n),'--repeats','3',*extra,'--output',str(stem)]
        if phase!='before':cmd+=['--local-quality-retries','12' if phase=='enabled' else '0']
        run(name,phase,cmd,cwd)
for ref in old:
    name=ref['name']
    if ref['phase']!='after' or not name.startswith(('delaunay','diagram')):continue
    for phase,cwd in [('before',baseline),('disabled',root)]:
        stem=out/(name+'-'+phase);cmd=ref['command'][:];cmd[cmd.index('--output')+1]=str(stem.with_suffix('.json'))
        run(name,phase,cmd,cwd)
assert len(rows)==41
