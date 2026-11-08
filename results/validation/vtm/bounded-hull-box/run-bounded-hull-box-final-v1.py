"""Pinned production/baseline full solves and standalone nonregression pairs."""
import gzip, hashlib, json, os, signal, subprocess, time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=w/'bounded-hull-box-final';baseline=w/'bounded-hull-box-baseline';python=root/'.venv/bin/python'
assert not out.exists()
assert '1103 passed' in (w/'bounded-hull-box-full-suite-v1.log').read_text()
source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
base_source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=baseline,text=True).strip()
assert base_source=='c74e492888311599efacd2ce682f545604b4bc1a'
assert not subprocess.check_output(['git','status','--porcelain'],text=True)
assert not subprocess.check_output(['git','status','--porcelain'],cwd=baseline,text=True)
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.')
assert env.get('WARP_CACHE_PATH')=='/opt/warpower-generated-cache'
out.mkdir();rows=[];archives=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(name,phase,cmd,cwd):
    log=out/(name+'-'+phase+'.log');start=time.time()
    with log.open('w') as stream:
        p=subprocess.Popen(cmd,cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=p.wait(timeout=300)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM);p.wait();code=124
    row=dict(name=name,phase=phase,command=cmd,cwd=str(cwd),source=base_source if cwd==baseline else source,exit_code=code,timeout_seconds=300,wall_seconds=time.time()-start)
    path=Path(cmd[cmd.index('--output')+1]);path=path/'result.json' if path.is_dir() else path
    if path.exists():
        d=json.loads(path.read_text());row.update(result=str(path),result_sha256=sha(path),valid=d.get('valid'),error=d.get('error'))
        row['median_ms']=d.get('median_gpu_milliseconds',d.get('median_additional_replay_ms'))
        row['quota_clean']=all(v['throttled_periods']==0 for k in ['cpu_quota_delta','initial_replays_cpu_quota_delta','additional_replays_cpu_quota_delta','first_replay_cpu_quota_delta'] if k in d for v in d[k].values())
        if 'full' in d:row.update(median_ms=d['full']['median_ms'],quota_clean=all(v['throttled_periods']==0 for v in d['full']['cpu'].values()))
        if 'median_seconds' in d:row.update(median_ms=1000*d['median_seconds'],quota_clean=not d['cpu_throttled'])
    rows.append(row);(out/'commands.json').write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True)
    assert code in [0,1] and path.exists() and row['quota_clean'],row
    if row['valid']==0:assert code==1 and row['error']==64 and not any(d['counts']),row
    else:assert code==0 and row['error'] in [None,0],row
    if name=='coarse-hand':assert row['valid']==0,row
    else:assert row['valid'] in [None,1],row
    if name=='graded-armadillo':
        duplicate=path.parent/'input.off';canonical=w/'scanned-armadillo-input/armadillo.off';assert sha(duplicate)==sha(canonical)
        duplicate.unlink();os.link(canonical,duplicate)
    if path.name=='result.json':
        for export_name in ['mesh.vtu','boundary.off']:
            raw=path.parent/export_name
            if not raw.exists():continue
            data=raw.read_bytes();packed=raw.with_suffix(raw.suffix+'.gz');assert not packed.exists()
            packed.write_bytes(gzip.compress(data,mtime=0));assert gzip.decompress(packed.read_bytes())==data
            archives.append(dict(original=str(raw),path=str(packed),raw_bytes=len(data),bytes=packed.stat().st_size,raw_sha256=hashlib.sha256(data).hexdigest(),sha256=sha(packed)))
            (out/'lossless-archives.json').write_text(json.dumps(archives,indent=2)+'\n');raw.unlink()

cases=[('hand-seed'+str(seed),'thin-hand',1000,['--seed',str(seed)]) for seed in [0,1,2,7,42,2026]]
cases += [('coarse-hand','thin-hand',1000,['--subdivisions','4','--crease-subdivisions','4']),('plate','counterbored-plate',1000,[]),('coarse-cavity','cavity',1000,['--subdivisions','4','--crease-subdivisions','4']),('coarse-disconnected','disconnected',1000,['--subdivisions','4','--crease-subdivisions','4']),('torus4096','torus',4096,[]),('graded-armadillo','graded-surface',4096,['--input',str(w/'scanned-armadillo-input/armadillo.off'),'--subdivisions','1','--crease-subdivisions','1','--feature-angle','120']),('hand-seed7-local4','thin-hand',1000,['--seed','7','--local-quality-retries','4'])]
for name,case,n,extra in cases:
    phases=[('before',baseline),('after',root)]
    for phase,cwd in phases:
        cmd=[str(python),'-m','examples.vtm_gallery','--case',case,'--n',str(n),'--initial','256','--quality-rounds','4','--repeats','3',*extra,'--output',str(out/(name+'-'+phase))]
        run(name,phase,cmd,cwd)
old=json.loads((w/'boundary-timing-commands.json').read_text())
for ref in old:
    name=ref['name']
    if ref['phase']!='after' or not name.startswith(('delaunay','diagram')):continue
    for phase,cwd in [('before',baseline),('after',root)]:
        cmd=ref['command'][:];cmd[cmd.index('--output')+1]=str(out/(name+'-'+phase+'.json'));run(name,phase,cmd,cwd)
assert len(rows)==36
