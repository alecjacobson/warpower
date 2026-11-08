"""Pinned production/baseline full solves and standalone nonregression pairs."""
import gzip, hashlib, json, os, signal, subprocess, time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=w/'larger-prefix-final-v1';baseline=w/'larger-prefix-baseline';python=root/'.venv/bin/python'
assert not out.exists()
assert '1292 passed' in (w/'larger-prefix-full-suite-v1.log').read_text()
source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
base_source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=baseline,text=True).strip()
assert base_source=='df74493c97633324147cb39ef4c41a7d34c7b90a'
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
        try:code=p.wait(timeout=600)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM);p.wait();code=124
    row=dict(name=name,phase=phase,command=cmd,cwd=str(cwd),source=base_source if cwd==baseline else source,exit_code=code,timeout_seconds=600,wall_seconds=time.time()-start)
    path=Path(cmd[cmd.index('--output')+1]);path=path/'result.json' if path.is_dir() else path
    if path.exists():
        d=json.loads(path.read_text());row.update(result=str(path),result_sha256=sha(path),valid=d.get('valid'),error=d.get('error'))
        row['median_ms']=d.get('median_gpu_milliseconds',d.get('median_additional_replay_ms',d.get('median_ms')))
        row['quota_clean']=all(v['throttled_periods']==0 for k in ['cpu_quota_delta','initial_replays_cpu_quota_delta','additional_replays_cpu_quota_delta','first_replay_cpu_quota_delta','initial_cpu_quota_delta'] if k in d for v in d[k].values())
        if 'full' in d:row.update(median_ms=d['full']['median_ms'],quota_clean=all(v['throttled_periods']==0 for v in d['full']['cpu'].values()))
        if 'median_seconds' in d:row.update(median_ms=1000*d['median_seconds'],quota_clean=not d['cpu_throttled'])
    rows.append(row);(out/'commands.json').write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True)
    assert code in [0,1] and path.exists() and row['quota_clean'],row
    if 'armadillo' in name and row['valid']==0:
        assert code==1 and row['error']==64 and not any(d['counts']),row
    else:
        assert row['valid'] in [None,1] and code==0 and row['error'] in [None,0],row
    if 'armadillo' in name:
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


# Seven original complete-solve controls, plus both newly useful reservations.
controls=json.loads((w/'prepared-new-cell-final-v2/commands.json').read_text())
names=['torus16384-fixed32','torus8192-progressive','reserved16384-target4096','graded-torus16384','torus4096','torus1000','torus8192-fixed32']
lookup={r['name']:r for r in controls if r['phase']=='after'}
for name in [*names,'reserved8192-target4096','reserved16384-target8192']:
    ref=lookup[name] if name in lookup else lookup['reserved16384-target4096']
    for phase,cwd in [('before',baseline),('after',root)]:
        cmd=ref['command'][:];assert cmd[1]=='-m'
        gallery=cmd[2]=='examples.vtm_gallery'
        if name=='reserved8192-target4096':cmd[cmd.index('--capacity')+1]='8192'
        if name=='reserved16384-target8192':cmd[cmd.index('--n')+1]='8192'
        cmd[cmd.index('--output')+1]=str(out/(name+'-'+phase) if gallery else out/(name+'-'+phase+'.json'))
        if '--save-mesh' in cmd:cmd[cmd.index('--save-mesh')+1]=str(out/(name+'-'+phase+'.npz'))
        run(name,phase,cmd,cwd)
# Retain known rejection behavior; these are not successful-mesh timings.
for file,name in [('extended-reuse-final-v1/commands.json','uniform-armadillo8192'),('extended-reuse-gallery-dense-v1/commands.json','armadillo8192-graded-q2')]:
    refs=json.loads((w/file).read_text());ref=next(r for r in refs if r['name']==name and r['phase']=='before')
    for phase,cwd in [('before',baseline),('after',root)]:
        cmd=ref['command'][:]
        if cmd[1]!='-m':cmd[1:2]=['-m','examples.vtm_gallery']
        cmd[cmd.index('--output')+1]=str(out/(name+'-'+phase));run(name,phase,cmd,cwd)
for name in ['delaunay1000','delaunay100000','delaunay-grid','delaunay-fitted-hand','diagram100000']:
    for phase,cwd in [('before',baseline),('after',root)]:
        cmd=lookup[name]['command'][:];cmd[cmd.index('--output')+1]=str(out/(name+'-'+phase+'.json'));run(name,phase,cmd,cwd)
for phase,cwd in [('before',baseline),('after',root)]:
    cmd=[str(python),str(w/'bench-larger-prefix-production-grid-v1.py'),'--output',str(out/('retained-grid-'+phase+'.json'))]
    run('retained-grid',phase,cmd,cwd)
assert len(rows)==34
