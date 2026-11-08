"""Pinned production/baseline full solves and standalone nonregression pairs."""
import gzip, hashlib, json, os, signal, subprocess, time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=w/'hierarchical-cache-final-v1';baseline=w/'hierarchical-cache-baseline';python=root/'.venv/bin/python'
assert not out.exists()
assert '1237 passed' in (w/'hierarchical-cache-full-suite-v1.log').read_text()
source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
base_source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=baseline,text=True).strip()
assert base_source=='7c3dfc74acb4e947636e428bc0ea866f66dfdf77'
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
    if row['valid']==0:assert code==1 and row['error']==64 and not any(d['counts']),row
    else:assert code==0 and row['error'] in [None,0],row
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


# Repeat the six successful pilot configurations using committed production.
controls=json.loads((w/'large-stackless-pilot-v1/commands.json').read_text())
for ref in controls:
    if ref['phase']!='before':continue
    name=ref['name']
    for phase,cwd in [('before',baseline),('after',root)]:
        cmd=ref['command'][:]
        gallery='gallery' in cmd[1]
        cmd[1:2]=['-m','examples.vtm_gallery' if gallery else 'benchmarks.mesher']
        cmd[cmd.index('--output')+1]=str(out/(name+'-'+phase) if gallery else out/(name+'-'+phase+'.json'))
        if '--save-mesh' in cmd:cmd[cmd.index('--save-mesh')+1]=str(out/(name+'-'+phase+'.npz'))
        run(name,phase,cmd,cwd)
old=json.loads((w/'boundary-timing-commands.json').read_text())
for ref in old:
    name=ref['name']
    if ref['phase']!='after' or not name.startswith(('delaunay','diagram')):continue
    for phase,cwd in [('before',baseline),('after',root)]:
        cmd=ref['command'][:];cmd[cmd.index('--output')+1]=str(out/(name+'-'+phase+'.json'));run(name,phase,cmd,cwd)
# Include the fixed-iteration 8k case and the tie-heavy retained-grid guard.
refs=json.loads((w/'hierarchical-split-pilot-v2/commands.json').read_text())
ref=next(r for r in refs if r['name']=='torus8192-fixed32' and r['phase']=='before')
for phase,cwd in [('before',baseline),('after',root)]:
    cmd=ref['command'][:];cmd[1:2]=['-m','benchmarks.mesher']
    cmd[cmd.index('--output')+1]=str(out/('torus8192-fixed32-'+phase+'.json'))
    cmd[cmd.index('--save-mesh')+1]=str(out/('torus8192-fixed32-'+phase+'.npz'))
    run('torus8192-fixed32',phase,cmd,cwd)
for phase,cwd in [('before',baseline),('after',root)]:
    cmd=[str(python),str(w/'bench-hierarchical-cache-grid-v1.py'),'--output',str(out/('retained-grid-'+phase+'.json'))]
    run('retained-grid',phase,cmd,cwd)
assert len(rows)==26
