"""Pinned production/baseline full solves and standalone nonregression pairs."""
import gzip, hashlib, json, os, signal, subprocess, time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=w/'exterior-bvh-final-v1';baseline=w/'exterior-bvh-baseline';python=root/'.venv/bin/python'
assert not out.exists()
assert '1317 passed' in (w/'exterior-bvh-full-suite-v1.log').read_text()
source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
base_source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=baseline,text=True).strip()
assert base_source=='6f878c7098b525e4cfdbda5e8d2ed45cf8b7e736'
assert not subprocess.check_output(['git','status','--porcelain'],text=True)
assert not subprocess.check_output(['git','status','--porcelain'],cwd=baseline,text=True)
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.'+os.pathsep+str(w))
assert env.get('WARP_CACHE_PATH')=='/opt/warpower-generated-cache'
out.mkdir();rows=[];archives=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(name,phase,cmd,cwd):
    log=out/(name+'-'+phase+'.log');start=time.time()
    with log.open('w') as stream:
        runenv=dict(env)
        if 'run-tangential-topology-gallery-v2.py' in cmd[1]:runenv['WARPOWER_TOPOLOGY_DIAGNOSTICS']=str(out/(name+'-'+phase+'-recovery.json'))
        p=subprocess.Popen(cmd,cwd=cwd,env=runenv,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
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


controls=json.loads((w/'larger-prefix-final-v1/commands.json').read_text())
lookup={r['name']:r for r in controls if r['phase']=='after'}
for name in ['torus16384-fixed32','torus1000']:
 for phase,cwd in [('before',baseline),('after',root)]:
  cmd=lookup[name]['command'][:]
  cmd[cmd.index('--output')+1]=str(out/(name+'-'+phase+'.json'))
  cmd[cmd.index('--save-mesh')+1]=str(out/(name+'-'+phase+'.npz'))
  run(name,phase,cmd,cwd)
cases=[('thin-hand-seed7','thin-hand',1000,['--seed','7'],False),
       ('cavity','cavity',1000,[],False),('disconnected','disconnected',1000,[],False),
       ('plate','counterbored-plate',1000,[],False),
       ('armadillo-graded4096','graded-surface',4096,[],False),
       ('armadillo-graded8192-recovery','graded-surface',8192,[],True),
       ('armadillo-uniform8192-recovery','surface',8192,[],True)]
for name,case,n,extra,experimental in cases:
 for phase,cwd in [('before',baseline),('after',root)]:
  cmd=[str(python)]+([str(w/'run-tangential-topology-gallery-v2.py')] if experimental else ['-m','examples.vtm_gallery'])
  cmd+=['--case',case,'--n',str(n),'--initial','256','--repeats','3',*extra]
  if 'armadillo' in name:
   cmd+=['--input',str(w/'scanned-armadillo-input/armadillo.off'),'--subdivisions','1','--crease-subdivisions','1','--feature-angle','120','--corner-angle','45']
  cmd+=['--output',str(out/(name+'-'+phase))]
  run(name,phase,cmd,cwd)
for name in ['delaunay1000','delaunay100000','delaunay-grid','delaunay-fitted-hand','diagram100000']:
 for phase,cwd in [('before',baseline),('after',root)]:
  cmd=lookup[name]['command'][:];cmd[cmd.index('--output')+1]=str(out/(name+'-'+phase+'.json'));run(name,phase,cmd,cwd)
assert len(rows)==28
assert all(r['valid'] in [None,1] for r in rows)
