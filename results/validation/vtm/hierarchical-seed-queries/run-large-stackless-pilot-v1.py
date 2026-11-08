"""Fresh interleaved controls/candidates; offline validation runs after all timing."""
import gzip,hashlib,json,os,signal,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=w/'large-stackless-pilot-v1';assert not out.exists()
source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();assert source=='5998b1384e97c04532fd5f928feb0e09c2fa702a'
assert not subprocess.check_output(['git','status','--porcelain'],text=True)
assert '106 passed' in (w/'large-stackless-primitive-v1.log').read_text()
assert 'failed' not in (w/'large-stackless-existing-v1.log').read_text() and 'passed' in (w/'large-stackless-existing-v1.log').read_text()
controls=json.loads((w/'extended-reuse-final-v1/commands.json').read_text())
lookup={r['name']:r for r in controls if r['phase']=='after'}
selected=['torus16384-fixed32','torus8192-progressive','reserved16384-target4096','graded-torus16384','torus4096','torus1000']
out.mkdir();rows=[];archives=[]
prototype_sha256=hashlib.sha256((w/'large_stackless_v1.py').read_bytes()).hexdigest()
for name in selected:
 for phase in ['before','after']:
    cmd=lookup[name]['command'][:]
    gallery=cmd[2]=='examples.vtm_gallery';assert cmd[1]=='-m'
    cmd[1:3]=[str(w/('run-large-stackless-gallery-v1.py' if gallery else 'run-large-stackless-mesher-v1.py')),*(['--candidate'] if phase=='after' else [])]
    dest=out/(name+'-'+phase)
    cmd[cmd.index('--output')+1]=str(dest if gallery else dest.with_suffix('.json'))
    if '--save-mesh' in cmd:cmd[cmd.index('--save-mesh')+1]=str(dest.with_suffix('.npz'))
    log=out/(name+'-'+phase+'.log');start=time.time()
    with log.open('w') as f:
        p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=p.wait(timeout=300)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);p.wait();code=124
    row=dict(name=name,phase=phase,command=cmd,source=source,prototype_sha256=prototype_sha256,exit_code=code,wall_seconds=time.time()-start,timeout_seconds=300)
    result=dest/'result.json' if gallery else dest.with_suffix('.json')
    if result.exists():
        d=json.loads(result.read_text());row.update(result=str(result),result_sha256=hashlib.sha256(result.read_bytes()).hexdigest(),valid=d['valid'],error=d['error'],counts=d['counts'],median_ms=d.get('median_gpu_milliseconds',d.get('median_additional_replay_ms')),quota_clean=all(v['throttled_periods']==0 for k in ['cpu_quota_delta','initial_replays_cpu_quota_delta','first_replay_cpu_quota_delta','additional_replays_cpu_quota_delta'] if k in d for v in d[k].values()))
    rows.append(row);(out/'commands.json').write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True)
    assert code==0 and row['valid']==1 and row['error']==0 and row['quota_clean'],row
    if gallery:
        for name2 in ['mesh.vtu','boundary.off']:
            p=dest/name2;raw=p.read_bytes();packed=p.with_suffix(p.suffix+'.gz');assert not packed.exists();packed.write_bytes(gzip.compress(raw,mtime=0));assert gzip.decompress(packed.read_bytes())==raw
            archives.append(dict(path=str(packed),sha256=hashlib.sha256(packed.read_bytes()).hexdigest(),raw_sha256=hashlib.sha256(raw).hexdigest()));(out/'lossless-archives.json').write_text(json.dumps(archives,indent=2)+'\n');p.unlink()
assert len(rows)==12
