"""Fresh interleaved controls/candidates; offline validation runs after all timing."""
import gzip,hashlib,json,os,signal,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=w/'prepared-new-cell-pilot-v1';assert not out.exists()
source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();assert source=='a4f7215fbf99249d11f765a4902c586daa606829'
assert not subprocess.check_output(['git','status','--porcelain'],text=True)
assert '36 passed' in (w/'prepared-new-cell-primitive-v1.log').read_text()
assert 'failed' not in (w/'prepared-new-cell-existing-v1.log').read_text() and 'passed' in (w/'prepared-new-cell-existing-v1.log').read_text()
controls=json.loads((w/'hierarchical-cache-final-v1/commands.json').read_text())
lookup={r['name']:r for r in controls if r['phase']=='after'}
selected=['torus16384-fixed32','graded-torus16384']
out.mkdir();rows=[];archives=[]
prototype_sha256={n:hashlib.sha256((w/n).read_bytes()).hexdigest() for n in ['prepared_new_cell_v1.py','prepared_new_cell_integration_v1.py']}
for name in selected:
 for phase in ['before','after']:
    cmd=lookup[name]['command'][:]
    gallery=cmd[2]=='examples.vtm_gallery';assert cmd[1]=='-m'
    cmd[1:3]=[str(w/('run-prepared-new-cell-gallery-v1.py' if gallery else 'run-prepared-new-cell-mesher-v1.py')),*(['--candidate'] if phase=='after' else [])]
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
assert len(rows)==4
