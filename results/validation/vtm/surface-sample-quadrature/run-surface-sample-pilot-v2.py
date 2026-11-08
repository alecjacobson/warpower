"""Paired complete solves for the qualified private sample-parallel kernel."""
import gzip
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

root=Path.cwd();w=root/'.work/vtm';out=w/'surface-sample-pilot-v2';assert not out.exists()
assert not subprocess.check_output(['git','status','--porcelain'],text=True)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
c=json.loads((w/'surface-sample-parallel-contracts-v1.json').read_text());assert c['complete'] and c['passed']
for p,h in c['source_sha256'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
for name in ['torus1000','torus16384','armadillo-uniform8192','armadillo-graded8192']:
    c=json.loads((w/'surface-sample-static-v1'/(name+'.json')).read_text());assert c['complete'] and all(r['quota_clean'] for r in c['rows'])
    assert c['rows'][1]['median_ms']<c['rows'][0]['median_ms']
out.mkdir();rows=[];archives=[]
controls=json.loads((w/'surface-query-order-final-v1/commands.json').read_text())
for name in ['torus1000','torus16384-fixed32','armadillo-uniform8192-recovery','armadillo-graded8192-recovery']:
    reference=next(c for c in controls if c['name']==name and c['phase']=='after')
    for phase in ['before','after']:
        label=name+'-'+phase;cmd=reference['command'][:]
        if name.startswith('torus'):
            cmd=[cmd[0],str(w/'run-surface-sample-full-v1.py'),*cmd[3:]]
            cmd[cmd.index('--output')+1]=str(out/(label+'.json'))
            cmd[cmd.index('--save-mesh')+1]=str(out/(label+'.npz'))
        else:
            cmd[1]=str(w/'run-surface-sample-full-v1.py')
            cmd[cmd.index('--output')+1]=str(out/label)
        env=dict(os.environ,WARPOWER_SAMPLE_PHASE=phase,WARPOWER_SAMPLE_KIND='benchmark' if name.startswith('torus') else 'gallery',WARPOWER_SAMPLE_REPORT=str(out/(label+'-observer.json')))
        start=time.time()
        with (out/(label+'.log')).open('w') as stream:
            p=subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=p.wait(timeout=600)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);p.wait();code=124
        row=dict(name=name,phase=phase,command=cmd,exit_code=code,wall_seconds=time.time()-start,timeout_seconds=600)
        result=Path(cmd[cmd.index('--output')+1]);result=result/'result.json' if result.is_dir() else result
        if result.exists():
            d=json.loads(result.read_text());keys=['cpu_quota_delta','initial_replays_cpu_quota_delta','additional_replays_cpu_quota_delta','first_replay_cpu_quota_delta','initial_cpu_quota_delta']
            row.update(result=str(result),result_sha256=hashlib.sha256(result.read_bytes()).hexdigest(),valid=d['valid'],error=d['error'],median_ms=d.get('median_gpu_milliseconds',d.get('median_additional_replay_ms')),quota_clean=all(v['throttled_periods']==0 for k in keys if k in d for v in d[k].values()))
        rows.append(row);(out/'commands.json').write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True)
        assert code==0 and row['valid']==1 and row['error']==0 and row['quota_clean'],row
        if 'armadillo' in name:
            duplicate=result.parent/'input.off';canonical=w/'scanned-armadillo-input/armadillo.off'
            assert hashlib.sha256(duplicate.read_bytes()).digest()==hashlib.sha256(canonical.read_bytes()).digest()
            duplicate.unlink();os.link(canonical,duplicate)
        if result.name=='result.json':
            for export_name in ['mesh.vtu','boundary.off']:
                raw=result.parent/export_name;data=raw.read_bytes();packed=raw.with_suffix(raw.suffix+'.gz')
                assert not packed.exists();packed.write_bytes(gzip.compress(data,mtime=0));assert gzip.decompress(packed.read_bytes())==data
                archives.append(dict(original=str(raw),path=str(packed),raw_sha256=hashlib.sha256(data).hexdigest(),sha256=hashlib.sha256(packed.read_bytes()).hexdigest()))
                (out/'lossless-archives.json').write_text(json.dumps(archives,indent=2)+'\n');raw.unlink()
assert len(rows)==8
