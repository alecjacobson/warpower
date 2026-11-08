"""Declared seed/shape grid; preserve failures and gzip redundant export encodings."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

root=Path.cwd();w=root/'.work/vtm';out=w/'proven-recovery-broad-v1';python=root/'.venv/bin/python'
assert not out.exists();assert (w/'hand-full-proven-recovery-v2-independent.json').exists();out.mkdir()
rows=[];archives=[]
runtime=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
cases=[('hand-seed'+str(seed),'thin-hand',1000,['--seed',str(seed)],['before','candidate']) for seed in [0,1,2,7,42,2026]]
cases += [('coarse-hand','thin-hand',1000,['--subdivisions','4','--crease-subdivisions','4'],['before','candidate']),
          ('plate','counterbored-plate',1000,[],['before','candidate']),
          ('coarse-cavity','cavity',1000,['--subdivisions','4','--crease-subdivisions','4'],['before','candidate']),
          ('coarse-disconnected','disconnected',1000,['--subdivisions','4','--crease-subdivisions','4'],['before','candidate']),
          ('torus4096','torus',4096,[],['before','candidate']),
          ('graded-armadillo','graded-surface',4096,['--input',str(w/'scanned-armadillo-input/armadillo.off'),'--subdivisions','1','--crease-subdivisions','1','--feature-angle','120'],['before','candidate'])]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def retain_exports(folder):
    for name in ['mesh.vtu','boundary.off']:
        p=folder/name
        if not p.exists():continue
        data=p.read_bytes();dest=p.with_suffix(p.suffix+'.gz');assert not dest.exists()
        dest.write_bytes(gzip.compress(data,mtime=0));assert gzip.decompress(dest.read_bytes())==data
        archives.append(dict(original=str(p),path=str(dest),raw_sha256=hashlib.sha256(data).hexdigest(),sha256=sha(dest),raw_bytes=len(data),bytes=dest.stat().st_size))
        (out/'lossless-archives.json').write_text(json.dumps(archives,indent=2)+'\n');p.unlink()


for name,case,n,extra,phases in cases:
    for phase in phases:
        if name=='hand-seed7' and phase=='candidate':
            folder=w/'hand-full-proven-recovery-v2/seed7';r=json.loads((folder/'result.json').read_text())
            rows.append(dict(name=name,phase=phase,source=runtime,reused_existing_complete_record=True,result=str(folder/'result.json'),result_sha256=sha(folder/'result.json'),exit_code=0,valid=r['valid'],error=r['error'],median_ms=r['median_additional_replay_ms']))
            (out/'commands.json').write_text(json.dumps(rows,indent=2)+'\n');print(rows[-1],flush=True);continue
        folder=out/(name+'-'+phase)
        cmd=[str(python),*(['-m','examples.vtm_gallery'] if phase=='before' else [str(w/'run-full-proven-recovery-v2.py')]),
             '--case',case,'--n',str(n),'--initial','256','--quality-rounds','4','--repeats','3',*extra,'--output',str(folder)]
        log=out/(name+'-'+phase+'.log');assert not log.exists();start=time.time()
        with log.open('w') as stream:
            p=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=p.wait(timeout=300)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGTERM);p.wait();code=124
        row=dict(name=name,phase=phase,source=runtime,command=cmd,exit_code=code,wall_seconds=time.time()-start,timeout_seconds=300)
        path=folder/'result.json'
        if path.exists():
            r=json.loads(path.read_text());row.update(result=str(path),result_sha256=sha(path),valid=r['valid'],error=r['error'],median_ms=r['median_additional_replay_ms'],counts=r['counts'])
            row['timed_quota_clean']=all(v['throttled_periods']==0 for k in ['first_replay_cpu_quota_delta','additional_replays_cpu_quota_delta'] for v in r[k].values())
            if phase=='candidate':row['pilot']=json.loads((folder/'pilot.json').read_text())['instances']
        rows.append(row);(out/'commands.json').write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True)
        assert code in [0,1] and path.exists(),row
        assert (code==0 and r['valid']==1 and r['error']==0) or (code==1 and r['valid']==0 and r['error']==64 and not any(r['counts'])),row
        assert row['timed_quota_clean'],row
        if name=='graded-armadillo':
            p=folder/'input.off';canonical=w/'scanned-armadillo-input/armadillo.off';assert sha(p)==sha(canonical)
            p.unlink();os.link(canonical,p)
        retain_exports(folder)
assert len(rows)==24
