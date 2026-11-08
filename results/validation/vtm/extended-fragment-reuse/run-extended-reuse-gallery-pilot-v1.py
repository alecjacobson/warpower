"""Clean graded large-mesh control/candidate pairs; independent checks follow timing."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

root=Path.cwd()
w=root/'.work/vtm'
out=w/'extended-reuse-gallery-pilot-v1'
assert not out.exists()
source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
assert source=='689fd31382c490251b953046176b15ccc35e5753'
assert not subprocess.check_output(['git','status','--porcelain'],text=True)
assert not subprocess.check_output(['git','diff','b44ab9e','--','warpower','tests','benchmarks','examples','README.md'])
out.mkdir()
rows=[]
archives=[]
asset=w/'scanned-armadillo-input/armadillo.off'
cases=[('armadillo8192-graded','graded-surface',8192,['--input',str(asset),'--subdivisions','1','--crease-subdivisions','1','--feature-angle','120']),('torus16384-graded','graded-torus',16384,[])]
for name,case,n,extra in cases:
    for phase in ['before','after']:
        dest=out/(name+'-'+phase)
        cmd=[str(root/'.venv/bin/python'),str(w/'run-extended-reuse-gallery-v1.py'),*(['--candidate'] if phase=='after' else []),'--case',case,'--n',str(n),'--initial','256','--quality-rounds','4','--repeats','3',*extra,'--output',str(dest)]
        start=time.time()
        with (out/(name+'-'+phase+'.log')).open('w') as f:
            p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                code=p.wait(timeout=600)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGTERM)
                p.wait()
                code=124
        row=dict(name=name,phase=phase,command=cmd,source=source,exit_code=code,timeout_seconds=600,wall_seconds=time.time()-start)
        if extra:
            row['input_sha256']=hashlib.sha256(asset.read_bytes()).hexdigest()
        if (dest/'result.json').exists():
            d=json.loads((dest/'result.json').read_text())
            row.update(valid=d['valid'],error=d['error'],counts=d['counts'],median_ms=d['median_additional_replay_ms'],quota_clean=all(v['throttled_periods']==0 for k in ['first_replay_cpu_quota_delta','additional_replays_cpu_quota_delta'] for v in d[k].values()),result_sha256=hashlib.sha256((dest/'result.json').read_bytes()).hexdigest())
        rows.append(row)
        (out/'commands.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(row,flush=True)
        for filename in ['mesh.vtu','boundary.off']:
            p=dest/filename
            if not p.exists():
                continue
            raw=p.read_bytes()
            packed=p.with_suffix(p.suffix+'.gz')
            assert not packed.exists()
            packed.write_bytes(gzip.compress(raw,mtime=0))
            assert gzip.decompress(packed.read_bytes())==raw
            archives.append(dict(path=str(packed),sha256=hashlib.sha256(packed.read_bytes()).hexdigest(),raw_sha256=hashlib.sha256(raw).hexdigest()))
            (out/'lossless-archives.json').write_text(json.dumps(archives,indent=2)+'\n')
            p.unlink()
