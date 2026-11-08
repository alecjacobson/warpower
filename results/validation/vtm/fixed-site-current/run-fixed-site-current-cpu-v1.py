"""Larger native size controls after all GPU timing; retain every native sample."""
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
out=w/'fixed-site-current-cpu-v1'
assert not out.exists()
assert len(json.loads((w/'fixed-site-current-larger-v1/commands.json').read_text()))==4
out.mkdir()
rows=[]
archives=[]
# h^-3 extrapolation from the published 0.163/3932-vertex sequential control.
# These are size pilots, not matched-count or matched-quality assertions.
for label,size in [('near8192',0.128),('near16384',0.102)]:
    for mode,threads in [('sequential',1),('parallel',4)]:
        name=f'{label}-{mode}'
        dest=out/name
        cmd=[str(root/'.venv/bin/python'),str(w/'run-fixed-site-current-native-v1.py'),'--binary',str(w/f'cpu-mesh3-target-{mode}'),'--surface',str(root/'results/validation/vtm/cpu-mesh3-torus-input.off'),'--size',str(size),'--facet-distance','0.025','--odt-iterations','50','--convergence','0.02','--freeze-bound','0.01','--threads',str(threads),'--repeats','3','--output',str(dest)]
        start=time.time()
        with (out/(name+'.log')).open('w') as f:
            p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                code=p.wait(timeout=600)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGTERM)
                p.wait()
                code=124
        row=dict(name=name,command=cmd,exit_code=code,timeout_seconds=600,wall_seconds=time.time()-start)
        if (dest/'result.json').exists():
            d=json.loads((dest/'result.json').read_text())
            row.update(median_seconds=d['median_complete_seconds'],unthrottled_samples=d['unthrottled_samples'],referenced_vertices=[r['independent_quality']['referenced_vertices'] for r in d['runs']])
        rows.append(row)
        (out/'commands.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(row,flush=True)
        for p in dest.glob('mesh-*.txt'):
            raw=p.read_bytes()
            q=p.with_suffix('.txt.gz')
            assert not q.exists()
            q.write_bytes(gzip.compress(raw,mtime=0))
            assert gzip.decompress(q.read_bytes())==raw
            archives.append(dict(path=str(q),sha256=hashlib.sha256(q.read_bytes()).hexdigest(),raw_sha256=hashlib.sha256(raw).hexdigest()))
            (out/'lossless-archives.json').write_text(json.dumps(archives,indent=2)+'\n')
            p.unlink()
