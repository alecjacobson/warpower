"""Preserve the native trace and exact kernel summary; SQLite is derived scratch."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import tempfile
import time

root=Path.cwd(); w=root/'.work/vtm'; label='tangential-production-uniform-native-v1'; stem=w/label; python=root/'.venv/bin/python'
dot=stem.with_suffix('.dot'); assert not dot.exists()
cmd=['nsys','profile','--trace=cuda','--cuda-graph-trace=node','--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=true','-o',str(stem),str(python),str(w/'profile-tangential-production-uniform-v1.py'),'--output',str(stem.with_suffix('.json')),'--dot',str(dot)]
start=time.time()
with stem.with_suffix('.log').open('w') as log:
    p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:code=p.wait(timeout=600)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid,signal.SIGTERM);p.wait();code=124
row=dict(command=cmd,exit_code=code,wall_seconds=time.time()-start,timeout_seconds=600)
(w/(label+'-command.json')).write_text(json.dumps(row,indent=2)+'\n');assert code==0,row
raw=stem.with_suffix('.nsys-rep'); original=raw.read_bytes(); packed=raw.with_suffix('.nsys-rep.gz');assert not packed.exists()
packed.write_bytes(gzip.compress(original,mtime=0));assert gzip.decompress(packed.read_bytes())==original
raw_digest=hashlib.sha256(original).hexdigest()
with tempfile.TemporaryDirectory(prefix='proven-recovery-profile-',dir='/dev/shm') as tmp:
    dbpath=Path(tmp)/'trace.sqlite'
    with (w/(label+'-export.log')).open('w') as log:
        subprocess.run(['nsys','export','--type','sqlite','--output',str(dbpath),str(raw)],stdout=log,stderr=subprocess.STDOUT,check=True)
    with (w/(label+'-audit.log')).open('w') as log:
        subprocess.run([str(python),'-m','benchmarks.graph_capture','--analyze',str(dbpath),'--dot',str(dot),'--scope','Production bounded tangential recovery and full uniform8K solve; profiling only','--output',str(w/(label+'-audit.json'))],stdout=log,stderr=subprocess.STDOUT,check=True)
    with sqlite3.connect(dbpath) as db:
        sql='select s.value,k.registersPerThread,k.blockX,count(*),sum(k.end-k.start)/1e6 from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName group by s.value,k.registersPerThread,k.blockX order by sum(k.end-k.start) desc'
        rows=[dict(zip(['kernel','registers','block_size','calls','milliseconds'],r)) for r in db.execute(sql)]
        events=list(db.execute('select s.value,k.start,k.end from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName order by k.start'))
        windows={}
        for stage in ['topology','quality']:
            begin=[e for e in events if e[0].startswith(stage+'_begin_')]
            end=[e for e in events if e[0].startswith(stage+'_end_')]
            assert len(begin)==len(end)==1,(stage,begin,end)
            lo,hi=begin[0][2],end[0][1]
            selected=[e for e in events if e[1]>=lo and e[2]<=hi]
            aggregate={}
            for name,st,finish in selected:
                v=aggregate.setdefault(name,dict(kernel=name,calls=0,milliseconds=0.0));v['calls']+=1;v['milliseconds']+=(finish-st)/1e6
            windows[stage]=dict(elapsed_ms=(hi-lo)/1e6,kernel_ms=sum(e[2]-e[1] for e in selected)/1e6,kernel_count=len(selected),kernels=sorted(aggregate.values(),key=lambda r:r['milliseconds'],reverse=True))
    (w/(label+'-windows.json')).write_text(json.dumps(windows,indent=2)+'\n')
    result=dict(trace_raw_sha256=raw_digest,trace_gz_sha256=hashlib.sha256(packed.read_bytes()).hexdigest(),kernels=rows,total_kernel_milliseconds=sum(r['milliseconds'] for r in rows))
    (w/(label+'-kernels.json')).write_text(json.dumps(result,indent=2)+'\n')
raw.unlink()
data=dot.read_bytes();dest=stem.with_suffix('.dot.gz');assert not dest.exists();dest.write_bytes(gzip.compress(data,mtime=0));assert gzip.decompress(dest.read_bytes())==data
dot.unlink()
print(json.dumps(rows[:12],indent=2),flush=True)
