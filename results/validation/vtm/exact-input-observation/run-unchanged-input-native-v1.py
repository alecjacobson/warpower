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

import argparse
p=argparse.ArgumentParser();p.add_argument('--loop',action='store_true');a=p.parse_args()
root=Path.cwd(); w=root/'.work/vtm'; label='unchanged-input-fixed16k-native-v1'; stem=w/label; python=root/'.venv/bin/python'
dot=stem.with_suffix('.dot'); assert not dot.exists()
cmd=['nsys','profile','--trace=cuda','--cuda-graph-trace=node','--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=true','-o',str(stem),str(python),str(w/'profile-unchanged-input-v1.py'),'--output',str(stem.with_suffix('.json')),'--dot',str(dot)]
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
        subprocess.run([str(python),'-m','benchmarks.graph_capture','--analyze',str(dbpath),'--dot',str(dot),'--scope','Exact-unchanged input observer; every reconstruction still executes; diagnostic profile only','--output',str(w/(label+'-audit.json'))],stdout=log,stderr=subprocess.STDOUT,check=True)
    with sqlite3.connect(dbpath) as db:
        sql='select s.value,k.registersPerThread,k.blockX,count(*),sum(k.end-k.start)/1e6 from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName group by s.value,k.registersPerThread,k.blockX order by sum(k.end-k.start) desc'
        rows=[dict(zip(['kernel','registers','block_size','calls','milliseconds'],r)) for r in db.execute(sql)]
    with sqlite3.connect(dbpath) as db:
        events=list(db.execute('select s.value,k.start,k.end from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName order by k.start'))
    observations=json.loads(stem.with_suffix('.observations.json').read_text())
    active=[r for r in observations['instances'] if r['records']]
    assert len(active)==1 and active[0]['capacity']==16384
    calls=[r for r in active[0]['records'] if r[0]==2];assert len(calls)==227
    begin=[i for i,e in enumerate(events) if e[0].startswith('record_before_')]
    end=[i for i,e in enumerate(events) if e[0].startswith('record_after_')]
    assert len(begin)==len(end)==len(calls)
    spans=[]
    for row,(b,e) in enumerate(zip(begin,end)):
        assert b<e and (row==0 or end[row-1]<b)
        items=events[b+1:e];lo,hi=events[b][2],events[e][1]
        assert all(lo<=x[1]<=x[2]<=hi for x in items)
        record=calls[row];spans.append(dict(call=row,previous_valid=record[1],changed_sites=record[2],unchanged=bool(record[1]==1 and record[2]==0),kernel_count=len(items),kernel_ms=sum(x[2]-x[1] for x in items)/1e6,span_ms=(hi-lo)/1e6))
    total_span=(max(x[2] for x in events)-min(x[1] for x in events))/1e6
    hit_span=sum(x['span_ms'] for x in spans if x['unchanged'])
    attribution=dict(scope='Diagnostic observer timeline, not a speedup. Windows exclude observer comparison/record/copy work; removing all work in an unchanged window is an optimistic avoided-span estimate, before cache overhead.',calls=spans,total_gpu_timeline_span_ms=total_span,unchanged_reconstruction_span_ms=hit_span,unchanged_span_fraction=hit_span/total_span,unchanged_calls=sum(x['unchanged'] for x in spans),kernel_activities=len(events),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (w/(label+'-attribution.json')).write_text(json.dumps(attribution,indent=2)+'\n')
    result=dict(trace_raw_sha256=raw_digest,trace_gz_sha256=hashlib.sha256(packed.read_bytes()).hexdigest(),kernels=rows,total_kernel_milliseconds=sum(r['milliseconds'] for r in rows))
    (w/(label+'-kernels.json')).write_text(json.dumps(result,indent=2)+'\n')
raw.unlink()
data=dot.read_bytes();dest=stem.with_suffix('.dot.gz');assert not dest.exists();dest.write_bytes(gzip.compress(data,mtime=0));assert gzip.decompress(dest.read_bytes())==data
dot.unlink()
print(json.dumps(rows[:12],indent=2),flush=True)
