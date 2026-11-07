"""Native capture audit; only reproducible SQLite export scratch uses shared memory."""
import gzip,hashlib,json,shutil,sqlite3,subprocess,tempfile
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';rows={}
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for data in iter(lambda:f.read(2**20),b''):h.update(data)
    return h.hexdigest()
for name in ['full','loop']:
    stem=w/f'recovery-classification-{name}-native'
    dot=stem.with_suffix('.dot')
    if not dot.exists():
        derived=Path('/opt/warpower-generated-cache/graph-derived');derived.mkdir(parents=True,exist_ok=True)
        dot=derived/(stem.name+'.dot')
        dot.write_bytes(gzip.decompress(stem.with_suffix('.dot.gz').read_bytes()))
    report=stem.with_suffix('.nsys-rep')
    if not report.exists():
        derived=Path('/opt/warpower-generated-cache/native-derived');derived.mkdir(parents=True,exist_ok=True)
        report=derived/(stem.name+'.nsys-rep')
        report.write_bytes(gzip.decompress(stem.with_suffix('.nsys-rep.gz').read_bytes()))
    assert report.exists() and dot.exists()
    with tempfile.TemporaryDirectory(prefix='warpower-derived-audit-',dir='/dev/shm') as directory:
        dbpath=Path(directory)/'trace.sqlite'
        with (w/f'recovery-classification-{name}-native-export.log').open('w') as log:
            subprocess.run(['nsys','export','--type','sqlite','--force-overwrite=true','--output',str(dbpath),str(report)],stdout=log,stderr=subprocess.STDOUT,check=True)
        scope=json.loads(stem.with_suffix('.json').read_text())['scope']
        with (w/f'recovery-classification-{name}-native-audit.log').open('w') as log:
            subprocess.run([str(root/'.venv/bin/python'),'-m','benchmarks.graph_capture','--analyze',str(dbpath),'--dot',str(dot),'--scope',scope,'--output',str(w/f'recovery-classification-{name}-native-audit.json')],stdout=log,stderr=subprocess.STDOUT,check=True)
        with sqlite3.connect(dbpath) as db:
            sql='select s.value kernel,k.registersPerThread,k.blockX,count(*) calls,sum(k.end-k.start)/1e6 milliseconds from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName group by s.value,k.registersPerThread,k.blockX order by milliseconds desc'
            kernels=[dict(zip(['kernel','registersPerThread','blockX','calls','milliseconds'],r)) for r in db.execute(sql)]
        # SQLite is a reproducible nsys export; preserve the authoritative
        # .nsys-rep.gz and all derived audit counters, not a second large archive.
        original_digest=sha(dbpath)
        rows[name]=dict(scope=scope,nsys_sha256=sha(report),sqlite_sha256=original_digest,sqlite_storage='Regenerable export, not archived separately; see exact nsys export command in this script.',kernel_activities=sum(r['calls'] for r in kernels),kernel_activity_milliseconds=sum(r['milliseconds'] for r in kernels),kernels=kernels)
    if name=='full':assert any(r['kernel'].startswith('walk_') and r['calls']>0 for r in kernels)
    data=dot.read_bytes();packed=stem.with_suffix('.dot.gz')
    if not packed.exists():packed.write_bytes(gzip.compress(data,mtime=0))
    assert gzip.decompress(packed.read_bytes())==data
    rows[name]['dot_sha256']=sha(dot);rows[name]['dot_gz_sha256']=sha(packed)
    # The graph DOT is a derived setup export. Its exact compressed copy is persistent.
    (w/'recovery-classification-native-audit-summary.json').write_text(json.dumps(rows,indent=2)+'\n')
(w/'recovery-classification-native-audit-summary.json').write_text(json.dumps(rows,indent=2)+'\n')
for name,row in rows.items():print(name,row['kernel_activities'],row['kernel_activity_milliseconds'],flush=True)
