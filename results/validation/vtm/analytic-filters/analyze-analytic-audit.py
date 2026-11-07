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
    stem=w/f'analytic-{name}-native'
    dot=stem.with_suffix('.dot')
    if not dot.exists() and dot.with_suffix('.dot.gz').exists():dot.write_bytes(gzip.decompress(dot.with_suffix('.dot.gz').read_bytes()))
    assert stem.with_suffix('.nsys-rep').exists() and dot.exists()
    with tempfile.TemporaryDirectory(prefix='warpower-derived-audit-',dir='/dev/shm') as directory:
        dbpath=Path(directory)/'trace.sqlite'
        with (w/f'analytic-{name}-native-export.log').open('w') as log:
            subprocess.run(['nsys','export','--type','sqlite','--force-overwrite=true','--output',str(dbpath),str(stem.with_suffix('.nsys-rep'))],stdout=log,stderr=subprocess.STDOUT,check=True)
        scope=json.loads(stem.with_suffix('.json').read_text())['scope']
        with (w/f'analytic-{name}-native-audit.log').open('w') as log:
            subprocess.run([str(root/'.venv/bin/python'),'-m','benchmarks.graph_capture','--analyze',str(dbpath),'--dot',str(stem.with_suffix('.dot')),'--scope',scope,'--output',str(w/f'analytic-{name}-native-audit.json')],stdout=log,stderr=subprocess.STDOUT,check=True)
        with sqlite3.connect(dbpath) as db:
            sql='select s.value kernel,k.registersPerThread,k.blockX,count(*) calls,sum(k.end-k.start)/1e6 milliseconds from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName group by s.value,k.registersPerThread,k.blockX order by milliseconds desc'
            kernels=[dict(zip(['kernel','registersPerThread','blockX','calls','milliseconds'],r)) for r in db.execute(sql)]
        compressed=w/f'analytic-{name}-native.sqlite.gz'
        with dbpath.open('rb') as src,compressed.open('wb') as dest:
            with gzip.GzipFile(fileobj=dest,mode='wb',mtime=0) as packed:shutil.copyfileobj(src,packed)
        original_digest=sha(dbpath);h=hashlib.sha256()
        with gzip.open(compressed,'rb') as src:
            for data in iter(lambda:src.read(2**20),b''):h.update(data)
        assert h.hexdigest()==original_digest
        rows[name]=dict(scope=scope,nsys_sha256=sha(stem.with_suffix('.nsys-rep')),sqlite_sha256=original_digest,sqlite_gz_sha256=sha(compressed),kernel_activities=sum(r['calls'] for r in kernels),kernel_activity_milliseconds=sum(r['milliseconds'] for r in kernels),kernels=kernels)
    dot=stem.with_suffix('.dot');data=dot.read_bytes();packed=dot.with_suffix('.dot.gz');packed.write_bytes(gzip.compress(data,mtime=0))
    assert gzip.decompress(packed.read_bytes())==data
    rows[name]['dot_sha256']=sha(dot);rows[name]['dot_gz_sha256']=sha(packed)
    dot.unlink()  # Exact persistent gzip copy verified before removing the raw duplicate.
    (w/'analytic-native-audit-summary.json').write_text(json.dumps(rows,indent=2)+'\n')
(w/'analytic-native-audit-summary.json').write_text(json.dumps(rows,indent=2)+'\n')
for name,row in rows.items():print(name,row['kernel_activities'],row['kernel_activity_milliseconds'],flush=True)
