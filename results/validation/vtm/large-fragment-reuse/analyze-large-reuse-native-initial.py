"""Derive kernel attribution from persistent reports using disposable SQLite scratch.

Only reproducible nsys export scratch is ephemeral. Original reports, source,
export logs, compact summaries and losslessly compressed SQLite stay persistent.
"""
import gzip,hashlib,json,shutil,sqlite3,subprocess,tempfile
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';rows={}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for part in iter(lambda:f.read(2**20),b''):h.update(part)
 return h.hexdigest()
for name in ['control','reuse']:
 stem=w/f'large-reuse-native-{name}';source=stem.with_suffix('.nsys-rep');assert source.exists()
 with tempfile.TemporaryDirectory(prefix='warpower-derived-nsys-',dir='/opt') as directory:
  dbpath=Path(directory)/'trace.sqlite'
  cmd=['nsys','export','--type','sqlite','--force-overwrite=true','--output',str(dbpath),str(source)]
  with (w/f'large-reuse-native-{name}-export.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
  with sqlite3.connect(dbpath) as db:
   sql='select s.value kernel,k.registersPerThread,k.blockX,count(*) calls,sum(k.end-k.start)/1e6 milliseconds from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName group by s.value,k.registersPerThread,k.blockX order by milliseconds desc'
   kernels=[dict(zip(['kernel','registersPerThread','blockX','calls','milliseconds'],r)) for r in db.execute(sql)]
  compressed=w/f'large-reuse-native-{name}.sqlite.gz'
  with dbpath.open('rb') as f,compressed.open('wb') as dest:
   with gzip.GzipFile(fileobj=dest,mode='wb',mtime=0) as packed:shutil.copyfileobj(f,packed)
  digest=sha(dbpath);h=hashlib.sha256()
  with gzip.open(compressed,'rb') as f:
   for part in iter(lambda:f.read(2**20),b''):h.update(part)
  assert h.hexdigest()==digest
  rows[name]=dict(source_sha256=sha(source),sqlite_uncompressed_sha256=digest,sqlite_gz_sha256=sha(compressed),sqlite_uncompressed_bytes=dbpath.stat().st_size,sqlite_gz_bytes=compressed.stat().st_size,kernel_activities=sum(r['calls'] for r in kernels),kernel_activity_milliseconds=sum(r['milliseconds'] for r in kernels),kernels=kernels)
report=dict(scope='One fresh complete torus4096 solve each; native activity durations are diagnostic and exclude gaps/other activity. Not clean median timings.',data=rows)
(w/'large-reuse-native-summary.json').write_text(json.dumps(report,indent=2)+'\n')
for name,r in rows.items():
 print(name,'activities',r['kernel_activities'],'kernel sum ms',r['kernel_activity_milliseconds'])
 for k in r['kernels'][:12]:print(k)
