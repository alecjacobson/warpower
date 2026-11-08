"""Diagnostic full-benchmark traces: identical existing standalone benchmark, no installer."""
import gzip,hashlib,json,os,sqlite3,subprocess,tempfile,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=w/'prepared-new-cell-hand-trace-v1';assert not out.exists();out.mkdir();rows=[]
for phase,cwd in [('before',w/'prepared-new-cell-baseline'),('after',root)]:
 stem=out/phase;dot=stem.with_suffix('.dot');cmd=['nsys','profile','--trace=cuda','--cuda-graph-trace=node','--sample=none','--cpuctxsw=none','--force-overwrite=true','-o',str(stem),str(root/'.venv/bin/python'),'-m','benchmarks.delaunay_recovery','--case','fitted-hand','--repeats','9','--output',str(stem.with_suffix('.json')),'--dot',str(dot)];start=time.time()
 with stem.with_suffix('.log').open('w') as f:p=subprocess.run(cmd,cwd=cwd,env=dict(os.environ,PYTHONPATH='.'),stdout=f,stderr=subprocess.STDOUT,timeout=300)
 row=dict(phase=phase,command=cmd,cwd=str(cwd),source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=cwd,text=True).strip(),exit_code=p.returncode,wall_seconds=time.time()-start);rows.append(row);(out/'commands.json').write_text(json.dumps(rows,indent=2)+'\n');assert p.returncode==0
 raw=stem.with_suffix('.nsys-rep');data=raw.read_bytes();packed=raw.with_suffix('.nsys-rep.gz');packed.write_bytes(gzip.compress(data,mtime=0));assert gzip.decompress(packed.read_bytes())==data
 archives=dict(raw_sha256=hashlib.sha256(data).hexdigest(),gzip_sha256=hashlib.sha256(packed.read_bytes()).hexdigest())
 with tempfile.TemporaryDirectory(prefix='hand-trace-',dir='/opt/warpower-generated-cache') as tmp:
  dbpath=Path(tmp)/'trace.sqlite'
  with (out/(phase+'-export.log')).open('w') as f:subprocess.run(['nsys','export','--type','sqlite','--output',str(dbpath),str(raw)],stdout=f,stderr=subprocess.STDOUT,check=True)
  with sqlite3.connect(dbpath) as db:
   tables=[r[0] for r in db.execute("select name from sqlite_master where type='table'")];names=['CUPTI_ACTIVITY_KIND_KERNEL','CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER','CUPTI_ACTIVITY_KIND_MEMCPY'];schema={n:list(db.execute('pragma table_info('+n+')')) for n in names if n in tables}
   launch=[]
   for table in names[1:3]:
    if table not in tables:continue
    for r in db.execute('select a.start,a.end,a.correlationId,s.value from '+table+' a join StringIds s on s.id=a.nameId where s.value like "%GraphLaunch%" order by a.start'):
     launch.append(dict(start=r[0],end=r[1],correlation_id=r[2],api=r[3],api_table=table))
   launch.sort(key=lambda r:r['start'])
   for l in launch:
    corr=l['correlation_id'];query='select s.value,k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ,k.registersPerThread,count(*),sum(k.end-k.start)/1e6,min(k.start),max(k.end) from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName where k.correlationId=? group by s.value,k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ,k.registersPerThread order by s.value,k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ,k.registersPerThread';kernels=[dict(zip(['name','grid_x','grid_y','grid_z','block_x','block_y','block_z','registers','calls','milliseconds','first_start','last_end'],r)) for r in db.execute(query,(corr,))];l['kernels']=kernels;l['kernel_calls']=sum(r['calls'] for r in kernels);l['kernel_ms']=sum(r['milliseconds'] for r in kernels);l['api_ms']=(l['end']-l['start'])/1e6
    if kernels:l['gpu_span_ms']=(max(k['last_end'] for k in kernels)-min(k['first_start'] for k in kernels))/1e6;l['first_kernel_after_api_start_ms']=(min(k['first_start'] for k in kernels)-l['start'])/1e6
    shape=[{k:v for k,v in r.items() if k not in ['milliseconds','first_start','last_end']} for r in kernels];l['kernel_signature_sha256']=hashlib.sha256(json.dumps(shape,sort_keys=True).encode()).hexdigest()
   summary=dict(scope='Diagnostic tracing includes setup and all benchmark phases; no clean performance inference',source=row['source'],archives=archives,schema=schema,graph_launches=launch,sqlite_sha256=hashlib.sha256(dbpath.read_bytes()).hexdigest())
   (out/(phase+'-trace-summary.json')).write_text(json.dumps(summary,indent=2)+'\n')
 raw.unlink();data=dot.read_bytes();q=dot.with_suffix('.dot.gz');q.write_bytes(gzip.compress(data,mtime=0));assert gzip.decompress(q.read_bytes())==data;dot.unlink();archives.update(dot_sha256=hashlib.sha256(data).hexdigest(),dot_gzip_sha256=hashlib.sha256(q.read_bytes()).hexdigest());(out/(phase+'-archives.json')).write_text(json.dumps(archives,indent=2)+'\n');print(phase,'graph launches',len(launch),[(l['kernel_calls'],round(l['kernel_ms'],4),round(l.get('gpu_span_ms',0),4),round(l['api_ms'],4)) for l in launch[:12]],flush=True)
