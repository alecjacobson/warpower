"""Include nested conditional-graph kernels by the enclosing launch interval."""
import gzip,hashlib,json,sqlite3,subprocess,tempfile
from pathlib import Path
w=Path('.work/vtm');out=w/'prepared-new-cell-hand-trace-v1';dest=out/'interval-comparison-v1.json';assert not dest.exists();report=dict(scope=__doc__,limitation='Profiler overhead affects timings. Unobserved graph copies can contribute to gaps; no unprofiled speed claim.',cases={})
def union(intervals):
 if not intervals:return 0
 intervals=sorted(intervals);total=0;lo,hi=intervals[0]
 for a,b in intervals[1:]:
  if a>hi:total+=hi-lo;lo,hi=a,b
  else:hi=max(hi,b)
 return total+hi-lo
for phase in ['before','after']:
 prior=json.loads((out/(phase+'-trace-summary.json')).read_text());benchmark=json.loads((out/(phase+'.json')).read_text());assert len(prior['graph_launches'])==42;rows=[]
 with tempfile.TemporaryDirectory(prefix='hand-intervals-',dir='/opt/warpower-generated-cache') as tmp:
  raw=Path(tmp)/'trace.nsys-rep';data=gzip.decompress((out/(phase+'.nsys-rep.gz')).read_bytes());assert hashlib.sha256(data).hexdigest()==prior['archives']['raw_sha256'];raw.write_bytes(data);dbpath=Path(tmp)/'trace.sqlite'
  with (out/(phase+'-interval-export-v1.log')).open('w') as f:subprocess.run(['nsys','export','--type','sqlite','--output',str(dbpath),str(raw)],stdout=f,stderr=subprocess.STDOUT,check=True)
  with sqlite3.connect(dbpath) as db:
   tables=[r[0] for r in db.execute("select name from sqlite_master where type='table'")]
   for index in range(2,11):
    launch=prior['graph_launches'][index];lo=min(k['first_start'] for k in launch['kernels']);hi=max(k['last_end'] for k in launch['kernels']);rawk=list(db.execute('select k.start,k.end,s.value,k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ,k.registersPerThread from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName where k.start>=? and k.end<=? order by k.start',(lo,hi)));groups={}
    for a,b,*key in rawk:
     key=tuple(key);r=groups.setdefault(key,dict(name=key[0],shape=key[1:],calls=0,milliseconds=0));r['calls']+=1;r['milliseconds']+=(b-a)/1e6
    kernels=sorted(groups.values(),key=lambda x:(x['name'],x['shape']));signature=[{k:v for k,v in r.items() if k!='milliseconds'} for r in kernels];intervals=[(r[0],r[1]) for r in rawk];copies={}
    for table in ['CUPTI_ACTIVITY_KIND_MEMCPY','CUPTI_ACTIVITY_KIND_MEMSET']:
     if table in tables:
      parts=list(db.execute('select start,end from '+table+' where start>=? and end<=?',(lo,hi)));intervals+=parts;copies[table]=dict(calls=len(parts),sum_ms=sum(b-a for a,b in parts)/1e6)
    busy=union(intervals)/1e6;span=(hi-lo)/1e6
    rows.append(dict(index=index,sample=index-2,event_ms=benchmark['full']['samples_ms'][index-2],api_ms=launch['api_ms'],span_ms=span,kernel_calls=len(rawk),kernel_ms=sum(k['milliseconds'] for k in kernels),observed_gpu_union_ms=busy,unattributed_span_ms=span-busy,outside_kernel_span_ms=benchmark['full']['samples_ms'][index-2]-span,kernel_signature_sha256=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest(),kernels=kernels,other_activities=copies))
 report['cases'][phase]=rows;print(phase,[(r['sample'],round(r['event_ms'],4),r['kernel_calls'],round(r['kernel_ms'],4),round(r['unattributed_span_ms'],4),r['kernel_signature_sha256'][:10]) for r in rows],flush=True)
sigs={r['kernel_signature_sha256'] for rows in report['cases'].values() for r in rows};report['all_full_kernel_signatures_equal']=len(sigs)==1;report['new_private_kernels_executed']=any(any(n in k['name'] for n in ['prepare_unproved','check_prepared_unproved','strict_unproved','exact_unproved']) for rows in report['cases'].values() for r in rows for k in r['kernels']);report['complete']=True;dest.write_text(json.dumps(report,indent=2)+'\n');print('same signatures',len(sigs)==1,'new private kernels',report['new_private_kernels_executed'],flush=True)
