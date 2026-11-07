"""Separate structural profiles explain the graded scan's measured reduction."""
import gzip,hashlib,json,os,signal,sqlite3,subprocess,tempfile,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';derived=Path('/opt/warpower-generated-cache/current-selection-profile');derived.mkdir(parents=True,exist_ok=True)
rows={};env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.',PYTHONDONTWRITEBYTECODE='1')
assert env['WARP_CACHE_PATH']=='/opt/warpower-generated-cache'
for phase,cwd in [('before',w/'extraction-geometry'),('after',root)]:
 stem=w/('current-selection-armadillo-'+phase);raw=derived/(stem.name+'.nsys-rep');dot=derived/(stem.name+'.dot');assert not stem.with_suffix('.json').exists()
 cmd=['nsys','profile','--trace=cuda','--cuda-graph-trace=node','--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=true','-o',str(raw.with_suffix('')),str(root/'.venv/bin/python'),str(w/'current-selection-armadillo-native.py'),'--input',str(w/'scanned-armadillo-input/armadillo.off'),'--output',str(stem.with_suffix('.json')),'--dot',str(dot)]
 start=time.time()
 with stem.with_suffix('.log').open('w') as log:
  p=subprocess.Popen(cmd,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=p.wait(timeout=300)
  except subprocess.TimeoutExpired:
   os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=10);code=124
 row=dict(command=cmd,cwd=str(cwd),exit_code=code,wall_seconds=time.time()-start,source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=cwd,text=True).strip());rows[phase]=row
 (w/'current-selection-armadillo-profile-summary.json').write_text(json.dumps(rows,indent=2)+'\n');assert code==0,row
 db=derived/(stem.name+'.sqlite')
 with (w/(stem.name+'-export.log')).open('w') as log:subprocess.run(['nsys','export','--type','sqlite','--force-overwrite=true','-o',str(db),str(raw)],stdout=log,stderr=subprocess.STDOUT,check=True)
 with sqlite3.connect(db) as c:
  sql='select s.value,count(*),sum(k.end-k.start)/1e6 from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName group by s.value order by sum(k.end-k.start) desc'
  row['kernels']=[dict(kernel=n,calls=k,milliseconds=t) for n,k,t in c.execute(sql)]
 row['result']=json.loads(stem.with_suffix('.json').read_text());row['archives']=[]
 row['derived_sqlite']=dict(path=str(db),sha256=hashlib.sha256(db.read_bytes()).hexdigest(),storage='Regenerable nsys export; exact command in this script.')
 for src in [raw,dot]:
  data=src.read_bytes();dest=w/(src.name+'.gz');assert not dest.exists();dest.write_bytes(gzip.compress(data,mtime=0));assert gzip.decompress(dest.read_bytes())==data
  row['archives'].append(dict(path=str(dest.relative_to(root)),raw_sha256=hashlib.sha256(data).hexdigest(),sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),raw_bytes=len(data),bytes=dest.stat().st_size))
 (w/'current-selection-armadillo-profile-summary.json').write_text(json.dumps(rows,indent=2)+'\n');print(phase,row['result'],row['kernels'][:8],flush=True)
assert rows['before']['result']==rows['after']['result']
