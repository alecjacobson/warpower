"""Complete gallery solve with previous-complex diagnostics; timings are instrumented."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from local_flip_observer_v1 import install,COLUMNS
instances=install()
from examples.vtm_gallery import main
try:main()
finally:
 out=Path(sys.argv[sys.argv.index('--output')+1]);out.mkdir(parents=True,exist_ok=True)
 rows=[]
 for index,m in enumerate(instances):
  o=m.observer;n=int(o.calls.numpy()[0]);assert n<=len(o.records) and o.overflow.numpy()[0]==0
  records=o.records.numpy()[:n];snap_points=o.points.numpy();snap_tets=o.tets.numpy();counts=o.counts.numpy();files=[]
  for slot in range(6):
   if counts[slot,0]==0:continue
   assert np.all(counts[slot]<=snap_tets.shape[2]);name=out/f'previous-{index}-snapshot-{slot}.npz'
   assert not name.exists();np.savez_compressed(name,previous_points=snap_points[slot,0],points=snap_points[slot,1],previous_tets=snap_tets[slot,0,:counts[slot,0]],tetrahedra=snap_tets[slot,1,:counts[slot,1]])
   files.append(dict(slot=slot,path=str(name),sha256=hashlib.sha256(name.read_bytes()).hexdigest(),counts=counts[slot].tolist()))
  errors,frequency=np.unique(records[:,2],return_counts=True) if n else ([],[])
  eligible=(records[:,0]==records[:,1])&(records[:,8]==0)
  rows.append(dict(index=index,capacity=m.n,calls=n,error_histogram={str(int(k)):int(v) for k,v in zip(errors,frequency)},already_certified=int(((records[:,2]==0)&eligible).sum()),sphere_only=int(((records[:,2]==8)&eligible).sum()),snapshots=files,records=records.tolist()))
 report=dict(scope='Observation only: an extra exact certificate evaluates previous connectivity on current sites before unchanged production reconstruction. No candidate output or default algorithm changes; timings are instrumented, not performance evidence.',columns=COLUMNS,instances=rows,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('.work/vtm/local_flip_observer_v1.py')]})
 (out/'observer.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps([{k:v for k,v in r.items() if k!='records'} for r in rows]),flush=True)
