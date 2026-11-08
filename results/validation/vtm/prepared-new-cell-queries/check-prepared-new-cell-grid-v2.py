"""Independent full-hull and physical quality checks for retained-grid stress outputs."""
import hashlib,json,subprocess,tempfile
from pathlib import Path
import numpy as np
from benchmarks.mesh_quality import measure
from tests.test_domain import cube
w=Path('.work/vtm');r=dict(scope=__doc__,cases={});data=[]
for phase in ['before','after']:
 path=w/f'prepared-new-cell-grid-v2-{phase}.json';d=json.loads(path.read_text());assert d['error']==0 and d['reuse_accepted']==[1]
 for key in ['initial_cpu_quota_delta','cpu_quota_delta']:assert all(x['throttled_periods']==0 for x in d[key].values())
 with np.load(path.with_suffix('.npz')) as f:
  p,t=f['points'],f['tetrahedra'];data.append((p,t,d))
  with tempfile.TemporaryDirectory(prefix='grid-proof-',dir='/opt/warpower-generated-cache') as tmp:
   pp,tp=Path(tmp)/'points.txt',Path(tmp)/'tets.txt';np.savetxt(pp,p,fmt='%.17g',header=str(len(p)),comments='');np.savetxt(tp,t,fmt='%d',header=str(len(t)),comments='')
   exact=json.loads(subprocess.run([str(w/'check-delaunay'),str(pp),str(tp)],text=True,capture_output=True,check=True).stdout);assert exact['valid']
  side=d['side']-1;inputs=cube(side/2,(side/2,)*3);quality=measure(p,t,*inputs)
  assert abs(quality['volume']-side**3)<1e-8
  r['cases'][phase]=dict(result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),mesh_sha256=hashlib.sha256(path.with_suffix('.npz').read_bytes()).hexdigest(),full_hull=exact,quality=quality,median_ms=d['median_ms'])
  print(phase,exact,flush=True)
def canonical(t):
 s=np.sort(t,axis=1);return s[np.lexsort(s.T[::-1])]
a,b=data;r['points_bitwise_equal']=bool(np.array_equal(a[0],b[0]));r['canonical_cells_equal']=bool(np.array_equal(canonical(a[1]),canonical(b[1])))
r['work_equal']={k:a[2][k]==b[2][k] for k in ['n','count','reuse_attempted','reuse_accepted','reuse_error','reuse_seeds','reuse_rounds','base_attempted','points_sha256']}
assert r['points_bitwise_equal'] and r['canonical_cells_equal'] and all(r['work_equal'].values())
r['relative_time_change']=b[2]['median_ms']/a[2]['median_ms']-1;r['complete']=True
(w/'prepared-new-cell-grid-v2-independent.json').write_text(json.dumps(r,indent=2)+'\n');print(r['relative_time_change'],flush=True)
