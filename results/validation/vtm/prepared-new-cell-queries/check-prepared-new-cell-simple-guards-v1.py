"""Independent simple-prototype guard validation and cross-variant equality."""
import hashlib,json,subprocess,tempfile
from pathlib import Path
import numpy as np
w=Path('.work/vtm');out=w/'prepared-new-cell-simple-guards-v1';dest=w/'prepared-new-cell-simple-guards-v1-independent.json';assert not dest.exists()
# Reuse established independent geometric oracles; this is offline evidence only.
s=(w/'check-prepared-new-cell-pilot-v2.py').read_text();prefix=s[:s.index('report=dict(scope=')];prefix=prefix.replace("out=w/'prepared-new-cell-pilot-v2';dest=w/'prepared-new-cell-pilot-v2-independent.json';assert not dest.exists()", "out=w/'prepared-new-cell-pilot-v2';dest=w/'prepared-new-cell-pilot-v2-independent.json'")
exec(compile(prefix,str(w/'check-prepared-new-cell-pilot-v2.py'),'exec'))
out=w/'prepared-new-cell-simple-guards-v1';dest=w/'prepared-new-cell-simple-guards-v1-independent.json';report=dict(scope=__doc__,cases={})
for name in ['torus8192-progressive','torus8192-fixed32']:
 p=out/(name+'.json');d=json.loads(p.read_text());quota(d);assert d['valid']==1 and d['error']==0
 row=dict(independent=independent(p.with_suffix('.npz'),None),comparison={},median_ms=timing(d),result_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 for phase in ['before','after']:
  ref=w/'prepared-new-cell-pilot-v2'/(name+'-'+phase+'.json');r=json.loads(ref.read_text());quota(r)
  with np.load(ref.with_suffix('.npz')) as a,np.load(p.with_suffix('.npz')) as b:c=compare(a,b)
  works={k:work(d)[k]==work(r)[k] for k in work(d)}
  assert all(works.values()) and all(c[k] for k in ['points_bitwise_equal','roles_equal','full_tets_equal','material_tets_equal','oriented_boundary_equal'])
  row['comparison'][phase]=dict(reference=str(ref),reference_sha256=hashlib.sha256(ref.read_bytes()).hexdigest(),geometry=c,work=works,relative_time_change=timing(d)/timing(r)-1)
 report['cases'][name]=row;dest.write_text(json.dumps(report,indent=2)+'\n');print(name,row['comparison'],flush=True)
# Exact full-hull validation of the third, retained-grid candidate.
from benchmarks.mesh_quality import measure
from tests.test_domain import cube
p=out/'retained-grid.json';d=json.loads(p.read_text());assert d['error']==0 and d['reuse_accepted']==[1]
for key in ['initial_cpu_quota_delta','cpu_quota_delta']:assert all(x['throttled_periods']==0 for x in d[key].values())
with np.load(p.with_suffix('.npz')) as f:
 points,tets=f['points'],f['tetrahedra']
 with tempfile.TemporaryDirectory(prefix='grid-proof-',dir='/opt/warpower-generated-cache') as tmp:
  pp,tp=Path(tmp)/'points.txt',Path(tmp)/'tets.txt';np.savetxt(pp,points,fmt='%.17g',header=str(len(points)),comments='');np.savetxt(tp,tets,fmt='%d',header=str(len(tets)),comments='')
  exact=json.loads(subprocess.run([str(w/'check-delaunay'),str(pp),str(tp)],text=True,capture_output=True,check=True).stdout);assert exact['valid']
 side=d['side']-1;quality=measure(points,tets,*cube(side/2,(side/2,)*3));assert abs(quality['volume']-side**3)<1e-8
 def canonical(t):
  s=np.sort(t,axis=1);return s[np.lexsort(s.T[::-1])]
 row=dict(full_hull=exact,quality=quality,comparison={},median_ms=d['median_ms'])
 for phase in ['before','after']:
  ref=w/f'prepared-new-cell-grid-v2-{phase}.json';r=json.loads(ref.read_text())
  with np.load(ref.with_suffix('.npz')) as f:assert np.array_equal(points,f['points']) and np.array_equal(canonical(tets),canonical(f['tetrahedra']))
  equal={k:d[k]==r[k] for k in ['n','count','reuse_attempted','reuse_accepted','reuse_error','reuse_seeds','reuse_rounds','base_attempted','points_sha256']};assert all(equal.values())
  row['comparison'][phase]=dict(reference=str(ref),reference_sha256=hashlib.sha256(ref.read_bytes()).hexdigest(),points_bitwise_equal=True,canonical_cells_equal=True,work=equal,relative_time_change=d['median_ms']/r['median_ms']-1)
report['cases']['retained-grid']=row;report['complete']=True;dest.write_text(json.dumps(report,indent=2)+'\n');print('grid',row['comparison'],flush=True)
