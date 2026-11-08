"""Independent validity for accepted loop frames and full-hull state after exhaustion."""
import hashlib,json,subprocess,tempfile
from pathlib import Path
import numpy as np
w=Path('.work/vtm');folder=w/'tangential-production-loop-v1';dest=w/'tangential-production-loop-v1-independent.json';assert not dest.exists()
source=(w/'check-surface-bound-reuse-final-v2.py').read_text().split("out=w/'surface-bound-reuse-final'")[0]
exec(compile(source,str(w/'check-surface-bound-reuse-final-v2.py'),'exec'))
d=json.loads((folder/'result.json').read_text());assert [r[0] for r in d['rows']]==[64,0,1,1,0] and [r[1] for r in d['rows']]==[0,1,0,0,1]
assert all(not any(d['rows'][i][2:10]) and not any(d['values'][i]) for i in [0,2,3])
report=dict(scope=__doc__,capture_sha256=hashlib.sha256((folder/'result.json').read_bytes()).hexdigest(),frames={})
for i in [1,4]:
 path=folder/f'frame{i}.npz';row=validate(path)
 with np.load(path) as m,tempfile.TemporaryDirectory(prefix='tangential-loop-',dir='/opt/warpower-generated-cache') as tmp:
  off=Path(tmp)/'boundary.off'
  with off.open('w') as f:
   f.write(f'OFF\n{len(m["points"])} {len(m["boundary"])} 0\n');np.savetxt(f,m['points'],fmt='%.17g');np.savetxt(f,np.column_stack((np.full(len(m['boundary']),3),m['boundary'])),fmt='%d')
  r=subprocess.run([str(w/'surface-reference'),str(off)],text=True,capture_output=True,check=True)
  assert int(r.stdout.splitlines()[0].split()[-1])==0,r.stdout
  row.update(boundary_self_intersection_pairs=0,boundary_reference_stdout=r.stdout)
  assert [len(m[k]) for k in ['points','full_tetrahedra','tetrahedra','boundary']]==d['rows'][i][2:6]
 report['frames'][str(i)]=row;dest.write_text(json.dumps(report,indent=2)+'\n');print(i,'valid',flush=True)
with np.load(folder/'frame1.npz') as a,np.load(folder/'frame4.npz') as b:report['recovery']=compare(a,b)
assert all(report['recovery'][k] for k in ['points_bitwise_equal','roles_equal','full_tets_equal','material_tets_equal','oriented_boundary_equal'])
with np.load(folder/'frame0.npz') as m,tempfile.TemporaryDirectory(prefix='tangential-exhaustion-',dir='/opt/warpower-generated-cache') as tmp:
 pp,tp=Path(tmp)/'points.txt',Path(tmp)/'tets.txt'
 np.savetxt(pp,m['points'],fmt='%.17g',header=str(len(m['points'])),comments='')
 np.savetxt(tp,m['full_tetrahedra'],fmt='%d',header=str(len(m['full_tetrahedra'])),comments='')
 hull=json.loads(subprocess.run([str(w/'check-delaunay'),str(pp),str(tp)],text=True,capture_output=True,check=True).stdout);assert hull['valid']
 equal=bool(np.array_equal(m['points'].view(np.uint32),m['baseline_points'].view(np.uint32)));assert equal
 def canonical(t):
  s=np.sort(t,axis=1);return s[np.lexsort(s.T[::-1])]
 report['exhaustion']=dict(full_hull=hull,points_restored_bitwise=equal,full_cells_restored=bool(np.array_equal(canonical(m['full_tetrahedra']),canonical(m['baseline_full_tetrahedra']))),public_zero_counts=True,source_sha256=hashlib.sha256((folder/'frame0.npz').read_bytes()).hexdigest())
 assert report['exhaustion']['full_cells_restored']
report['complete']=True;dest.write_text(json.dumps(report,indent=2)+'\n');print(report['recovery'],report['exhaustion'],flush=True)
