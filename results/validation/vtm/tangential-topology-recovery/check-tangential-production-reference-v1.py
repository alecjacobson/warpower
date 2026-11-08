"""Independent original-reference, role, displacement and quality guards in accepted loop frames."""
import hashlib,json
from pathlib import Path
import numpy as np
w=Path('.work/vtm');dest=w/'tangential-production-reference-v1.json';assert not dest.exists()
basepath=w/'uniform-armadillo-8192-transaction-v1/failed-candidate.npz'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(t):
 a=np.sort(t,axis=1);return a[np.lexsort(a.T[::-1])]
def measures(p,t):
 cells=p.astype(float)[t];edges=cells[:,1:]-cells[:,:1];v=np.linalg.det(edges)/6;assert np.all(v>0)
 offsets=np.linalg.solve(2*edges,np.sum(edges*edges,axis=2)[...,None])[...,0];radius=np.linalg.norm(offsets,axis=1);areas=np.zeros(len(t))
 for k in range(4):
  f=cells[:,[j for j in range(4) if j!=k]];areas+=np.linalg.norm(np.cross(f[:,1]-f[:,0],f[:,2]-f[:,0]),axis=1)/2
 q=9*v/(areas*radius)
 return dict(minimum_radius_ratio=float(q.min()),quality_deficit=float(np.square(np.maximum(0,.3-q)).sum()),volume=float(v.sum()))
with np.load(basepath) as b:
 original=np.unique(b['tetrahedra']);old=measures(b['points'],b['tetrahedra']);rows=[]
 for folder in ['tangential-production-loop-v2','tangential-production-loop-native-v1-frames']:
  with np.load(w/folder/'frame0.npz') as z:
   assert np.array_equal(b['points'].view(np.uint32),z['points'].view(np.uint32))
   assert np.array_equal(canonical(b['full_tetrahedra']),canonical(z['full_tetrahedra']))
   assert np.array_equal(b['roles'],z['roles'])
  for frame in [1,4]:
   path=w/folder/f'frame{frame}.npz'
   with np.load(path) as c:
    missing=np.setdiff1d(original,np.unique(c['tetrahedra']));assert len(missing)==0
    assert np.array_equal(b['roles'],c['roles'])
    moved=np.flatnonzero(np.any(b['points'].view(np.uint32)!=c['points'].view(np.uint32),axis=1));assert len(moved)==1
    site=int(moved[0]);assert b['roles'][site]==1
    incident=b['full_tetrahedra'][np.any(b['full_tetrahedra']==site,axis=1)];neighbors=np.setdiff1d(np.unique(incident),[site])
    edge=float(np.linalg.norm(b['points'][neighbors].astype(float)-b['points'][site].astype(float),axis=1).min())
    length=float(np.linalg.norm(c['points'][site].astype(float)-b['points'][site].astype(float)));assert 0<length<=.2*edge
    current=measures(c['points'],c['tetrahedra'])
    assert current['minimum_radius_ratio']>=old['minimum_radius_ratio']*(1-1e-10)
    assert current['quality_deficit']<old['quality_deficit']*(1-1e-10)
    assert abs(current['volume']-old['volume'])<=old['volume']*(.01+1e-10)
    rows.append(dict(path=str(path),sha256=sha(path),original_referenced_vertices=len(original),retained_original_references=True,all_roles_equal=True,moved_site=site,moved_role=int(b['roles'][site]),shortest_original_edge=edge,actual_binary32_displacement=length,relative_displacement=length/edge,before=old,after=current))
r=dict(scope=__doc__,baseline_sha256=sha(basepath),source_sha256=sha(Path(__file__)),frames=rows,complete=True)
dest.write_text(json.dumps(r,indent=2)+'\n');print(rows[0]);print(len(rows),'accepted frames pass independent reference/role/radius/minimum/deficit/volume guards')
