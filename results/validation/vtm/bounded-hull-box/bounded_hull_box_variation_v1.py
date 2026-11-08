"""Audit retained exterior alternatives without claiming identical full connectivity."""
import hashlib
from fractions import Fraction as F
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cells(t):return {tuple(sorted(map(int,row))) for row in t}
def oriented(t):return sorted(min(tuple(row),tuple(np.roll(row,1)),tuple(np.roll(row,2))) for row in t)
def six_volume(p,t):
 a,b,c,d=[[F(float(x)) for x in p[i]] for i in t]
 u,v,w=[[x-y for x,y in zip(row,a)] for row in [b,c,d]]
 return u[0]*(v[1]*w[2]-v[2]*w[1])-u[1]*(v[0]*w[2]-v[2]*w[0])+u[2]*(v[0]*w[1]-v[1]*w[0])

def audit_pair(directory,name,checked,work_equal):
 paths={phase:Path(directory)/(name+'-'+phase)/'mesh.npz' for phase in ['before','after']}
 a,b=[np.load(p) for p in paths.values()]
 for key in ['input_points','input_faces','roles']:np.testing.assert_array_equal(a[key],b[key])
 for phase,path in paths.items():
  r=checked[phase]['independent'];assert r['mesh_sha256']==sha(path) and r['full_hull']['valid']
  assert r['all_boundary_vertex_links_single_cycles'] and r['vtu_round_trip_exact'] and r['boundary_self_intersection_pairs']==0
 assert all(work_equal.values())
 ma,mb=cells(a['tetrahedra']),cells(b['tetrahedra']);assert ma==mb and oriented(a['boundary'])==oriented(b['boundary'])
 qa,qb=[{tuple(sorted(map(int,t))):float(q) for t,q in zip(d['tetrahedra'],d['radius_ratio'])} for d in [a,b]];keys=sorted(qa)
 np.testing.assert_allclose([qa[t] for t in keys],[qb[t] for t in keys],rtol=1e-11,atol=1e-14)
 for metric in ['volume','two_sided_area_weighted_rms','mean_radius_ratio']:
  np.testing.assert_allclose(checked['before']['independent']['quality'][metric],checked['after']['independent']['quality'][metric],rtol=1e-11,atol=1e-14)
 fa,fb=cells(a['full_tetrahedra']),cells(b['full_tetrahedra']);removed,added=sorted(fa-fb),sorted(fb-fa)
 assert not set(removed)&ma and not set(added)&mb
 changes=np.argwhere(a['points'].view(np.uint32)!=b['points'].view(np.uint32))
 if removed or added:assert len(changes)>0,'Identical points with changed full topology require separate investigation.'
 exact=[]
 for phase,data,ts in [('before',a,removed),('after',b,added)]:
  for t in ts:
   volume=abs(six_volume(data['points'],t));assert volume>0
   exact.append(dict(phase=phase,vertices=t,absolute_six_volume=str(volume),absolute_volume=float(volume/6)))
 r=dict(mesh_sha256={phase:sha(p) for phase,p in paths.items()},changed_coordinates=[dict(site=int(i),axis=int(j),role=int(a['roles'][i]),before=float(a['points'][i,j]),after=float(b['points'][i,j])) for i,j in changes],full_hull_changes=dict(removed=removed,added=added),exact_changed_exterior_volumes=exact,material_connectivity_equal=True,oriented_material_boundary_equal=True,aligned_radius_ratios_allclose=True,maximum_material_quality_difference=max(abs(qa[t]-qb[t]) for t in keys),quality_and_fit_allclose=True,recorded_work_equal=True,full_connectivity_equal=not removed and not added)
 a.close();b.close();return r
