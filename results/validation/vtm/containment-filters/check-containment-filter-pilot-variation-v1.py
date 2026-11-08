"""Audit pilot geometry differences without claiming bitwise reproducibility."""
import hashlib,json
from fractions import Fraction
from pathlib import Path
import numpy as np
w=Path('.work/vtm');source=w/'containment-filter-pilot-v3-independent.json';report=json.loads(source.read_text());out=w/'containment-filter-pilot-variation-v1.json';assert not out.exists()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cells(array):return {tuple(sorted(map(int,t))) for t in array}
def exact_six_volume(p,t):
 a,b,c,d=[[Fraction(float(x)) for x in p[i]] for i in t]
 b,c,d=[[x-y for x,y in zip(row,a)] for row in [b,c,d]]
 return b[0]*(c[1]*d[2]-c[2]*d[1])-b[1]*(c[0]*d[2]-c[2]*d[0])+b[2]*(c[0]*d[1]-c[1]*d[0])
rows={}
for name,checked in report['cases'].items():
 paths={phase:w/'containment-filter-pilot-v2'/(name+'-'+phase)/'mesh.npz' for phase in ['before','after']}
 a,b=[np.load(p) for p in paths.values()]
 for key in ['input_points','input_faces','roles']:np.testing.assert_array_equal(a[key],b[key])
 for phase,path in paths.items():
  c=checked['independent'][phase];assert c['mesh_sha256']==sha(path) and c['full_hull']['valid'] and c['all_boundary_vertex_links_single_cycles'] and c['boundary_self_intersection_pairs']==0
 assert all(checked['work_equal'].values())
 changes=np.argwhere(a['points'].view(np.uint32)!=b['points'].view(np.uint32));full_a,full_b=cells(a['full_tetrahedra']),cells(b['full_tetrahedra'])
 material_a,material_b=cells(a['tetrahedra']),cells(b['tetrahedra']);assert material_a==material_b and cells(a['boundary'])==cells(b['boundary'])
 qa,qb=[{tuple(sorted(map(int,t))):float(q) for t,q in zip(d['tetrahedra'],d['radius_ratio'])} for d in [a,b]]
 keys=sorted(qa);np.testing.assert_allclose([qa[t] for t in keys],[qb[t] for t in keys],rtol=1e-11,atol=1e-14)
 changed_full={'removed':sorted(full_a-full_b),'added':sorted(full_b-full_a)}
 assert not (set(changed_full['removed']) & material_a) and not (set(changed_full['added']) & material_b)
 exact=[]
 for phase,d,ts in [('before',a,changed_full['removed']),('after',b,changed_full['added'])]:
  for t in ts:
   v=exact_six_volume(d['points'],t);assert v!=0
   exact.append(dict(phase=phase,vertices=t,absolute_six_volume=str(abs(v))))
 qualities={phase:checked['independent'][phase]['quality'] for phase in ['before','after']}
 for metric in ['volume','two_sided_area_weighted_rms','mean_radius_ratio']:
  np.testing.assert_allclose(qualities['before'][metric],qualities['after'][metric],rtol=1e-11,atol=1e-14)
 rows[name]=dict(mesh_sha256={phase:sha(p) for phase,p in paths.items()},changed_coordinates=[dict(site=int(i),axis=int(j),role=int(a['roles'][i]),before=float(a['points'][i,j]),after=float(b['points'][i,j])) for i,j in changes],full_hull_changes=changed_full,exact_changed_exterior_volumes=exact,material_cells_equal=True,boundary_cells_equal=True,maximum_material_quality_difference=max(abs(qa[t]-qb[t]) for t in keys),work_equal=True)
 assert (len(changed_full['removed']),len(changed_full['added']))==((4,4) if name=='hand-seed7' else (0,0))
 for d in [a,b]:d.close()
trajectory=json.loads((w/'containment-trajectory-v1/commands.json').read_text());assert len(trajectory)==4
for r in trajectory:
 p=Path(r['decision-audit.json']['path']);assert sha(p)==r['decision-audit.json']['sha256']
 d=json.loads(p.read_text());assert r['exit_code']==0 and d['statistics'][0]>0 and d['statistics'][1]==0 and d['statistics'][2]==d['statistics'][3]
result=dict(scope='All pilot material cells, boundaries, per-cell qualities and work agree. Four exterior full-hull cells differ in one pair after tiny near-zero coordinate changes, and both complete hulls are independently valid. No claim of bitwise reproducibility or of a traced causal kernel. The duplicate-predicate trajectory audits separately show exact and filtered containment decisions agree on every tested eligible cell.',independent_report_sha256=sha(source),cases=rows,trajectory_audit_sha256=sha(w/'containment-trajectory-v1/commands.json'),all_material_meshes_equivalent=True)
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))
