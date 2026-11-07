"""Offline independent validation after all resident timing jobs have finished."""
import hashlib,json,subprocess,tempfile,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from benchmarks.mesh_quality import measure
from benchmarks.surface_distance import NativeSurfaceDistance

root=Path('/home/horde/projects/power-diagram');w=root/'.work/vtm'
def canonical(t):
 t=np.sort(t,axis=1);return t[np.lexsort(t.T[::-1])]
def oriented(t):return sorted(min(tuple(f),tuple(np.roll(f,1)),tuple(np.roll(f,2))) for f in t)
def compare(a,b):
 r=dict(points_bitwise_equal=bool(np.array_equal(a['points'].view(np.uint32),b['points'].view(np.uint32))),maximum_coordinate_difference=float(np.max(np.abs(a['points'].astype(float)-b['points']))),roles_equal=bool(np.array_equal(a['roles'],b['roles'])),full_tets_equal=bool(np.array_equal(canonical(a['full_tetrahedra']),canonical(b['full_tetrahedra']))),material_tets_equal=bool(np.array_equal(canonical(a['tetrahedra']),canonical(b['tetrahedra']))),oriented_boundary_equal=oriented(a['boundary'])==oriented(b['boundary']))
 return r
def quota(d):
 for k in ['cpu_quota_delta','initial_replays_cpu_quota_delta','additional_replays_cpu_quota_delta','first_replay_cpu_quota_delta']:
  if k in d:assert all(v['throttled_periods']==0 for v in d[k].values()),(k,d[k])
def validate(path,vtu=None):
 d=np.load(path)
 with tempfile.TemporaryDirectory(prefix='large-reuse-pair-check-',dir=w) as tmp:
  pp,tp=Path(tmp)/'points.txt',Path(tmp)/'tets.txt'
  np.savetxt(pp,d['points'],fmt='%.17g',header=str(len(d['points'])),comments='')
  np.savetxt(tp,d['full_tetrahedra'],fmt='%d',header=str(len(d['full_tetrahedra'])),comments='')
  full=json.loads(subprocess.run([str(w/'check-delaunay'),str(pp),str(tp)],text=True,capture_output=True,check=True).stdout)
 assert full['valid']
 distance=NativeSurfaceDistance(w/'surface-distance',w)
 q=measure(d['points'],d['tetrahedra'],d['input_points'],d['input_faces'],distance_query=distance) if distance else measure(d['points'],d['tetrahedra'],d['input_points'],d['input_faces'])
 if distance:q['distance_oracle']=distance.metadata
 if 'radius_ratio' in d:np.testing.assert_allclose(list(q['radius_ratio_percentiles'].values()),np.percentile(d['radius_ratio'],[0,1,5,50,95,100]),rtol=1e-10,atol=1e-12)
 links={}
 for f in d['boundary']:
  for i in range(3):
   v=int(f[i]);a,b=map(int,np.delete(f,i));g=links.setdefault(v,{});g.setdefault(a,set()).add(b);g.setdefault(b,set()).add(a)
 for v,g in links.items():
  seen=set();todo=[next(iter(g))]
  while todo:
   at=todo.pop()
   if at not in seen:seen.add(at);todo.extend(g[at]-seen)
  assert len(seen)==len(g) and all(len(a)==2 for a in g.values()),v
 r=dict(mesh_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),full_hull=full,all_boundary_vertex_links_single_cycles=True,quality=q)
 if vtu:
  xml=ET.parse(vtu).getroot();p=np.fromstring(xml.find('.//Points/DataArray').text,sep=' ',dtype=np.float32).reshape(-1,3);t=np.fromstring(xml.find(".//Cells/DataArray[@Name='connectivity']").text,sep=' ',dtype=np.int32).reshape(-1,4);qq=np.fromstring(xml.find(".//CellData/DataArray[@Name='radius_ratio']").text,sep=' ')
  np.testing.assert_array_equal(p.view(np.uint32),d['points'].view(np.uint32));np.testing.assert_array_equal(t,d['tetrahedra']);np.testing.assert_array_equal(qq,d['radius_ratio']);r['vtu_round_trip_exact']=True
  text=subprocess.run([str(w/'surface-reference'),str(path.parent/'boundary.off')],text=True,capture_output=True,check=True).stdout
  assert int(text.split()[2])==0,text
  (path.parent/'boundary-self-intersections.txt').write_text(text);r['boundary_self_intersection_pairs']=0
 return r

out=w/'large-reuse-proof-pairs';summary={}
commands=json.loads((out/'commands.json').read_text());assert len(commands)==6 and all(r['exit_code']==0 for r in commands)
for name in ['graded-torus4096','armadillo-uniform','armadillo-graded']:
 before,after=[out/(name+'-'+phase) for phase in ['control','proofs']]
 a,b=[json.loads((p/'result.json').read_text()) for p in [before,after]]
 for d in [a,b]:quota(d);assert d['valid']==1 and d['error']==0
 comparison=compare(np.load(before/'mesh.npz'),np.load(after/'mesh.npz'))
 assert all(comparison[k] for k in ['roles_equal','full_tets_equal','material_tets_equal','oriented_boundary_equal']),(name,comparison)
 work={k:a[k]==b[k] for k in ['rounds','steps','moves','refinement_rejections','quality_rejections','counts']}
 assert all(work.values()),(name,work)
 parameters=lambda d:{k:v for k,v in d['parameters'].items() if k not in ['output']}
 assert parameters(a)==parameters(b)
 independent=validate(after/'mesh.npz',after/'mesh.vtu')
 summary[name]=dict(before_ms=a['median_additional_replay_ms'],after_ms=b['median_additional_replay_ms'],comparison=comparison,work_equal=work,parameters_equal=True,independent=independent)
 (w/'large-reuse-proof-pairs-independent.json').write_text(json.dumps(summary,indent=2)+'\n')
 print('checked',name,summary[name]['before_ms'],summary[name]['after_ms'],comparison,flush=True)
