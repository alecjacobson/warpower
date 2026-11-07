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
 with tempfile.TemporaryDirectory(prefix='prepared-check-',dir=w) as tmp:
  pp,tp=Path(tmp)/'points.txt',Path(tmp)/'tets.txt'
  np.savetxt(pp,d['points'],fmt='%.17g',header=str(len(d['points'])),comments='')
  np.savetxt(tp,d['full_tetrahedra'],fmt='%d',header=str(len(d['full_tetrahedra'])),comments='')
  full=json.loads(subprocess.run([str(w/'check-delaunay'),str(pp),str(tp)],text=True,capture_output=True,check=True).stdout)
 assert full['valid']
 distance=NativeSurfaceDistance(w/'surface-distance',w) if len(d['input_faces'])>10000 else None
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

def files(phase,case):
 stem=w/(f'seed-pairs-control-{case}' if phase=='before' else f'prepared-final-{case}')
 return (stem.with_suffix('.json'),stem.with_suffix('.npz'),None) if case.startswith('torus') else (stem/'result.json',stem/'mesh.npz',stem/'mesh.vtu')
cases=['torus1000','torus4096','graded-torus','thin-hand','cavity','disconnected','counterbored-plate','armadillo-uniform','armadillo-graded']
for phase in ['before','after']:
 for case in cases:
  rp,_,_=files(phase,case);quota(json.loads(rp.read_text()))
summary=dict(cases={},standalone={},source='68e2386',baseline='a84e274')
for case in cases:
 a,pa,_=files('before',case);b,pb,vb=files('after',case);a,b=json.loads(a.read_text()),json.loads(b.read_text())
 assert a['error']==b['error']==0 and a['valid']==b['valid']==1
 r=compare(np.load(pa),np.load(pb))
 assert all(r[k] for k in ['roles_equal','full_tets_equal','material_tets_equal','oriented_boundary_equal']),(case,r)
 keys=['rounds','optimizer_steps','quality_moves'] if case.startswith('torus') else ['rounds','steps','moves']
 assert all(a[k]==b[k] for k in keys)
 param=lambda d:{k:v for k,v in d['parameters'].items() if k not in ['output','save_mesh','dot']}
 assert param(a)==param(b)
 ms=lambda d:d.get('median_gpu_milliseconds',d.get('median_additional_replay_ms'))
 row=dict(before_ms=ms(a),after_ms=ms(b),speedup=ms(a)/ms(b),same_parameters=True,same_iteration_work=True,comparison=r,independent=validate(pb,vb))
 if 'persistent_workspace_pool_bytes' in a:
  row['before_pool_bytes']=a['persistent_workspace_pool_bytes'];row['after_pool_bytes']=b['persistent_workspace_pool_bytes']
 if case.startswith('armadillo-'):
  row['comparison_with_previous_published']=compare(np.load(root/f'results/examples/vtm/{case}/mesh.npz'),np.load(pb))
  assert all(v for k,v in row['comparison_with_previous_published'].items() if k!='maximum_coordinate_difference')
 summary['cases'][case]=row
 (w/'prepared-independent.json').write_text(json.dumps(summary,indent=2)+'\n')
 print('checked',case,flush=True)
for case in ['delaunay1000','delaunay100000','delaunay-grid','delaunay-fitted-hand','diagram100000']:
 a,b=[json.loads((w/f'prepared-{phase}-{case}.json').read_text()) for phase in ['before','final']]
 if 'full' in a:
  for d in [a,b]:assert all(x['throttled_periods']==0 for x in d['full']['cpu'].values())
  am,bm=a['full']['median_ms'],b['full']['median_ms']
 else:
  assert a['points_sha256']==b['points_sha256'] and a['nnz']==b['nnz'] and a['status_counts']==b['status_counts']
  for d in [a,b]:assert not d['cpu_throttled']
  am,bm=1000*a['median_seconds'],1000*b['median_seconds']
 summary['standalone'][case]=dict(before_ms=am,after_ms=bm,relative_change=bm/am-1,n=a['n'])
(w/'prepared-independent.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary['standalone'],indent=2),flush=True)
a=np.load(root/'results/examples/vtm/torus/mesh.npz');b=np.load(w/'prepared-final-torus1000.npz');original=compare(a,b)
assert all(v for k,v in original.items() if k!='maximum_coordinate_difference'),original
(w/'prepared-original-torus-comparison.json').write_text(json.dumps(original,indent=2)+'\n')
