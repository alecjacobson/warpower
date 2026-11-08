"""Independent production geometry, unchanged healthy trajectories and API pairs."""
import hashlib,json
from pathlib import Path
w=Path('.work/vtm');source=(w/'check-surface-bound-reuse-final-v2.py').read_text().split("out=w/'surface-bound-reuse-final'")[0];exec(compile(source,str(w/'check-surface-bound-reuse-final-v2.py'),'exec'))
from benchmarks.feature_quality import measure as feature_measure
out=w/'proven-exterior-final';dest=w/'proven-exterior-final-independent-v1.json';assert not dest.exists()
commands=json.loads((out/'commands.json').read_text());assert len(commands)==37
report=dict(scope='Final production pinned pairs. Invalid results excluded from mesh speed comparisons.',cases={},healthy_equivalence={},standalone={})
def save():dest.write_text(json.dumps(report,indent=2)+'\n')
def params(d):return {k:v for k,v in d['parameters'].items() if k not in ['output','proven_exterior_recovery']}
for r in commands:
 assert r['quota_clean']
 path=Path(r['result']);assert hashlib.sha256(path.read_bytes()).hexdigest()==r['result_sha256']
 if r['name'].startswith(('diagram','delaunay')):continue
 name,phase=r['name'],r['phase'];folder=path.parent;raw=json.loads(path.read_text());quota(raw)
 checked=dict(valid=raw['valid'],error=raw['error'],counts=raw['counts'],rounds=raw['rounds'],steps=raw['steps'],moves=raw['moves'],median_ms=raw['median_additional_replay_ms'],first_ms=raw['first_replay_gpu_ms'],all_timed_quotas_clean=True,result_sha256=r['result_sha256'],source=r['source'])
 if not raw['valid']:
  assert raw['error']==64 and not any(raw['counts']) and r['exit_code']==1
  checked['scope']='Explicitly invalid material output. Failure time retained, excluded from valid-mesh speed comparisons.'
 else:
  assert raw['error']==0 and r['exit_code']==0
  checked['independent']=validate(folder/'mesh.npz',folder/'mesh.vtu.gz')
  if name.startswith('hand-seed') or name=='plate':
   with np.load(folder/'mesh.npz') as d:
    distance=NativeSurfaceDistance(w/'surface-distance',scratch)
    checked['features']=feature_measure(d['points'],d['boundary'],d['roles'],np.unique(d['tetrahedra']),d['input_points'],d['input_faces'],feature_angle=45,corner_angle=45,samples_per_edge=16,distance_query=distance)
 if phase!='before':
  previous=json.loads((out/(name+'-before')/'result.json').read_text());assert params(raw)==params(previous)
  assert previous['input_off_sha256']==raw['input_off_sha256']
  if 'proven_exterior_recovery' in raw:checked['recovery']=raw['proven_exterior_recovery']
  if previous['valid'] and name!='hand-seed7-local4':
   assert raw['valid'] and checked['recovery']['enabled_after_fitting']==0
   with np.load(out/(name+'-before')/'mesh.npz') as a,np.load(folder/'mesh.npz') as b:comparison=compare(a,b)
   assert all(comparison[k] for k in ['roles_equal','full_tets_equal','material_tets_equal','oriented_boundary_equal']),comparison
   assert comparison['maximum_coordinate_difference']<=1e-8,comparison
   work={k:previous[k]==raw[k] for k in ['counts','topology','rounds','steps','moves','quality_rejections','refinement_rejections']};assert all(work.values()),work
   report['healthy_equivalence'][name]=dict(comparison=comparison,work_equal=work,before_ms=previous['median_additional_replay_ms'],after_ms=raw['median_additional_replay_ms'],relative_time_change=raw['median_additional_replay_ms']/previous['median_additional_replay_ms']-1)
  if phase=='disabled':
   assert raw['valid']==previous['valid']==0 and raw['error']==previous['error']==64
   assert all(raw[k]==previous[k] for k in ['counts','rounds','steps','moves','quality_rejections','refinement_rejections'])
 report['cases'].setdefault(name,{})[phase]=checked;save();print(name,phase,checked['valid'],checked['median_ms'],flush=True)
for name in ['delaunay1000','delaunay100000','delaunay-grid','delaunay-fitted-hand','diagram100000']:
 a,b=[json.loads((out/(name+'-'+phase+'.json')).read_text()) for phase in ['before','after']]
 if 'full' in a:
  for d in [a,b]:assert all(v['throttled_periods']==0 for v in d['full']['cpu'].values()) and d['error']==0
  skip={'first_replay_ms','full','base','auxiliary','certificate'};same={k:a[k]==b[k] for k in a if k not in skip};assert all(same.values()),same
  am,bm=a['full']['median_ms'],b['full']['median_ms']
 else:
  keys=['points_sha256','weights_sha256','distribution','n','seed','weighted','nnz','status_counts','diagnostics'];same={k:a[k]==b[k] for k in keys};assert all(same.values()),same
  assert not a['cpu_throttled'] and not b['cpu_throttled'];am,bm=1000*a['median_seconds'],1000*b['median_seconds']
 report['standalone'][name]=dict(before_ms=am,after_ms=bm,relative_change=bm/am-1,unchanged=same,all_timed_quotas_clean=True);save()
print(json.dumps(report['standalone'],indent=2),flush=True)
assert len(report['healthy_equivalence'])==9
