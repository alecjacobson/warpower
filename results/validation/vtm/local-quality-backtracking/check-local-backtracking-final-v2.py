"""Independent production meshes, unchanged-default pairs, and standalone nonregression."""
import hashlib,json
from pathlib import Path
w=Path('.work/vtm');source=(w/'check-surface-bound-reuse-final-v2.py').read_text().split("out=w/'surface-bound-reuse-final'")[0];exec(compile(source,str(w/'check-surface-bound-reuse-final-v2.py'),'exec'))
from benchmarks.feature_quality import measure as feature_measure
out=w/'local-backtracking-final';dest=w/'local-backtracking-final-independent-v2.json';assert not dest.exists()
commands=json.loads((out/'commands.json').read_text());assert len(commands)==41
for r in commands:assert r['exit_code']==(1 if r['name'] in ['coarse-hand','hand-seed7'] and r.get('valid')==0 else 0) and r['quota_clean'],r
report=dict(runtime='6aa4ba63a34910ba7e9148966dd8a9015a261d06',baseline='39f8f54fa5cc410b46cd9b057e617fc92f7e4c28',cases={},default_equivalence={},standalone={})
def save():dest.write_text(json.dumps(report,indent=2)+'\n')
def params(d):
 p={k:v for k,v in d['parameters'].items() if k not in ['output','local_quality_retries']};return p
for r in commands:
 if r['name'].startswith(('diagram','delaunay')):continue
 name,phase=r['name'],r['phase'];folder=out/(name+'-'+phase);raw=json.loads((folder/'result.json').read_text());quota(raw)
 if raw['valid']==0:
  assert name in ['coarse-hand','hand-seed7']
  assert raw['valid']==0 and raw['error']==64
  report['cases'].setdefault(name,{})[phase]=dict(valid=0,error=64,result_sha256=hashlib.sha256((folder/'result.json').read_bytes()).hexdigest(),input_sha256=hashlib.sha256((folder/'input.off').read_bytes()).hexdigest(),counts=raw['counts'],scope='Explicitly invalid material output; failure status and input retained. Gallery does not export a mesh for this result.')
  save();continue
 assert raw['valid']==1 and raw['error']==0
 checked=validate(folder/'mesh.npz',folder/'mesh.vtu.gz')
 if name in ['thin-hand','hand-seed7','plate','plate-seed7']:
  with np.load(folder/'mesh.npz') as d:
   distance=NativeSurfaceDistance(w/'surface-distance',scratch)
   checked['features']=feature_measure(d['points'],d['boundary'],d['roles'],np.unique(d['tetrahedra']),d['input_points'],d['input_faces'],feature_angle=45,corner_angle=45,samples_per_edge=16,distance_query=distance)
   checked['features']['distance_oracle']=distance.metadata
 checked.update(median_ms=raw['median_additional_replay_ms'],first_ms=raw['first_replay_gpu_ms'],all_timed_quotas_clean=True,counts=raw['counts'],steps=raw['steps'],moves=raw['moves'],rounds=raw['rounds'])
 report['cases'].setdefault(name,{})[phase]=checked
 if phase!='before':
  before=out/(name+'-before');previous=json.loads((before/'result.json').read_text());assert params(raw)==params(previous)
  assert previous['input_off_sha256']==raw['input_off_sha256']
  if not previous['valid']:
   save();continue
  with np.load(before/'mesh.npz') as a,np.load(folder/'mesh.npz') as b:
   assert all(np.array_equal(a[k],b[k]) for k in ['input_points','input_faces'])
   if phase=='disabled':
    comparison=compare(a,b);assert all(comparison[k] for k in ['roles_equal','full_tets_equal','material_tets_equal','oriented_boundary_equal']),comparison
    assert comparison['maximum_coordinate_difference']<=1e-8,comparison
    work={k:previous[k]==raw[k] for k in ['counts','steps','moves','rounds','quality_rejections','refinement_rejections']};assert all(work.values()),work
    report['default_equivalence'][name]=dict(comparison=comparison,work_equal=work,before_ms=previous['median_additional_replay_ms'],after_ms=raw['median_additional_replay_ms'])
 save();print(name,phase,checked['median_ms'],checked['quality']['radius_ratio_percentiles']['0'],checked['quality']['two_sided_area_weighted_rms'],flush=True)
for name in ['delaunay1000','delaunay100000','delaunay-grid','delaunay-fitted-hand','diagram100000']:
 a,b=[json.loads((out/(name+'-'+phase+'.json')).read_text()) for phase in ['before','disabled']]
 if 'full' in a:
  for d in [a,b]:assert all(v['throttled_periods']==0 for v in d['full']['cpu'].values()) and d['error']==0
  skip={'first_replay_ms','full','base','auxiliary','certificate'};same={k:a[k]==b[k] for k in a if k not in skip};assert all(same.values()),same
  am,bm=a['full']['median_ms'],b['full']['median_ms']
 else:
  keys=['points_sha256','weights_sha256','distribution','n','seed','weighted','nnz','status_counts','diagnostics'];same={k:a[k]==b[k] for k in keys};assert all(same.values()),same
  assert not a['cpu_throttled'] and not b['cpu_throttled'];am,bm=1000*a['median_seconds'],1000*b['median_seconds']
 report['standalone'][name]=dict(before_ms=am,after_ms=bm,relative_change=bm/am-1,unchanged=same,all_timed_quotas_clean=True);save()
print(json.dumps(report['standalone'],indent=2),flush=True)
