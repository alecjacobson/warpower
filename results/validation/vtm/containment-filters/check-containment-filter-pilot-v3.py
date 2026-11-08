"""Independent geometry and work comparisons for complete containment-filter pilots."""
import hashlib,json
from pathlib import Path
w=Path('.work/vtm');source=(w/'check-surface-bound-reuse-final-v2.py').read_text().split("out=w/'surface-bound-reuse-final'")[0];exec(compile(source,str(w/'check-surface-bound-reuse-final-v2.py'),'exec'))
from benchmarks.feature_quality import measure as feature_measure
out=w/'containment-filter-pilot-v2';dest=w/'containment-filter-pilot-v3-independent.json';assert not dest.exists()
rows=json.loads((out/'commands.json').read_text());assert len(rows)==10 and all(r['exit_code']==0 and r['quota_clean'] for r in rows)
report=dict(scope='Independent validation of all pilot outputs; canonical differences retained for a separate audit before adoption.',cases={})
for name in ['hand-seed7','hand-seed7-local4','hand-seed2','coarse-cavity','torus4096']:
 records={}; checked={}
 for phase in ['before','after']:
  folder=out/(name+'-'+phase);path=folder/'result.json';d=json.loads(path.read_text());row=next(r for r in rows if r['name']==name and r['phase']==phase)
  assert hashlib.sha256(path.read_bytes()).hexdigest()==row['result_sha256'];quota(d);assert d['valid']==1 and d['error']==0
  records[phase]=d;checked[phase]=validate(folder/'mesh.npz',folder/'mesh.vtu.gz')
  if name.startswith('hand'):
   with np.load(folder/'mesh.npz') as m:
    checked[phase]['features']=feature_measure(m['points'],m['boundary'],m['roles'],np.unique(m['tetrahedra']),m['input_points'],m['input_faces'],feature_angle=45,corner_angle=45,samples_per_edge=16,distance_query=NativeSurfaceDistance(w/'surface-distance',scratch))
 a,b=records['before'],records['after'];assert {k:v for k,v in a['parameters'].items() if k!='output'}=={k:v for k,v in b['parameters'].items() if k!='output'};assert a['input_off_sha256']==b['input_off_sha256']
 with np.load(out/(name+'-before')/'mesh.npz') as x,np.load(out/(name+'-after')/'mesh.npz') as y:comparison=compare(x,y)
 work={k:a[k]==b[k] for k in ['counts','topology','rounds','steps','moves','quality_rejections','refinement_rejections']}
 row=dict(before_ms=a['median_additional_replay_ms'],after_ms=b['median_additional_replay_ms'],speedup=a['median_additional_replay_ms']/b['median_additional_replay_ms'],comparison=comparison,work_equal=work,independent=checked)
 report['cases'][name]=row;dest.write_text(json.dumps(report,indent=2)+'\n');print(name,row['before_ms'],row['after_ms'],comparison,work,flush=True)
 assert all(work.values()),row
 assert comparison['roles_equal'] and comparison['maximum_coordinate_difference']<=1e-8,row
 row['canonical_unchanged']=all(comparison[k] for k in ['full_tets_equal','material_tets_equal','oriented_boundary_equal']);dest.write_text(json.dumps(report,indent=2)+'\n')
