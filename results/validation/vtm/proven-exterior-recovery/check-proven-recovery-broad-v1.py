"""Independent complete seed/shape outputs and unchanged successful trajectories."""
import hashlib
import json
from pathlib import Path

w=Path('.work/vtm')
source=(w/'check-surface-bound-reuse-final-v2.py').read_text().split("out=w/'surface-bound-reuse-final'")[0]
exec(compile(source,str(w/'check-surface-bound-reuse-final-v2.py'),'exec'))
from benchmarks.feature_quality import measure as feature_measure

out=w/'proven-recovery-broad-v1';dest=w/'proven-recovery-broad-v1-independent.json';assert not dest.exists()
commands=json.loads((out/'commands.json').read_text());assert len(commands)==24
report=dict(scope='Private complete seed/shape grid. Invalid baseline/final candidates are retained and excluded from valid-mesh speed results. Healthy pairs must preserve topology/roles/work; no production adoption claim.',cases={},unchanged={})
def save():dest.write_text(json.dumps(report,indent=2)+'\n')
folders={}
for row in commands:
    name,phase=row['name'],row['phase'];path=Path(row['result']);folder=path.parent;folders[name,phase]=folder
    assert hashlib.sha256(path.read_bytes()).hexdigest()==row['result_sha256']
    raw=json.loads(path.read_text())
    for k in ['first_replay_cpu_quota_delta','additional_replays_cpu_quota_delta']:
        assert all(v['throttled_periods']==0 for v in raw[k].values())
    checked=dict(valid=raw['valid'],error=raw['error'],counts=raw['counts'],rounds=raw['rounds'],steps=raw['steps'],moves=raw['moves'],
                 median_ms=raw['median_additional_replay_ms'],first_ms=raw['first_replay_gpu_ms'],result_sha256=row['result_sha256'])
    if not raw['valid']:
        assert raw['error']==64 and not any(raw['counts']) and row['exit_code']==1
        checked['scope']='Explicitly invalid material result; no mesh performance or geometry claim.'
    else:
        assert raw['error']==0 and row['exit_code']==0
        vtu=folder/'mesh.vtu.gz';vtu=vtu if vtu.exists() else folder/'mesh.vtu'
        checked['independent']=validate(folder/'mesh.npz',vtu)
        if name.startswith('hand-seed') or name=='plate':
            with np.load(folder/'mesh.npz') as d:
                distance=NativeSurfaceDistance(w/'surface-distance',scratch)
                checked['features']=feature_measure(d['points'],d['boundary'],d['roles'],np.unique(d['tetrahedra']),d['input_points'],d['input_faces'],feature_angle=45,corner_angle=45,samples_per_edge=16,distance_query=distance)
    if phase=='candidate':
        pilot=json.loads((folder/'pilot.json').read_text())['instances'][0];checked['pilot']=pilot
        before=folders[name,'before'];previous=json.loads((before/'result.json').read_text())
        params=lambda r:{k:v for k,v in r['parameters'].items() if k!='output'}
        assert params(raw)==params(previous)
        assert previous['input_off_sha256']==raw['input_off_sha256']
        if previous['valid']:
            assert raw['valid'] and pilot['proof_enabled']==0
            with np.load(before/'mesh.npz') as a,np.load(folder/'mesh.npz') as b:
                comparison=compare(a,b)
            assert all(comparison[k] for k in ['roles_equal','full_tets_equal','material_tets_equal','oriented_boundary_equal']),comparison
            assert comparison['maximum_coordinate_difference']<=1e-8,comparison
            work={k:previous[k]==raw[k] for k in ['counts','topology','rounds','steps','moves','quality_rejections','refinement_rejections']};assert all(work.values()),work
            report['unchanged'][name]=dict(comparison=comparison,work_equal=work,relative_time_change=raw['median_additional_replay_ms']/previous['median_additional_replay_ms']-1)
    report['cases'].setdefault(name,{})[phase]=checked;save()
    print(name,phase,checked['valid'],checked['median_ms'],flush=True)
print('valid_outputs',sum(r['valid'] for c in report['cases'].values() for r in c.values()),'healthy_pairs',len(report['unchanged']),flush=True)
