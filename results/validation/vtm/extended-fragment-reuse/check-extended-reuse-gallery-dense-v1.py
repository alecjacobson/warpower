"""Independent healthy graded output plus explicit unchanged failed-output contract."""
from pathlib import Path
import hashlib
import json

w=Path('.work/vtm')
source=(w/'check-surface-bound-reuse-final-v2.py').read_text().split("out=w/'surface-bound-reuse-final'")[0]
exec(compile(source,str(w/'check-surface-bound-reuse-final-v2.py'),'exec'))
out=w/'extended-reuse-gallery-dense-v1'
dest=w/'extended-reuse-gallery-dense-v1-independent.json'
assert not dest.exists()
commands=json.loads((out/'commands.json').read_text())
assert len(commands)==2
report=dict(scope=__doc__,cases={})
keys=['valid','error','status','counts','rounds','steps','moves','last_errors','last_bad_extraction','topology','quality_rejections','refinement_rejections']
for name in ['armadillo8192-graded-q2']:
    rows={}
    raw={}
    for phase in ['before','after']:
        folder=out/(name+'-'+phase)
        path=folder/'result.json'
        d=json.loads(path.read_text())
        quota(d)
        raw[phase]=d
        rows[phase]=dict(median_ms=d['median_additional_replay_ms'],result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),state={k:d[k] for k in keys})
        if d['valid']==1:
            assert d['error']==0
            rows[phase]['independent']=validate(folder/'mesh.npz',folder/'mesh.vtu.gz')
        else:
            assert d['error']!=0 and d['counts']==[0,0,0,0] and d['topology']==[0,0,0,0]
            rows[phase]['failed_output_contract']=True
    rows['work_and_state_equal']={k:raw['before'][k]==raw['after'][k] for k in keys}
    if all(raw[p]['valid']==1 for p in ['before','after']):
        with np.load(out/(name+'-before')/'mesh.npz') as a,np.load(out/(name+'-after')/'mesh.npz') as b:
            rows['comparison']=compare(a,b)
            rows['input_arrays_equal']={k:bool(np.array_equal(a[k],b[k])) for k in ['input_points','input_faces']}
        rows['speedup']=rows['before']['median_ms']/rows['after']['median_ms']
    else:
        rows['successful_mesh_speedup_claim']=False
    report['cases'][name]=rows
    dest.write_text(json.dumps(report,indent=2)+'\n')
    print(name,rows.get('comparison'),rows['work_and_state_equal'],rows.get('speedup'),flush=True)
for name,rows in report['cases'].items():
    assert all(rows['work_and_state_equal'].values()),name
    if 'comparison' in rows:
        assert all(rows['input_arrays_equal'].values()),name
        assert all(rows['comparison'][k] for k in ['roles_equal','full_tets_equal','material_tets_equal','oriented_boundary_equal']),name
        assert rows['comparison']['maximum_coordinate_difference']<=1e-8,name
report['complete']=True
dest.write_text(json.dumps(report,indent=2)+'\n')
