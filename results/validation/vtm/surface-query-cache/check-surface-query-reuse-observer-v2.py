"""Independent observer meshes; compare available same-runtime final controls."""
import json
from pathlib import Path
w=Path('.work/vtm')
source=(w/'check-surface-bound-reuse-final-v2.py').read_text().split("out=w/'surface-bound-reuse-final'")[0]
exec(compile(source,str(w/'check-surface-bound-reuse-final-v2.py'),'exec'))
from benchmarks.feature_quality import measure as feature_measure
out=w/'surface-query-reuse-observer-v2';dest=w/'surface-query-reuse-observer-independent-v2.json';assert not dest.exists()
report=dict(scope=__doc__,cases={})
mapping={'torus4096':'torus4096','hand7':'hand-seed7','plate':'plate','graded-armadillo':'graded-armadillo'}
prior=json.loads((w/'bounded-hull-box-final-independent-v2.json').read_text())
for case in ['torus1000','torus4096','hand7','plate','graded-armadillo']:
    folder=out/case;raw=json.loads((folder/'result.json').read_text())
    assert raw['valid']==1 and raw['error']==0
    r=dict(independent=validate(folder/'mesh.npz',folder/'mesh.vtu.gz'))
    if case in ['hand7','plate']:
        with np.load(folder/'mesh.npz') as d:
            r['features']=feature_measure(d['points'],d['boundary'],d['roles'],np.unique(d['tetrahedra']),d['input_points'],d['input_faces'],feature_angle=45,corner_angle=45,samples_per_edge=16,distance_query=NativeSurfaceDistance(w/'surface-distance',scratch))
    if case in mapping:
        name=mapping[case];base=w/'bounded-hull-box-final'/(name+'-after')
        a=json.loads((base/'result.json').read_text())
        assert raw['input_off_sha256']==a['input_off_sha256']
        params=lambda x:{k:v for k,v in x['parameters'].items() if k not in ['output','repeats']}
        assert params(raw)==params(a)
        with np.load(base/'mesh.npz') as x,np.load(folder/'mesh.npz') as y:
            r['comparison']=compare(x,y)
            for key in ['input_points','input_faces']:np.testing.assert_array_equal(x[key],y[key])
        r['baseline_mesh_sha256']=prior['cases'][name]['after']['independent']['mesh_sha256']
        assert r['baseline_mesh_sha256']==hashlib.sha256((base/'mesh.npz').read_bytes()).hexdigest()
        r['work_equal']={k:raw[k]==a[k] for k in ['counts','topology','rounds','steps','moves','quality_rejections','refinement_rejections']}
    report['cases'][case]=r;dest.write_text(json.dumps(report,indent=2)+'\n');print(case,r.get('comparison'),r.get('work_equal'),flush=True)
    if 'comparison' in r:
        assert r['comparison']['maximum_coordinate_difference']<=1e-8
        assert all(r['comparison'][k] for k in ['roles_equal','material_tets_equal','oriented_boundary_equal'])
        assert all(r['work_equal'].values())
report['complete']=True;dest.write_text(json.dumps(report,indent=2)+'\n')
