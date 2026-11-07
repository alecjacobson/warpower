"""Independent quality and feature measurements of every valid full pilot."""
import json
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm'
source=(w/'check-surface-bound-reuse-final-v2.py').read_text().split("out=w/'surface-bound-reuse-final'")[0]
exec(compile(source,str(w/'check-surface-bound-reuse-final-v2.py'),'exec'))
from benchmarks.feature_quality import measure as feature_measure
out=w/'representable-full-v1';dest=w/'representable-full-independent-v1.json';assert not dest.exists()
result=dict(scope='Independent geometry and physical/feature quality, outside GPU execution; no speed claim.',cases={})
for name in ['torus','dense-hand','plate']:
    result['cases'][name]={}
    for phase in ['baseline','candidate']:
        folder=out/(name+'-'+phase);p=folder/'mesh.npz'
        raw=json.loads((folder/'result.json').read_text());assert raw['valid']==1 and raw['error']==0
        checked=validate(p,folder/'mesh.vtu.gz')
        if name!='torus':
            with np.load(p) as d:
                distance=NativeSurfaceDistance(w/'surface-distance',scratch)
                checked['features']=feature_measure(d['points'],d['boundary'],d['roles'],np.unique(d['tetrahedra']),d['input_points'],d['input_faces'],feature_angle=45,corner_angle=45,samples_per_edge=16,distance_query=distance)
                checked['features']['distance_oracle']=distance.metadata
        result['cases'][name][phase]=checked
        dest.write_text(json.dumps(result,indent=2)+'\n')
        q=checked['quality'];print(name,phase,'valid',checked['full_hull']['valid'],'min',q['radius_ratio_percentiles']['0'],'rms',q['two_sided_area_weighted_rms'],flush=True)
    a,b=[np.load(out/(name+'-'+phase)/'mesh.npz') for phase in ['baseline','candidate']]
    result['cases'][name]['comparison']=compare(a,b)
    dest.write_text(json.dumps(result,indent=2)+'\n')
