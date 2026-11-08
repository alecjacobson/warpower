"""Independent geometry, complete physical quality and plate features for broad pairs."""
import json
from pathlib import Path
w=Path('.work/vtm');source=(w/'check-surface-bound-reuse-final-v2.py').read_text().split("out=w/'surface-bound-reuse-final'")[0];exec(compile(source,str(w/'check-surface-bound-reuse-final-v2.py'),'exec'))
from benchmarks.feature_quality import measure as feature_measure
out=w/'local-backtracking-broad-v1';dest=w/'local-backtracking-broad-independent-v1.json';assert not dest.exists();report={}
for case in ['torus1000','torus4096','plate','cavity','disconnected']:
 report[case]={}
 for mode in ['baseline','candidate']:
  folder=out/(case+'-'+mode);r=json.loads((folder/'result.json').read_text());assert r['valid']==1 and r['error']==0;quota(r)
  checked=validate(folder/'mesh.npz',folder/'mesh.vtu.gz')
  if case=='plate':
   with np.load(folder/'mesh.npz') as d:
    distance=NativeSurfaceDistance(w/'surface-distance',scratch)
    checked['features']=feature_measure(d['points'],d['boundary'],d['roles'],np.unique(d['tetrahedra']),d['input_points'],d['input_faces'],feature_angle=45,corner_angle=45,samples_per_edge=16,distance_query=distance)
    checked['features']['distance_oracle']=distance.metadata
  report[case][mode]=checked;dest.write_text(json.dumps(report,indent=2)+'\n')
  print(case,mode,'min',checked['quality']['radius_ratio_percentiles']['0'],'RMS',checked['quality']['two_sided_area_weighted_rms'],flush=True)
