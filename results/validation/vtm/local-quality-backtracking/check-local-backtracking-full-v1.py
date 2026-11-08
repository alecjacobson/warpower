"""Independent complete meshes for all six hand policies, including weaker ones."""
import json,gzip,hashlib
from pathlib import Path
w=Path('.work/vtm')
source=(w/'check-surface-bound-reuse-final-v2.py').read_text().split("out=w/'surface-bound-reuse-final'")[0]
exec(compile(source,str(w/'check-surface-bound-reuse-final-v2.py'),'exec'))
from benchmarks.feature_quality import measure as feature_measure
folders=[w/'local-backtracking-full-v1/dense-hand-combined']
folders+=sorted(p for dirname in ['local-backtracking-pilots-v2','local-backtracking-pilots-v3'] for p in (w/dirname).iterdir() if p.is_dir())
report={};dest=w/'local-backtracking-full-independent-v1.json';assert not dest.exists()
for folder in folders:
 r=json.loads((folder/'result.json').read_text());assert r['valid']==1 and r['error']==0
 p=folder/'mesh.npz';vtu=folder/'mesh.vtu.gz';vtu=vtu if vtu.exists() else folder/'mesh.vtu'
 checked=validate(p,vtu)
 with np.load(p) as d:
  distance=NativeSurfaceDistance(w/'surface-distance',scratch)
  checked['features']=feature_measure(d['points'],d['boundary'],d['roles'],np.unique(d['tetrahedra']),d['input_points'],d['input_faces'],feature_angle=45,corner_angle=45,samples_per_edge=16,distance_query=distance)
  checked['features']['distance_oracle']=distance.metadata
 report[str(folder)]=checked;dest.write_text(json.dumps(report,indent=2)+'\n')
 print(str(folder),'min',checked['quality']['radius_ratio_percentiles']['0'],'RMS',checked['quality']['two_sided_area_weighted_rms'],flush=True)
