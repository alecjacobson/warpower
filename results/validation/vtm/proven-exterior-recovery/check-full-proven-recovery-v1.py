"""Independent checks of the fresh seed7 prototype, including VTU and features."""
import json
from pathlib import Path

w = Path('.work/vtm')
source = (w/'check-surface-bound-reuse-final-v2.py').read_text().split("out=w/'surface-bound-reuse-final'")[0]
exec(compile(source, str(w/'check-surface-bound-reuse-final-v2.py'), 'exec'))
from benchmarks.feature_quality import measure as feature_measure

folder = w/'hand-full-proven-recovery-v1/seed7'
out = w/'hand-full-proven-recovery-v1-independent.json'
assert not out.exists()
raw = json.loads((folder/'result.json').read_text()); quota(raw)
assert raw['valid'] == 1 and raw['error'] == 0
checked = validate(folder/'mesh.npz', folder/'mesh.vtu')
with np.load(folder/'mesh.npz') as d:
    distance = NativeSurfaceDistance(w/'surface-distance', scratch)
    checked['features'] = feature_measure(d['points'], d['boundary'], d['roles'],
        np.unique(d['tetrahedra']), d['input_points'], d['input_faces'], feature_angle=45,
        corner_angle=45, samples_per_edge=16, distance_query=distance)
checked['scope'] = 'Independent complete fresh private seed7 prototype; not yet production recovery or a performance comparison.'
out.write_text(json.dumps(checked, indent=2)+'\n')
print(dict(minimum=checked['quality']['radius_ratio_percentiles']['0'],
           rms=checked['quality']['two_sided_area_weighted_rms'],
           self_intersections=checked['boundary_self_intersection_pairs']), flush=True)
