"""Validate the complete paired-query observer output outside timing."""
import json
from pathlib import Path
w=Path('.work/vtm');source=(w/'check-surface-bound-reuse-final-v2.py').read_text().split("out=w/'surface-bound-reuse-final'")[0];exec(compile(source,str(w/'check-surface-bound-reuse-final-v2.py'),'exec'))
from benchmarks.feature_quality import measure as feature_measure
folder=w/'surface-query-cache-paired-audit-hand1-v1';dest=w/'surface-query-cache-paired-audit-independent-v1.json';assert not dest.exists()
audit=json.loads((folder/'paired-query-audit.json').read_text());total=np.sum([r['statistics'] for r in audit['instances']],axis=0).tolist();assert all(total[i]==0 for i in [2,5,6,7]) and total[1]>0 and total[4]>0
r=dict(scope=__doc__,query_statistics=dict(zip(audit['columns'],total)),query_audit_sha256=hashlib.sha256((folder/'paired-query-audit.json').read_bytes()).hexdigest(),independent=validate(folder/'mesh.npz',folder/'mesh.vtu.gz'))
with np.load(folder/'mesh.npz') as data:
 r['comparisons']={}
 for phase in ['before','after']:
  with np.load(w/f'surface-query-cache-final/hand-seed1-{phase}/mesh.npz') as ref:r['comparisons'][phase]=compare(ref,data)
 r['features']=feature_measure(data['points'],data['boundary'],data['roles'],np.unique(data['tetrahedra']),data['input_points'],data['input_faces'],feature_angle=45,corner_angle=45,samples_per_edge=16,distance_query=NativeSurfaceDistance(w/'surface-distance',scratch))
assert any(c['material_tets_equal'] and c['oriented_boundary_equal'] and c['roles_equal'] for c in r['comparisons'].values())
raw=json.loads((folder/'result.json').read_text());base=json.loads((w/'surface-query-cache-final/hand-seed1-before/result.json').read_text());r['work_equal']={k:raw[k]==base[k] for k in ['counts','topology','rounds','steps','moves','quality_rejections','refinement_rejections']};assert all(r['work_equal'].values())
r['complete']=True;dest.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r['query_statistics'],indent=2))
