"""Bind the variation audit to its independent snapshots and check each cell quality."""
import hashlib,json
from pathlib import Path
import numpy as np
w=Path('.work/vtm');source=w/'proven-exterior-baseline-variation-v3.json';out=w/'proven-exterior-physical-cells-v1.json';assert not out.exists()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
audit=json.loads(source.read_text());assert audit['audited_existing_baseline_variation']
reports=[json.loads(Path(r['path']).read_text()) for r in audit['independent_reports']]
for row in audit['independent_reports']:assert sha(Path(row['path']))==row['sha256']
checks=[reports[0]['cases']['hand-seed1']['before'],reports[1]['cases']['hand-seed1']['before'],reports[1]['cases']['hand-seed1']['after']]
rows={}
for (name,row),checked in zip(audit['variants'].items(),checks):
 path=Path(row['path'])/'mesh.npz';assert sha(path)==row['mesh_sha256']==checked['independent']['mesh_sha256']
 with np.load(path) as d:
  c=d['points'].astype(float)[d['tetrahedra']];edges=c[:,1:]-c[:,:1];det=np.linalg.det(edges);assert (det>0).all()
  centers=np.linalg.solve(2*edges,np.sum(edges*edges,axis=2)[...,None])[...,0]
  area=sum(np.linalg.norm(np.cross(c[:,b]-c[:,a],c[:,k]-c[:,a]),axis=1) for a,b,k in [(0,1,2),(0,1,3),(0,2,3),(1,2,3)])
  q=3*det/(area*np.linalg.norm(centers,axis=1));np.testing.assert_allclose(q,d['radius_ratio'],rtol=1e-11,atol=1e-14)
  rows[name]=dict(mesh_sha256=sha(path),material_cells=len(q),maximum_quality_difference=float(np.max(np.abs(q-d['radius_ratio']))),all_volumes_positive=True)
result=dict(scope=__doc__,source_sha256=sha(Path(__file__)),variation_audit_sha256=sha(source),all_snapshot_hashes_match_independent_reports=True,cases=rows)
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
