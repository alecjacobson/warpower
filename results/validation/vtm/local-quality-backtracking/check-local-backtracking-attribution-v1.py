"""Independently check the first rejected trial's full hull and limiting cells."""
import hashlib,json,subprocess,tempfile
from pathlib import Path
import numpy as np
w=Path('.work/vtm');source=w/'local-backtracking-attribution-v1.npz';d=np.load(source)
with tempfile.TemporaryDirectory(prefix='local-attribution-',dir='/opt/warpower-generated-cache') as tmp:
 pp,tp=Path(tmp)/'points.txt',Path(tmp)/'tets.txt'
 np.savetxt(pp,d['points'],fmt='%.17g',header=str(len(d['points'])),comments='')
 np.savetxt(tp,d['tets'],fmt='%d',header=str(len(d['tets'])),comments='')
 full=json.loads(subprocess.check_output([str(w/'check-delaunay'),str(pp),str(tp)],text=True));assert full['valid']
kept=np.flatnonzero(d['keep']!=0);cells=d['points'].astype(float)[d['tets'][kept]];edge=cells[:,1:]-cells[:,:1]
volume=np.linalg.det(edge)/6;assert np.all(volume>0)
offset=np.linalg.solve(2*edge,np.sum(edge*edge,axis=2)[...,None])[...,0];radius=np.linalg.norm(offset,axis=1);area=np.zeros(len(kept))
for opposite in range(4):
 face=cells[:,[j for j in range(4) if j!=opposite]];area+=np.linalg.norm(np.cross(face[:,1]-face[:,0],face[:,2]-face[:,0]),axis=1)/2
quality=9*volume/(area*radius);np.testing.assert_allclose(quality,d['quality'][kept],rtol=1e-9,atol=1e-12)
bad=kept[quality<float(d['previous_minimum'][0])*(1-1e-10)];assert len(bad)==2
vertices=np.unique(d['tets'][bad]);moved=np.any(d['points']!=d['original'],axis=1)
oldring=np.unique(d['old_tets'][np.isin(d['old_tets'],vertices).any(axis=1)])
record=dict(snapshot_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),full_hull=full,
 independent_retained_minimum=float(quality.min()),limiting_cells=bad.tolist(),limiting_vertices=vertices.tolist(),
 moved_limiting_vertices=vertices[moved[vertices]].tolist(),old_neighborhood_vertices=oldring.tolist(),
 moved_selected_old_neighborhood=oldring[moved[oldring]&(d['mask'][oldring]!=0)].tolist(),
 focus403_direct=bool(403 in vertices),focus403_old_neighborhood=bool(403 in oldring),
 scope='Independent full-hull certification and physical qualities of the retained set in the first rejected midpoint-hand trial. Old-neighborhood membership is a graph diagnostic, not a proof of causal independence.')
p=w/'local-backtracking-attribution-independent-v1.json';assert not p.exists();p.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
