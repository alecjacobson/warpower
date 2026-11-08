"""Record edge-incidence witnesses for the rejected initial CPU scan control."""
import hashlib
import json
from pathlib import Path
import numpy as np
from benchmarks.cpu_mesher import read_mesh

w=Path('.work/vtm')
out=w/'cpu-scan-topology-failure-v1.json'
assert not out.exists()
rows=[]
for path in sorted((w/'cpu-scan-feature-pilot-v1/sequential').glob('mesh-*.txt')):
    p,t=read_mesh(path)
    fs=np.concatenate([t[:,[i for i in range(4) if i!=k]] for k in range(4)])
    faces,counts=np.unique(np.sort(fs,axis=1),axis=0,return_counts=True)
    boundary=faces[counts==1]
    es=np.concatenate([boundary[:,[0,1]],boundary[:,[1,2]],boundary[:,[2,0]]])
    edges,inc=np.unique(np.sort(es,axis=1),axis=0,return_counts=True)
    bad=edges[inc!=2]
    determinants=np.linalg.det(p[t[:,1:]]-p[t[:,:1]])
    assert len(bad)>0 and np.all(determinants>0) and counts.max()==2
    witnesses=[]
    for edge,n in zip(bad,inc[inc!=2]):
        incident=boundary[np.sum(np.isin(boundary,edge),axis=1)==2]
        witnesses.append(dict(edge=edge.tolist(),incident_faces=int(n),faces=incident.tolist(),coordinates=p[edge].tolist()))
    rows.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),sites=len(p),tetrahedra=len(t),boundary_faces=len(boundary),minimum_determinant=float(determinants.min()),nonmanifold_boundary_edges=len(bad),witnesses=witnesses))
assert len(rows)==3
report=dict(scope=__doc__,meshes=rows,all_same_mesh=len({r['sha256'] for r in rows})==1,valid_material_manifold=False,cpu_quota_status='Unavailable: the v2 runner retained records only after independent quality completed. Native stdout and meshes survive; quota samples cannot be recovered and no clean performance claim is made.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
out.write_text(json.dumps(report,indent=2)+'\n')
print([(r['sites'],r['tetrahedra'],r['nonmanifold_boundary_edges']) for r in rows])
