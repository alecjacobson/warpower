"""Offline diagnosis of the retained invalid selection; never publish it as a valid mesh."""
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np

w=Path('.work/vtm')
directory=w/'graded-armadillo-8192-diagnostic-v1'
path=directory/'failed-candidate.npz'
dest=directory/'local-selection-analysis-v1.json'
assert not dest.exists()
d=np.load(path)
points=d['points']
tets=d['full_tetrahedra']
selected=d['selected'].astype(bool)
distance=d['signed_distance']
roles=d['roles']
local=np.asarray([[1,2,3],[0,3,2],[0,1,3],[0,2,1]])

def boundary_state(keep):
    faces=Counter(tuple(sorted(map(int,f))) for tet in tets[keep] for f in tet[local])
    assert all(n<=2 for n in faces.values())
    boundary=[f for f,n in faces.items() if n==1]
    edges=Counter(tuple(sorted((f[a],f[b]))) for f in boundary for a,b in [(0,1),(1,2),(2,0)])
    return dict(selected_cells=int(np.count_nonzero(keep)),boundary_faces=len(boundary),bad_edges=[dict(vertices=list(edge),incidence=count) for edge,count in edges.items() if count!=2])

def exact_volume(ids):
    q=[[Fraction(float(x)) for x in points[i]] for i in ids]
    a,b,c=[[row[k]-q[0][k] for k in range(3)] for row in q[1:]]
    det=a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])
    return det/6

state=boundary_state(selected)
stars=[]
prune=np.zeros(len(tets),bool)
for edge in state['bad_edges']:
    a,b=edge['vertices']
    incident=np.flatnonzero(np.any(tets==a,axis=1)&np.any(tets==b,axis=1))
    cells=[]
    for i in incident:
        eligible=bool(selected[i] and distance[i]>0 and np.all(roles[tets[i]]<2))
        prune[i]|=eligible
        volume=exact_volume(tets[i])
        assert volume>0
        cells.append(dict(row=int(i),vertices=tets[i].tolist(),selected=bool(selected[i]),signed_center_distance=float(distance[i]),circumradius=float(d['circumradii'][i]),eligible_exterior_band_removal=eligible,vertex_roles=roles[tets[i]].tolist(),volume_float=float(volume),volume_exact=str(volume)))
    stars.append(dict(edge=edge,points=points[[a,b]].tolist(),cells=cells))
after=selected&~prune
before_vertices=set(map(int,tets[selected].ravel()))
after_vertices=set(map(int,tets[after].ravel()))
removed=np.flatnonzero(prune)
record=dict(scope=__doc__,snapshot_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),initial=state,stars=stars,simulated_first_band_prune=dict(rows=removed.tolist(),exact_removed_volume=str(sum((exact_volume(tets[i]) for i in removed),Fraction(0))),orphaned_vertices=sorted(before_vertices-after_vertices),after=boundary_state(after)),limitations=['This reconstructs the first exterior-center band-prune decision from the saved final state; it is not a replay of the full prior fitting/recovery history.','No runtime change, accepted repaired mesh or explanation of later/earlier recovery states is asserted.'])
dest.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record['simulated_first_band_prune'],indent=2),flush=True)
