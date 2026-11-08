"""Offline feasibility study of the rejected trial's pinched material vertex star."""
from collections import defaultdict
from pathlib import Path
import hashlib,json
import numpy as np
folder=Path('.work/vtm/graded-armadillo-8192-transaction-v1')
with np.load(folder/'failed-candidate.npz') as d,np.load(folder/'last-rejected-band.npz') as trial:
    full=d['full_tetrahedra'];sel=trial['selected'][:len(full)]!=0;distance=d['signed_distance'];roles=d['roles']
    pinches=json.loads((folder/'vertex-link-analysis-v1.json').read_text())['disconnected_vertex_links'];stars=[]
    for pinch in pinches:
        vertex=pinch['vertex'];rows=np.flatnonzero(sel&np.any(full==vertex,axis=1));faces=defaultdict(list);adj={int(i):set() for i in rows}
        for row in rows:
            ids=full[row];others=[int(x) for x in ids if x!=vertex]
            for i in range(3):
                key=tuple(sorted([vertex,others[i],others[(i+1)%3]]));faces[key].append(int(row))
        for shared in faces.values():
            assert len(shared)<=2
            if len(shared)==2:a,b=shared;adj[a].add(b);adj[b].add(a)
        unseen=set(adj);components=[]
        while unseen:
            todo=[min(unseen)];component=set()
            while todo:
                x=todo.pop()
                if x in component:continue
                component.add(x);todo.extend(adj[x]-component)
            unseen-=component;ids=sorted(component)
            components.append(dict(rows=ids,tetrahedra=full[ids].tolist(),signed_center_distance=distance[ids].tolist(),all_centers_exterior=bool(np.all(distance[ids]>0)),all_sites_without_prescribed_feature_roles=bool(np.all(roles[full[ids]]<2))))
        stars.append(dict(vertex=vertex,selected_incident_cells=len(rows),components=components))
    record=dict(scope=__doc__,stars=stars,snapshot_sha256=hashlib.sha256((folder/'failed-candidate.npz').read_bytes()).hexdigest(),trial_sha256=hashlib.sha256((folder/'last-rejected-band.npz').read_bytes()).hexdigest(),limitation='Offline classification only. Removing a star component has not been implemented, accepted or validated as a complete repaired mesh.')
p=folder/'pinched-star-analysis-v1.json';assert not p.exists();p.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2),flush=True)
