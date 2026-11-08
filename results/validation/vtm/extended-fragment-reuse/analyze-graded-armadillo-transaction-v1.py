"""Locate disconnected boundary vertex links in the actual rejected band trial."""
from collections import Counter, defaultdict
import hashlib,json
from pathlib import Path
import numpy as np
w=Path('.work/vtm');folder=w/'graded-armadillo-8192-transaction-v1'
with np.load(folder/'failed-candidate.npz') as d, np.load(folder/'last-rejected-band.npz') as trial:
    full=d['full_tetrahedra'];sel=trial['selected'][:len(full)]!=0
    local=np.asarray([[1,2,3],[0,3,2],[0,1,3],[0,2,1]])
    faces=Counter(tuple(sorted(map(int,f))) for tet in full[sel] for f in tet[local])
    assert all(n<=2 for n in faces.values())
    boundary=[f for f,n in faces.items() if n==1]
    edges=Counter(tuple(sorted((f[i],f[j]))) for f in boundary for i,j in [(0,1),(1,2),(2,0)])
    assert all(n==2 for n in edges.values())
    links=defaultdict(list)
    for f in boundary:
        for v in f:links[v].append(tuple(x for x in f if x!=v))
    pinches=[]
    for v,es in links.items():
        adj=defaultdict(set)
        for a,b in es:adj[a].add(b);adj[b].add(a)
        assert all(len(x)==2 for x in adj.values())
        unseen=set(adj);components=[]
        while unseen:
            todo=[min(unseen)];component=set()
            while todo:
                p=todo.pop()
                if p in component:continue
                component.add(p);todo.extend(adj[p]-component)
            unseen-=component;components.append(sorted(component))
        if len(components)>1:pinches.append(dict(vertex=v,link_cycles=components,position=d['points'][v].tolist(),role=int(d['roles'][v])))
    record=dict(scope=__doc__,snapshot_sha256=hashlib.sha256((folder/'failed-candidate.npz').read_bytes()).hexdigest(),trial_sha256=hashlib.sha256((folder/'last-rejected-band.npz').read_bytes()).hexdigest(),selected_cells=int(sel.sum()),boundary_triangles=len(boundary),all_edges_incidence_two=True,disconnected_vertex_links=pinches,volume_fraction_removed=float(1-trial['volume'][0]/trial['before_volume'][0]),diagnostics=int(trial['diagnostics'][0]),conclusion='The actual six-cell prune leaves disconnected vertex links (diagnostic bit4). The volume guard passes. Rollback is necessary; relaxing edge/link guards would publish an invalid boundary.')
    assert pinches and record['diagnostics']==4 and record['volume_fraction_removed']<.01
p=folder/'vertex-link-analysis-v1.json';assert not p.exists();p.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2),flush=True)
