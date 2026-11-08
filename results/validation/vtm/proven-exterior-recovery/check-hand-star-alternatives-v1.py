"""Offline selection alternatives, never accepted as a production repair."""
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

w = Path('.work/vtm')
source = w / 'hand-seed7-final-v1.npz'
d = np.load(source)
full, points, roles = d['full_tetrahedra'], d['points'], d['roles']
original = d['selected'] != 0
cells = points.astype(float)[full]
volume = np.linalg.det(cells[:, 1:] - cells[:, :1]) / 6
before_volume = volume[original].sum()
before_vertices = set(map(int, np.unique(full[original])))


def feature_edges(tets):
    return set(tuple(sorted((int(t[i]), int(t[j])))) for t in tets
               for i in range(4) for j in range(i+1, 4) if roles[t[i]] >= 2 and roles[t[j]] >= 2)


protected = feature_edges(full[original])
rows = []
changes = [710, 1254, 1529, 4983]
for bits in itertools.product([0, 1], repeat=4):
    selected = original.copy(); selected[changes] = bits
    tets = full[selected]
    faces = Counter(tuple(sorted(f)) for t in tets for f in
                    [t[[1, 2, 3]], t[[0, 3, 2]], t[[0, 1, 3]], t[[0, 2, 1]]])
    boundary = [f for f, n in faces.items() if n == 1]
    edges = Counter(tuple(sorted((f[i], f[(i+1) % 3]))) for f in boundary for i in range(3))
    bad_edges = [e for e, n in edges.items() if n != 2]
    links = {}
    for f in boundary:
        for i in range(3):
            v = int(f[i]); x, y = map(int, np.delete(f, i)); g = links.setdefault(v, {})
            g.setdefault(x, set()).add(y); g.setdefault(y, set()).add(x)
    bad_links = []
    for v, g in links.items():
        seen = set(); todo = [next(iter(g))]
        while todo:
            at = todo.pop()
            if at not in seen:
                seen.add(at); todo.extend(g[at] - seen)
        if len(seen) != len(g) or any(len(x) != 2 for x in g.values()):
            bad_links.append(v)
    orphaned = sorted(before_vertices - set(map(int, np.unique(tets))))
    lost = sorted(protected - feature_edges(tets))
    v = volume[selected].sum()
    row = dict(selected_for_changes=list(bits), added=np.flatnonzero(selected & ~original).tolist(),
               removed=np.flatnonzero(original & ~selected).tolist(), bad_edges=[list(map(int, e)) for e in bad_edges],
               bad_vertex_links=bad_links, orphaned_sites=orphaned,
               lost_edges_between_feature_sites=lost, volume_change=float(v / before_volume - 1),
               boundary_euler=len(links) - len(edges) + len(boundary))
    rows.append(row)
    if not bad_edges and not bad_links:
        print(row, flush=True)
record = dict(scope='Offline exhaustive16 subsets of four exterior-center cells around the one bad edge. Topological alternatives only; additions lack a whole-cell material proof and no candidate is adopted.',
              input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), changes=changes, alternatives=rows)
out = w / 'hand-seed7-star-alternatives-v1.json'; assert not out.exists()
out.write_text(json.dumps(record, indent=2) + '\n')
