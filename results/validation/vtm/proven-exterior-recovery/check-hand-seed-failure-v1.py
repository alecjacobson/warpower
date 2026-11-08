"""Independently certify the full hull and enumerate invalid material edge stars."""
import argparse
from collections import Counter
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--input', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
assert not a.output.exists()
d = np.load(a.input)
points, full, tets = d['points'], d['full_tetrahedra'], d['tetrahedra']
with tempfile.TemporaryDirectory(prefix='hand-seed-hull-', dir='/opt/warpower-generated-cache') as tmp:
    pp, tt = Path(tmp) / 'points.txt', Path(tmp) / 'tets.txt'
    np.savetxt(pp, points, fmt='%.17g', header=str(len(points)), comments='')
    np.savetxt(tt, full, fmt='%d', header=str(len(full)), comments='')
    hull = json.loads(subprocess.check_output(['.work/vtm/check-delaunay', str(pp), str(tt)], text=True))
assert hull['valid']
ids = np.flatnonzero(d['selected'])
np.testing.assert_array_equal(ids, d['source_indices'])
np.testing.assert_array_equal(full[ids], tets)
faces = Counter(tuple(sorted(f)) for tet in tets for f in
                [tet[[1, 2, 3]], tet[[0, 3, 2]], tet[[0, 1, 3]], tet[[0, 2, 1]]])
assert max(faces.values()) <= 2
boundary = sorted(f for f, count in faces.items() if count == 1)
assert boundary == sorted(map(tuple, np.sort(d['boundary'], axis=1)))
edges = Counter(tuple(sorted((f[i], f[(i+1) % 3]))) for f in boundary for i in range(3))
bad = {edge: count for edge, count in edges.items() if count != 2}
cells = points.astype(float)[full]
volumes = np.linalg.det(cells[:, 1:] - cells[:, :1]) / 6
assert np.all(volumes > 0)
rows = []
for edge, incidence in sorted(bad.items()):
    star = np.flatnonzero(np.isin(full, edge).sum(axis=1) == 2)
    rows.append(dict(vertices=list(map(int, edge)), incidence=incidence,
                     points=points[list(edge)].tolist(), roles=d['roles'][list(edge)].tolist(),
                     boundary_faces=[list(map(int, f)) for f in boundary if set(edge) <= set(f)],
                     star=[dict(source_index=int(i), vertices=full[i].tolist(),
                                selected=int(d['selected'][i]), roles=d['roles'][full[i]].tolist(),
                                signed_distance=float(d['signed_distance'][i]), center=d['centers'][i].tolist(),
                                radius=float(d['radii'][i]), radius_ratio=float(d['input_quality'][i]),
                                volume=float(volumes[i])) for i in star]))
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
    if len(seen) != len(g) or any(len(neighbors) != 2 for neighbors in g.values()):
        bad_links.append(v)
r = dict(scope='Independent full-hull certification and face/edge/vertex-link enumeration of the saved final material selection. Invalid candidates are diagnostic only.',
         input_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(), full_hull=hull,
         material_count=len(tets), boundary_faces=len(boundary), boundary_matches_independent_incidence=True,
         edge_incidence_histogram=dict(sorted(Counter(edges.values()).items())), bad_edge_count=len(bad),
         bad_vertex_links=sorted(bad_links), material_volume=float(volumes[ids].sum()), bad_edges=rows)
a.output.write_text(json.dumps(r, indent=2) + '\n')
print(json.dumps({k: v for k, v in r.items() if k not in ['bad_edges', 'bad_vertex_links']}), flush=True)
for row in rows:
    print('edge', row['vertices'], 'roles', row['roles'],
          'star', [(s['source_index'], s['selected'], s['roles'], s['signed_distance']) for s in row['star']], flush=True)
