"""Exact offline proof: surface-plane separation and rational-centroid winding."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def vec(p):
    # Every finite binary32 coordinate is an integer multiple of 2^-149.
    out = []
    for x in p:
        n, d = float(x).as_integer_ratio()
        assert (1 << 149) % d == 0
        out.append(n * ((1 << 149) // d))
    return tuple(out)


def sub(a, b):
    return tuple(x-y for x, y in zip(a, b))


def det(a, b, c):
    return a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])


def sign(x):
    return (x > 0) - (x < 0)


def winding(triangles, numerator, direction):
    count = 0
    for triangle in triangles:
        a, b, c = [tuple(4*q[k]-numerator[k] for k in range(3)) for q in triangle]
        n = det(a, b, c)
        u, v, w = det(direction, b, c), det(direction, c, a), det(direction, a, b)
        denominator = u+v+w
        if denominator == 0:
            if n == 0:
                return None
            continue
        if n == 0 or sign(n) != sign(denominator):
            continue
        if all(t == 0 or sign(t) == sign(denominator) for t in [u, v, w]):
            if any(t == 0 for t in [u, v, w]):
                return None
            count += sign(denominator)
    return count


p = argparse.ArgumentParser()
p.add_argument('--input', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
assert not a.output.exists()
d = np.load(a.input)
points, full = d['points'], d['full_tetrahedra']
integer_points = [vec(p) for p in points]
surface = d['input_points'][d['input_faces']]
triangles = [[vec(p) for p in t] for t in surface]
lo, hi = surface.min(axis=1), surface.max(axis=1)
directions = [(11, 23, 37), (-17, 29, 43), (31, -47, 19)]
assert winding(triangles, (0, 0, 0), directions[0]) == 1
rows = []
for i in np.flatnonzero((d['selected'] != 0) & (d['signed_distance'] > 0)):
    tet = points[full[i]]; ip = [integer_points[int(v)] for v in full[i]]
    overlapping = np.flatnonzero(np.all(hi >= tet.min(axis=0), axis=1) & np.all(lo <= tet.max(axis=0), axis=1))
    mixed = []
    for f in overlapping:
        x, y, z = triangles[f]
        sides = [sign(det(sub(y, x), sub(z, x), sub(p, x))) for p in ip]
        if 1 in sides and -1 in sides:
            mixed.append(int(f))
    numerator = tuple(sum(p[k] for p in ip) for k in range(3))
    counts = []
    if not mixed:
        for direction in directions:
            counts.append(winding(triangles, numerator, direction))
            if counts[-1] is not None:
                break
    exterior = bool(not mixed and counts and counts[-1] == 0)
    rows.append(dict(source_index=int(i), vertices=full[i].tolist(),
                     mixed_plane_faces=mixed, centroid_winding_attempts=counts,
                     proved_open_interior_exterior=exterior, roles=d['roles'][full[i]].tolist()))
proved = [r['source_index'] for r in rows if r['proved_open_interior_exterior']]
record = dict(scope='Offline arbitrary-precision integer proof on original float32 sites. For each tested cell, overlapping surface triangle planes cannot cut its open interior, and its exact rational centroid has unambiguous winding0. Failed proof is inconclusive. No runtime change or surface-distance bound.',
              input_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              tested_selected_exterior_centers=len(rows), proved_exterior_source_indices=proved, cells=rows)
a.output.write_text(json.dumps(record, indent=2) + '\n')
print('tested', len(rows), 'proved exterior', proved, flush=True)
