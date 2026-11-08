"""Exact rational validation of every exported interval and triangle event."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np
import warp as wp
from proven_exterior_filter_v2 import centroid_row, direction_row, determinant, triangle
from warpower import _filtered_predicates as fp

wp.set_module_options({'enable_backward': False, 'fast_math': False, 'fuse_fp': False})


@wp.kernel
def inspect(rows: wp.array2d[wp.vec3], bounds: wp.array2d[wp.float64],
            flags: wp.array[int], result: wp.array[wp.vec2i]):
    i = wp.tid()
    p0, p1, p2, p3 = rows[i, 0], rows[i, 1], rows[i, 2], rows[i, 3]
    q0, q1, q2, direction = rows[i, 4], rows[i, 5], rows[i, 6], rows[i, 7]
    a, b, c = centroid_row(q0, p0, p1, p2, p3), centroid_row(q1, p0, p1, p2, p3), centroid_row(q2, p0, p1, p2, p3)
    d = direction_row(direction)
    n, u, v, w = determinant(a, b, c), determinant(d, b, c), determinant(d, c, a), determinant(d, a, b)
    denominator = fp.plus(fp.plus(u, v), w)
    bounds[i, 0] = n.value
    bounds[i, 1] = n.error
    bounds[i, 2] = u.value
    bounds[i, 3] = u.error
    bounds[i, 4] = v.value
    bounds[i, 5] = v.error
    bounds[i, 6] = w.value
    bounds[i, 7] = w.error
    bounds[i, 8] = denominator.value
    bounds[i, 9] = denominator.error
    flags[i] = int(fp.certain(n) and fp.certain(u) and fp.certain(v) and fp.certain(w) and fp.certain(denominator))
    result[i] = triangle(p0, p1, p2, p3, q0, q1, q2, direction)


def ivec(p):
    out = []
    for x in p:
        n, d = float(x).as_integer_ratio(); out.append(n*((1 << 149)//d))
    return out


def det(a, b, c):
    return a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])


def sign(x):
    return (x > 0)-(x < 0)


def reference(row):
    p = [ivec(v) for v in row[:4]]
    a, b, c = [[4*q[k]-sum(x[k] for x in p) for k in range(3)] for q in map(ivec, row[4:7])]
    d = ivec(row[7]); n = det(a, b, c); u, v, w = det(d, b, c), det(d, c, a), det(d, a, b)
    den = u+v+w
    expected = [0, 0]
    if den == 0:
        if n == 0: expected = [0, 1]
    elif n != 0 and sign(n) == sign(den):
        if all(t == 0 or sign(t) == sign(den) for t in [u, v, w]):
            expected = [0, 1] if any(t == 0 for t in [u, v, w]) else [sign(den), 0]
    return [Fraction(x, 1 << 447) for x in [n, u, v, w, den]], expected


w = Path('.work/vtm'); dest = w/'centroid-ray-filter-v3-oracle.json'; assert not dest.exists()
rng = np.random.default_rng(206)
rows = [rng.uniform(-4, 4, (2000, 8, 3)).astype(np.float32)]
# Random finite binary32 bit patterns include subnormals, signs and disparate scales.
bits = ((rng.integers(0, 2, (1000, 8, 3), dtype=np.uint32) << 31)
        | (rng.integers(0, 255, (1000, 8, 3), dtype=np.uint32) << 23)
        | rng.integers(0, 1 << 23, (1000, 8, 3), dtype=np.uint32))
rows.append(bits.view(np.float32))
hand_counts = {}
for name in ['hand-seed7', 'hand-seed7-midpoint']:
    data = np.load(w/(name+'-final-v1.npz'))
    ids = np.flatnonzero((data['selected'] != 0) & (data['signed_distance'] > 0))
    local = []
    for cell in data['full_tetrahedra'][ids]:
        for face in data['input_faces']:
            local.append(np.vstack([data['points'][cell], data['input_points'][face], [11, 23, 37]]))
    hand_counts[name] = len(local); rows.append(np.array(local, np.float32))
# Exact corner/edge hits, coplanar rays, parallel rays and zero determinants.
special = []
for exponent in [-140, -100, 0, 100, 125]:
    for direction in [[1, 1, 1], [1, 0, 0], [0, 1, 0], [11, 23, 37]]:
        for triangle_points in [[[1, 1, 1], [1, 1, -1], [1, -1, 1]],
                                [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                                [[0, 0, 0], [1, 0, 0], [2, 0, 0]]]:
            sites = np.array([[-.5, -.5, -.5], [.5, .5, -.5], [.5, -.5, .5], [-.5, .5, .5]], np.float32)
            special.append(np.vstack([sites*np.float32(2.0**exponent), np.array(triangle_points,np.float32)*np.float32(2.0**exponent), direction]))
rows.append(np.array(special,np.float32)); data = np.concatenate(rows)
assert np.isfinite(data).all()
source = wp.array(data, dtype=wp.vec3, device='cuda:0')
bounds = wp.empty((len(data), 10), dtype=wp.float64, device='cuda:0')
flags = wp.empty(len(data), dtype=int, device='cuda:0'); events = wp.empty(len(data), dtype=wp.vec2i, device='cuda:0')
wp.launch(inspect, len(data), inputs=[source, bounds, flags, events], block_dim=32)
b, f, actual = bounds.numpy(), flags.numpy(), events.numpy()
assert np.isfinite(b).all() and (b[:, 1::2] >= 0).all()
for i, row in enumerate(data):
    exact, expected = reference(row)
    np.testing.assert_array_equal(actual[i], expected, err_msg=str(i))
    for j, x in enumerate(exact):
        value, error = map(Fraction, map(float, b[i, 2*j:2*j+2]))
        assert value-error <= x <= value+error, (i,j,x,value,error)
snapshot = w/'centroid-ray-filter-v3-oracle.npz'; assert not snapshot.exists()
np.savez_compressed(snapshot, cases=data, bounds=b, filtered=f, events=actual)
record = dict(scope='Exact Python-integer/Fraction oracle checks all five binary64 enclosures and exact ray events for every case; instrumented correctness only, not timing.',
              cases=len(data), hand_cases=hand_counts, filtered=int(f.sum()), integer_fallback=int(len(f)-f.sum()),
              ambiguous_events=int((actual[:,1]!=0).sum()), all_event_matches=True, all_intervals_enclose_exact=True,
              source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [__file__,'.work/vtm/proven_exterior_filter_v2.py']},
              snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest())
dest.write_text(json.dumps(record,indent=2)+'\n'); print(json.dumps(record),flush=True)
