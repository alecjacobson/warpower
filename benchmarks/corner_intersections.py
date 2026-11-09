"""Offline exact-rational diagnosis of rejected material-boundary candidates.

This CPU evidence tool is not imported by the mesher. It clips each overlapping
input triangle in exact tetrahedral barycentric coordinates. Closed contact and
intersection with the tetrahedron's strict interior are different results.
"""

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import numpy as np


def rational_points(points):
    return [tuple(Fraction(float(x)) for x in p) for p in points]


def determinant(points):
    a, b, c, d = points
    x, y, z = [tuple(v[i] - a[i] for i in range(3)) for v in (b, c, d)]
    return (
        x[0] * (y[1] * z[2] - y[2] * z[1])
        - x[1] * (y[0] * z[2] - y[2] * z[0])
        + x[2] * (y[0] * z[1] - y[1] * z[0])
    )


def barycentric(tet, point):
    total = determinant(tet)
    if total == 0:
        raise ValueError("tetrahedron must have nonzero volume")
    out = []
    for k in range(4):
        sub = list(tet)
        sub[k] = point
        out.append(determinant(sub) / total)
    return tuple(out)


def clip_triangle(tet, triangle):
    """Return closed-contact flag and optional strict-interior certificate.

    Certificate: nonnegative triangle barycentric weights summing to one.
    Clipping carries these alongside the four tet weights using exact rational
    interpolation. An average of all surviving vertices is strictly interior
    exactly when no tet coordinate vanishes on the whole clipped polygon.
    """
    polygon = [
        barycentric(tet, p) + tuple(Fraction(int(j == i)) for j in range(3))
        for i, p in enumerate(triangle)
    ]
    for k in range(4):
        if not polygon:
            return False, None
        clipped = []
        for p, q in zip(polygon, polygon[1:] + polygon[:1]):
            if p[k] >= 0:
                clipped.append(p)
            if (p[k] < 0 < q[k]) or (q[k] < 0 < p[k]):
                t = p[k] / (p[k] - q[k])
                clipped.append(tuple(a + t * (b - a) for a, b in zip(p, q)))
        polygon = clipped
    if not polygon:
        return False, None
    if any(all(p[k] == 0 for p in polygon) for k in range(4)):
        return True, None
    certificate = tuple(sum(p[k] for p in polygon) / len(polygon) for k in range(4, 7))
    verify_certificate(tet, triangle, certificate)
    return True, certificate


def verify_certificate(tet, triangle, certificate):
    """Reconstruct a point and check four exact determinants, without clipping."""
    assert len(certificate) == 3 and sum(certificate) == 1
    assert all(w >= 0 for w in certificate)
    point = tuple(sum(w * p[k] for w, p in zip(certificate, triangle)) for k in range(3))
    weights = barycentric(tet, point)
    assert sum(weights) == 1 and all(w > 0 for w in weights)


def candidates(data):
    cells, keep = data["full_tetrahedra"], data["selected"]
    material = data["tetrahedra"]
    faces = material[:, [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]]].reshape(-1, 3)
    _, index, counts = np.unique(
        np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True
    )
    boundary = faces[index[counts == 1]]
    edges, counts = np.unique(
        np.sort(boundary[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1),
        axis=0,
        return_counts=True,
    )
    bad = edges[counts != 2]
    touches = np.zeros(len(cells), bool)
    for edge in bad:
        touches |= np.sum(np.isin(cells, edge), axis=1) == 2
    return bad, np.flatnonzero(touches & (keep == 0))


def diagnose(path, reference_input=None):
    data = np.load(path)
    points, full = data["points"], data["full_tetrahedra"]
    surface, faces = data["input_points"], data["input_faces"]
    assert points.dtype == surface.dtype == np.float32
    assert np.isfinite(points).all() and np.isfinite(surface).all()
    bad, ids = candidates(data)
    triangles = surface[faces]
    lower, upper = triangles.min(axis=1), triangles.max(axis=1)
    exact_points, exact_surface = rational_points(points), rational_points(surface)
    rows, pair_count = [], 0
    reference = reference_input.open("w") if reference_input else None
    try:
        for cell in ids:
            p = points[full[cell]]
            overlap = np.flatnonzero(
                np.all(upper >= p.min(axis=0), axis=1) & np.all(lower <= p.max(axis=0), axis=1)
            )
            tet = [exact_points[k] for k in full[cell]]
            hits, boundary_only, separated = [], [], 0
            for face in overlap:
                triangle = [exact_surface[k] for k in faces[face]]
                contact, certificate = clip_triangle(tet, triangle)
                if reference:
                    coords = np.concatenate([p, triangles[face]]).ravel()
                    reference.write(
                        f"{int(cell)} {int(face)} "
                        + " ".join(format(float(x), ".17g") for x in coords)
                        + "\n"
                    )
                pair_count += 1
                if certificate is not None:
                    hits.append(dict(face=int(face), weights=[str(w) for w in certificate]))
                elif contact:
                    boundary_only.append(int(face))
                else:
                    separated += 1
            rows.append(
                dict(
                    cell=int(cell),
                    overlapping_boxes=len(overlap),
                    separated=separated,
                    boundary_only=boundary_only,
                    strict_hits=hits,
                )
            )
    finally:
        if reference:
            reference.close()
    return dict(
        scope=__doc__,
        input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        sites=len(points),
        bad_edges=bad.tolist(),
        candidates=len(ids),
        pair_count=pair_count,
        strict_crossing_cells=sum(bool(r["strict_hits"]) for r in rows),
        no_strict_crossing_cells=[r["cell"] for r in rows if not r["strict_hits"]],
        cells=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-input", type=Path)
    args = parser.parse_args()
    for p in (args.output, args.reference_input):
        if p is not None and p.exists():
            parser.error(f"refusing to overwrite {p}")
    report = diagnose(args.input, args.reference_input)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print({k: v for k, v in report.items() if k not in ["cells", "bad_edges"]}, flush=True)


if __name__ == "__main__":
    main()
