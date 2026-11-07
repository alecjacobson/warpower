"""Offline feature-fit measurements; never imported by the device mesher.

Nearest-chain assignment describes the saved geometry, not the mesher's sample
ownership. Centroid fitting does not promise exact edge/curve conformity.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tests.test_domain import brute_distances


def feature_graph(points, faces, feature_angle=45.0, corner_angle=45.0):
    """Independent float64 normals and segment graph for a closed oriented surface."""
    if not (0 < feature_angle < 180 and 0 < corner_angle < 180):
        raise ValueError("Feature and corner angles must be between 0 and 180 degrees")
    p = np.asarray(points, np.float64)
    tri = p[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths == 0) or not np.isfinite(p).all():
        raise ValueError("Require finite points and nondegenerate triangles")
    normals /= lengths[:, None]
    incidence = {}
    for i, face in enumerate(faces):
        for k in range(3):
            a, b = int(face[k]), int(face[(k + 1) % 3])
            incidence.setdefault(tuple(sorted((a, b))), []).append((i, a < b))
    if any(len(v) != 2 or v[0][1] == v[1][1] for v in incidence.values()):
        raise ValueError("Require oppositely paired input edges")
    edges = np.array(
        [
            e
            for e, fs in sorted(incidence.items())
            if normals[fs[0][0]] @ normals[fs[1][0]] < np.cos(np.deg2rad(feature_angle))
        ],
        np.int32,
    ).reshape(-1, 2)
    adjacency = {}
    for a, b in edges.tolist():
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)
    corners = set()
    for a, neighbors in adjacency.items():
        if len(neighbors) != 2:
            corners.add(a)
        else:
            directions = p[neighbors] - p[a]
            directions /= np.linalg.norm(directions, axis=1)[:, None]
            if directions[0] @ directions[1] > -np.cos(np.deg2rad(corner_angle)):
                corners.add(a)
    unused = set(map(tuple, edges.tolist()))
    chains = []

    def walk(a, b):
        path = [a, b]
        unused.remove(tuple(sorted((a, b))))
        while b not in corners and b != path[0]:
            c = next(c for c in adjacency[b] if c != a)
            unused.remove(tuple(sorted((b, c))))
            path.append(c)
            a, b = b, c
        chains.append(path)

    for a in sorted(corners):
        for b in sorted(adjacency[a]):
            if tuple(sorted((a, b))) in unused:
                walk(a, b)
    while unused:
        walk(*min(unused))
    return edges, sorted(corners), chains


def project_chain(queries, vertices):
    """Distance and arclength of the nearest point on a piecewise linear curve."""
    a, delta = vertices[:-1], np.diff(vertices, axis=0)
    length = np.linalg.norm(delta, axis=1)
    offset = queries[:, None, :] - a[None, :, :]
    fraction = np.clip(np.einsum("qsi,si->qs", offset, delta) / length**2, 0, 1)
    d2 = np.sum((offset - fraction[:, :, None] * delta[None, :, :]) ** 2, axis=2)
    which = np.argmin(d2, axis=1)
    row = np.arange(len(queries))
    arclength = np.r_[0, np.cumsum(length)]
    s = arclength[which] + fraction[row, which] * length[which]
    return np.sqrt(d2[row, which]), s, float(arclength[-1])


def measure(
    points,
    boundary,
    roles,
    material_vertices,
    input_points,
    input_faces,
    *,
    feature_angle=45.0,
    corner_angle=45.0,
    samples_per_edge=16,
    distance_query=brute_distances,
):
    if samples_per_edge < 1:
        raise ValueError("samples_per_edge must be positive")
    p, ip = np.asarray(points, np.float64), np.asarray(input_points, np.float64)
    roles = np.asarray(roles)
    edges, corners, chains = feature_graph(ip, input_faces, feature_angle, corner_angle)
    boundary_vertices = np.unique(boundary)
    if not len(boundary_vertices):
        raise ValueError("Require a nonempty output boundary")
    boundary_edges = {
        tuple(sorted((int(f[k]), int(f[(k + 1) % 3])))) for f in boundary for k in range(3)
    }
    corner_rows = []
    for i in corners:
        distance = np.linalg.norm(p[boundary_vertices] - ip[i], axis=1)
        at = int(np.argmin(distance))
        site = int(boundary_vertices[at])
        corner_rows.append(
            dict(
                input_id=i,
                nearest_boundary_site=site,
                distance=float(distance[at]),
                exact_coordinate_match=bool(np.array_equal(p[site], ip[i])),
                role=int(roles[site]),
            )
        )
    anchors = {
        row["input_id"]: row["nearest_boundary_site"]
        for row in corner_rows
        if row["exact_coordinate_match"]
    }
    crease_ids = np.flatnonzero(roles == 2)
    distance, arclength, length = [], [], []
    for chain in chains:
        d, s, total = project_chain(p[crease_ids], ip[chain])
        distance.append(d)
        arclength.append(s)
        length.append(total)
    assigned = np.argmin(distance, axis=0) if chains else np.full(len(crease_ids), -1)
    crease_distances = np.min(distance, axis=0) if chains else np.array([])
    records = []
    for i, chain in enumerate(chains):
        selected = (assigned == i) & np.isin(crease_ids, boundary_vertices)
        ids, s = crease_ids[selected], arclength[i][selected]
        closed = chain[0] == chain[-1]
        endpoint_anchors = []
        if not closed:
            for vertex, position in [(chain[0], 0.0), (chain[-1], length[i])]:
                if vertex in anchors and anchors[vertex] not in ids:
                    endpoint_anchors.append(anchors[vertex])
                    ids, s = np.r_[ids, anchors[vertex]], np.r_[s, position]
        order = np.lexsort((ids, s))
        ids, s = ids[order], s[order]
        gaps = (
            np.diff(np.r_[s, s[0] + length[i]])
            if closed and len(s)
            else np.diff(np.r_[0, s, length[i]])
        )
        pairs = list(zip(ids[:-1], ids[1:]))
        if closed and len(ids) > 1:
            pairs.append((ids[-1], ids[0]))
        missing = [
            list(map(int, pair))
            for pair in pairs
            if tuple(sorted(map(int, pair))) not in boundary_edges
        ]
        records.append(
            dict(
                input_vertex_ids=chain,
                closed=closed,
                length=length[i],
                assigned_boundary_crease_sites=int(selected.sum()),
                exact_endpoint_anchors=endpoint_anchors,
                ordered_boundary_site_ids=ids.tolist(),
                maximum_projection_distance=(
                    float(max(distance[i][selected])) if selected.any() else None
                ),
                maximum_arclength_gap=float(max(gaps)),
                maximum_gap_fraction_of_length=float(max(gaps) / length[i]),
                consecutive_pairs=len(pairs),
                pairs_connected_by_boundary_edge=len(pairs) - len(missing),
                missing_boundary_pairs=missing,
            )
        )
    fit = None
    if len(edges):
        fractions = (np.arange(samples_per_edge) + 0.5) / samples_per_edge
        segments = ip[edges]
        samples = (
            segments[:, None, 0, :] * (1 - fractions[None, :, None])
            + segments[:, None, 1, :] * fractions[None, :, None]
        ).reshape(-1, 3)
        weights = np.repeat(
            np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1) / samples_per_edge,
            samples_per_edge,
        )
        distances = distance_query(p, boundary, samples)
        fit = dict(
            samples_per_edge=samples_per_edge,
            quadrature="Uniform midpoint samples weighted by input segment length; no Hausdorff bound.",
            sample_count=len(samples),
            rms=float(np.sqrt(weights @ distances**2 / weights.sum())),
            maximum_sample_distance=float(distances.max()),
        )
    return dict(
        scope=(
            "Offline float64 feature graph and nearest-chain projection. Boundary-referenced "
            "role-2 sites are assigned to their nearest input chain, with exact corners as "
            "open-chain anchors. Connectivity/gaps are diagnostics, not a proof of feature "
            "conformity or runtime sample ownership. Corner coordinates are checked exactly."
        ),
        feature_angle=feature_angle,
        corner_angle=corner_angle,
        input_feature_edges=len(edges),
        input_corners=len(corners),
        feature_chains=len(chains),
        bbox_diagonal=float(np.linalg.norm(np.ptp(ip, axis=0))),
        role_counts={str(i): int(np.sum(roles == i)) for i in range(4)},
        crease_sites_on_material=int(np.isin(crease_ids, material_vertices).sum()),
        crease_sites_on_boundary=int(np.isin(crease_ids, boundary_vertices).sum()),
        crease_site_distance_percentiles=(
            dict(zip(map(str, [0, 50, 95, 100]), np.percentile(crease_distances, [0, 50, 95, 100])))
            if len(crease_distances)
            else None
        ),
        corner_sites_on_boundary=int(np.isin(np.flatnonzero(roles == 3), boundary_vertices).sum()),
        corners=corner_rows,
        chains=records,
        feature_to_boundary=fit,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-angle", type=float, default=45.0)
    parser.add_argument("--corner-angle", type=float, default=45.0)
    parser.add_argument("--samples-per-edge", type=int, default=16)
    parser.add_argument("--distance-binary", type=Path)
    args = parser.parse_args()
    distances = brute_distances
    if args.distance_binary:
        from benchmarks.surface_distance import NativeSurfaceDistance

        distances = NativeSurfaceDistance(args.distance_binary, args.output.parent / "scratch")
    with np.load(args.mesh) as data:
        result = measure(
            data["points"],
            data["boundary"],
            data["roles"],
            np.unique(data["tetrahedra"]),
            data["input_points"],
            data["input_faces"],
            feature_angle=args.feature_angle,
            corner_angle=args.corner_angle,
            samples_per_edge=args.samples_per_edge,
            distance_query=distances,
        )
    if args.distance_binary:
        result["distance_oracle"] = distances.metadata
    result["mesh_sha256"] = hashlib.sha256(args.mesh.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ["corners", "chains"]}))


if __name__ == "__main__":
    main()
