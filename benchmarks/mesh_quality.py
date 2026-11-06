"""Independent CPU measurements of a saved mesh; never used by the GPU mesher."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from tests.test_domain import brute_distances
from tests.test_extraction import independent_topology


def boundary_error(points, faces, target, triangles):
    tri = points[faces]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
    bary = np.full((3, 3), 1 / 6)
    np.fill_diagonal(bary, 2 / 3)
    samples = np.einsum("ij,fjk->fik", bary, tri).reshape(-1, 3)
    weights = np.repeat(area / 3, 3)
    distances = brute_distances(target, triangles, samples)
    integral = float(weights @ (distances * distances))
    return dict(
        area=float(weights.sum()),
        squared_integral=integral,
        rms=float(np.sqrt(integral / weights.sum())),
        max_sample=float(distances.max()),
    )


def measure(points, tetrahedra, input_points, input_faces):
    p, t = np.asarray(points, np.float64), np.asarray(tetrahedra)
    p0, f0 = np.asarray(input_points, np.float64), np.asarray(input_faces)
    if not len(t):
        raise ValueError("A nonempty mesh is required")
    cells = p[t]
    edges = cells[:, 1:] - cells[:, :1]
    volumes = np.linalg.det(edges) / 6
    if not np.all(volumes > 0):
        raise ValueError("All tetrahedra must have positive volume")
    offsets = np.linalg.solve(2 * edges, np.sum(edges * edges, axis=2)[..., None])[..., 0]
    radii = np.linalg.norm(offsets, axis=1)
    areas = np.zeros(len(t))
    normals = []
    for opposite in range(4):
        face = cells[:, [i for i in range(4) if i != opposite]]
        cross = np.cross(face[:, 1] - face[:, 0], face[:, 2] - face[:, 0])
        length = np.linalg.norm(cross, axis=1)
        areas += length / 2
        cross[np.einsum("ij,ij->i", cross, cells[:, opposite] - face[:, 0]) > 0] *= -1
        normals.append(cross / length[:, None])
    angles = np.concatenate(
        [
            np.degrees(np.arccos(np.clip(-np.einsum("ij,ij->i", normals[a], normals[b]), -1, 1)))
            for a, b in itertools.combinations(range(4), 2)
        ]
    )
    quality = 9 * volumes / (areas * radii)
    boundary, components, chi = independent_topology(p, t)
    a = boundary_error(p, boundary, p0, f0)
    b = boundary_error(p0, f0, p, boundary)
    symmetric = float(
        np.sqrt((a["squared_integral"] + b["squared_integral"]) / (a["area"] + b["area"]))
    )
    tri = p0[f0] - p0.mean(axis=0)
    domain_volume = float(
        np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6
    )
    percentiles = [0, 1, 5, 50, 95, 100]
    return dict(
        scope="Independent NumPy solves, face/edge incidence and brute-force triangle distances; outside mesher timing. Three quadrature points per boundary face, not a Hausdorff certificate.",
        sites=len(p),
        referenced_vertices=len(np.unique(t)),
        tetrahedra=len(t),
        boundary_triangles=len(boundary),
        components=components,
        boundary_euler=chi,
        volume=float(volumes.sum()),
        input_volume=domain_volume,
        relative_volume_error=float(volumes.sum() / domain_volume - 1),
        mean_radius_ratio=float(quality.mean()),
        radius_ratio_percentiles=dict(
            zip(map(str, percentiles), np.percentile(quality, percentiles).tolist())
        ),
        dihedral_degrees_min=float(angles.min()),
        dihedral_degrees_max=float(angles.max()),
        output_to_input=a,
        input_to_output=b,
        two_sided_area_weighted_rms=symmetric,
        two_sided_area_weighted_rms_percent_bbox_diagonal=float(
            100 * symmetric / np.linalg.norm(np.ptp(p0, axis=0))
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mesh", type=Path, help="NPZ containing points, tetrahedra, input_points, input_faces"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.mesh) as data:
        result = measure(
            data["points"], data["tetrahedra"], data["input_points"], data["input_faces"]
        )
    result["input_sha256"] = hashlib.sha256(args.mesh.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
