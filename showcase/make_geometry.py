"""Build display meshes from Warpower's actual power-diagram adjacency."""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection

import warpower


def main():
    rng = np.random.default_rng(54)
    candidates = rng.normal(size=(12000, 3))
    candidates /= np.linalg.norm(candidates, axis=1)[:, None]
    candidates *= (1.85 * rng.random(len(candidates)) ** (1 / 3))[:, None]
    chosen = [int(np.argmin(np.linalg.norm(candidates, axis=1)))]
    distance = np.full(len(candidates), np.inf)
    for _ in range(175):
        distance = np.minimum(distance, np.sum((candidates - candidates[chosen[-1]]) ** 2, axis=1))
        chosen.append(int(np.argmax(distance)))
    inner = candidates[chosen]
    k = np.arange(240)
    z = 1 - 2 * (k + 0.5) / len(k)
    theta = k * np.pi * (3 - np.sqrt(5))
    guards = 2.65 * np.column_stack(
        (np.sqrt(1 - z * z) * np.cos(theta), np.sqrt(1 - z * z) * np.sin(theta), z)
    )
    points = np.concatenate((inner, guards)).astype(np.float32)
    weights = np.zeros(len(points), dtype=np.float32)
    weights[: len(inner)] = rng.uniform(-0.065, 0.085, len(inner))
    diagram = warpower.power_diagram(points, weights)
    adjacency, offsets, status = (
        getattr(diagram, key).numpy() for key in ("adjacency", "offsets", "status")
    )
    assert np.all(status == 0), np.unique(status, return_counts=True)
    assert diagram.diagnostics["asymmetric_cells_remaining"] == 0
    p64, w64 = points.astype(float), weights.astype(float)
    meshes = []
    display_edges = set()
    max_violation = 0.0
    for i in range(len(inner)):
        ids = adjacency[offsets[i] : offsets[i + 1]]
        visible = [int(j) for j in ids if j < len(inner)]
        visible.sort(key=lambda j: np.sum((p64[j] - p64[i]) ** 2))
        for j in visible[:2]:
            display_edges.add(tuple(sorted((i, j))))
        q = p64[ids] - p64[i]
        a = 2 * q
        b = w64[ids] - w64[i] - np.sum(q * q, axis=1)
        norms = np.linalg.norm(a, axis=1)
        solution = linprog(
            [0, 0, 0, -1],
            A_ub=np.column_stack((a, norms)),
            b_ub=-b,
            bounds=[(None, None)] * 3 + [(0, None)],
            method="highs",
        )
        assert solution.success and solution.x[3] > 1e-7
        vertices = HalfspaceIntersection(np.column_stack((a, b)), solution.x[:3]).intersections
        all_q = p64 - p64[i]
        residual = vertices @ (2 * all_q).T + w64 - w64[i] - np.sum(all_q * all_q, axis=1)
        max_violation = max(max_violation, float(residual.max()))
        assert residual.max() < 1e-7, (i, residual.max())
        faces = []
        for normal, offset in zip(a, b):
            ids_on_face = np.flatnonzero(np.abs(vertices @ normal + offset) < 1e-7)
            if len(ids_on_face) < 3:
                continue
            n = normal / np.linalg.norm(normal)
            axis = np.eye(3)[np.argmin(np.abs(n))]
            u = np.cross(n, axis)
            u /= np.linalg.norm(u)
            v = np.cross(n, u)
            centered = vertices[ids_on_face] - vertices[ids_on_face].mean(axis=0)
            order = np.argsort(np.arctan2(centered @ v, centered @ u))
            faces.append(ids_on_face[order].tolist())
        world = vertices + p64[i]
        center = world.mean(axis=0)
        meshes.append(
            dict(
                id=i,
                center=center.tolist(),
                seed=p64[i].tolist(),
                vertices=(world - center).tolist(),
                faces=faces,
                weight=float(w64[i]),
            )
        )
    out = Path("results/local/showcase")
    out.mkdir(parents=True, exist_ok=True)
    payload = dict(
        seed=54,
        total_sites=len(points),
        displayed_cells=len(meshes),
        boundary_sites=len(guards),
        construction="warpower.power_diagram",
        reconstruction="float64 halfspace intersections using Warpower CSR neighbors",
        max_all_site_halfspace_violation=max_violation,
        statuses=dict(zip(*[a.tolist() for a in np.unique(status, return_counts=True)])),
        display_edges=sorted(display_edges),
        cells=meshes,
    )
    (out / "geometry.json").write_text(json.dumps(payload))
    np.savez_compressed(
        out / "sites.npz", points=points, weights=weights, adjacency=adjacency, offsets=offsets
    )
    print(
        f"Built {len(meshes)} actual cells from {len(points)} sites; maximum all-site violation {max_violation:.3g}."
    )


if __name__ == "__main__":
    main()
