"""Independent double-precision, all-halfspace reference for small problems."""

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection


def reference_cells(points, weights=None, padding=1.00001):
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    weights = np.zeros(n) if weights is None else np.asarray(weights, dtype=np.float64)
    lo, hi = points.min(axis=0) - padding, points.max(axis=0) + padding
    neighbors, vertices = [], []
    for i, p in enumerate(points):
        ids = np.arange(n)[np.arange(n) != i]
        q = points[ids] - p
        normals = 2 * q
        offsets = weights[ids] - weights[i] - np.einsum("ij,ij->i", q, q)
        # All computations use coordinates relative to p, avoiding cancellation.
        a = np.concatenate((normals, np.eye(3), -np.eye(3)))
        b = np.concatenate((offsets, p - hi, lo - p))
        norms = np.linalg.norm(a, axis=1)
        zero = norms == 0
        if np.any(b[zero] > 0):
            neighbors.append(set()); vertices.append(np.empty((0, 3))); continue
        active = ~zero
        a, b, norms = a[active], b[active], norms[active]
        result = linprog([0, 0, 0, -1], A_ub=np.column_stack((a, norms)), b_ub=-b,
                         bounds=[(None, None)] * 3 + [(0, None)], method="highs")
        if not result.success or result.x[3] < 1e-11:
            neighbors.append(set()); vertices.append(np.empty((0, 3))); continue
        v = HalfspaceIntersection(np.column_stack((a, b)), result.x[:3]).intersections
        row = set()
        for j, normal, offset in zip(ids, normals, offsets):
            length = np.linalg.norm(normal)
            if length == 0:
                continue
            on = v[np.abs(v @ normal + offset) <= 2e-7 * length]
            if len(on) >= 3 and np.linalg.matrix_rank(on - on[0], tol=1e-8) >= 2:
                row.add(int(j))
        neighbors.append(row)
        vertices.append(v + p)
    return neighbors, vertices


def csr_sets(adjacency, offsets):
    return [set(map(int, adjacency[offsets[i]:offsets[i + 1]]))
            for i in range(len(offsets) - 1)]
