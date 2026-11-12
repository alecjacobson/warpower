"""Enumerate bounded facets using rational vertices; no distance/area tolerance."""

import json
from fractions import Fraction as F
from itertools import combinations
from pathlib import Path

import numpy as np


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def exact_graph(points):
    p = np.asarray(points, np.float32)
    lower = p.min(axis=0) - np.float32(1.00001)
    upper = p.max(axis=0) + np.float32(1.00001)
    pts = [[F(float(x)) for x in row] for row in p]
    n = len(p)
    graph = []
    for i in range(n):
        rows = []
        ids = []
        for j in range(n):
            if j == i:
                continue
            q = [pts[j][a] - pts[i][a] for a in range(3)]
            rows.append([-2 * x for x in q] + [dot(q, q)])
            ids.append(j)
        for k in range(6):
            axis, side = divmod(k, 2)
            sign = 1 - 2 * side
            r = [F(0)] * 4
            r[axis] = F(sign)
            r[3] = -sign * (F(float([lower, upper][side][axis])) - pts[i][axis])
            rows.append(r)
            ids.append(-1)
        vertices = [set() for _ in rows]
        for indexes in combinations(range(len(rows)), 3):
            a, b, c = [rows[k] for k in indexes]
            bc = cross(b, c)
            ca = cross(c, a)
            ab = cross(a, b)
            d = dot(a[:3], bc)
            if d == 0:
                continue
            x = tuple(-(a[3] * bc[k] + b[3] * ca[k] + c[3] * ab[k]) / d for k in range(3))
            values = [dot(r[:3], x) + r[3] for r in rows]
            if min(values) < 0:
                continue
            for k, v in enumerate(values):
                if v == 0:
                    vertices[k].add(x)
        neighbors = []
        for j, vs in zip(ids, vertices):
            if j < 0 or len(vs) < 3:
                continue
            vs = list(vs)
            a = vs[0]
            if any(
                any(cross([b[k] - a[k] for k in range(3)], [c[k] - a[k] for k in range(3)]))
                for b, c in combinations(vs[1:], 2)
            ):
                neighbors.append(j)
        graph.append(sorted(neighbors))
    return graph


if __name__ == "__main__":
    rows = json.loads(Path(__file__).with_name("five-site-fixtures.json").read_text())
    for row in rows:
        assert exact_graph(row["points"]) == row["exact_neighbors"]
    print("Verified all 35 cells in seven fixtures using rational arithmetic")
