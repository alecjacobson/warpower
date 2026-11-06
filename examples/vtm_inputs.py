"""Fixed example inputs; host preparation happens before device meshing."""

from pathlib import Path

import numpy as np


def read_off(path):
    """Read an ASCII triangular OFF surface without modifying its geometry."""
    tokens = []
    for line in Path(path).read_text().splitlines():
        tokens.extend(line.partition("#")[0].split())
    it = iter(tokens)
    if next(it) != "OFF":
        raise ValueError("Expected an ASCII OFF file")
    n, m, _ = int(next(it)), int(next(it)), int(next(it))
    points = np.array([[float(next(it)) for _ in range(3)] for _ in range(n)], np.float32)
    faces = []
    for _ in range(m):
        if int(next(it)) != 3:
            raise ValueError("Only triangular OFF surfaces are supported")
        faces.append([int(next(it)) for _ in range(3)])
    if next(it, None) is not None:
        raise ValueError("Unexpected trailing OFF data")
    return points, np.array(faces, np.int32)


def write_off(path, points, faces):
    with Path(path).open("w") as f:
        f.write(f"OFF\n{len(points)} {len(faces)} 0\n")
        np.savetxt(f, points, fmt="%.17g")
        np.savetxt(f, np.column_stack((np.full(len(faces), 3), faces)), fmt="%d")


def torus(nu=48, nv=24, major=2.0, minor=0.7):
    u, v = np.meshgrid(
        np.arange(nu) * 2 * np.pi / nu, np.arange(nv) * 2 * np.pi / nv, indexing="ij"
    )
    points = np.stack(
        (
            (major + minor * np.cos(v)) * np.cos(u),
            (major + minor * np.cos(v)) * np.sin(u),
            minor * np.sin(v),
        ),
        axis=-1,
    )
    faces = []
    for i in range(nu):
        for j in range(nv):
            a, b = i * nv + j, ((i + 1) % nu) * nv + j
            c, d = ((i + 1) % nu) * nv + (j + 1) % nv, i * nv + (j + 1) % nv
            faces.extend(((a, b, c), (a, c, d)))
    return points.reshape(-1, 3).astype(np.float32), np.array(faces, np.int32)


def _triangulate_polygon(p):
    """Ear clipping of these simple CCW example polygons, retaining all corners."""

    def cross(a, b, c):
        u, v = b - a, c - a
        return u[0] * v[1] - u[1] * v[0]

    ring, triangles = list(range(len(p))), []
    while len(ring) > 3:
        for at, b in enumerate(ring):
            a, c = ring[at - 1], ring[(at + 1) % len(ring)]
            if cross(p[a], p[b], p[c]) <= 0:
                continue
            blocked = any(
                cross(p[a], p[b], p[q]) >= 0
                and cross(p[b], p[c], p[q]) >= 0
                and cross(p[c], p[a], p[q]) >= 0
                for q in ring
                if q not in (a, b, c)
            )
            if not blocked:
                triangles.append((a, b, c))
                del ring[at]
                break
        else:
            raise ValueError("Example outline is not a simple triangulable CCW polygon")
    triangles.append(tuple(ring))
    return np.array(triangles, np.int32)


def thin_hand():
    """Procedural palm and five narrow fingers, not the paper's scanned hand."""
    outline = np.array(
        [
            (-1, -1),
            (1, -1),
            (1, 0.4),
            (0.9, 0.4),
            (0.9, 1.4),
            (0.65, 1.4),
            (0.65, 0.4),
            (0.5, 0.4),
            (0.5, 1.8),
            (0.25, 1.8),
            (0.25, 0.4),
            (0.1, 0.4),
            (0.1, 2),
            (-0.15, 2),
            (-0.15, 0.4),
            (-0.3, 0.4),
            (-0.3, 1.7),
            (-0.55, 1.7),
            (-0.55, 0.4),
            (-0.75, 0.4),
            (-1.15, 0.95),
            (-1.4, 0.75),
            (-1, 0.1),
        ],
        np.float64,
    )
    n = len(outline)
    bottom = np.column_stack((outline, np.full(n, -0.12)))
    top = np.column_stack((outline, np.full(n, 0.12)))
    cap = _triangulate_polygon(outline)
    faces = [*cap[:, ::-1], *(cap + n)]
    for a in range(n):
        b = (a + 1) % n
        faces.extend(((a, b, b + n), (a, b + n, a + n)))
    return np.vstack((bottom, top)).astype(np.float32), np.array(faces, np.int32)


def cube(scale=1.0, center=(0.0, 0.0, 0.0)):
    points = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        np.float32,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        np.int32,
    )
    return (points * scale + np.array(center, np.float32)).astype(np.float32), faces


def combine(*surfaces):
    points, faces, offset = [], [], 0
    for p, f in surfaces:
        points.append(p)
        faces.append(f + offset)
        offset += len(p)
    return np.concatenate(points), np.concatenate(faces)


def surface(name, off=None):
    if name in ("torus", "graded-torus"):
        return torus()
    if name == "thin-hand":
        return thin_hand()
    if name == "cavity":
        p, f = cube(0.4)
        return combine(cube(), (p, f[:, ::-1].copy()))
    if name == "disconnected":
        return combine(cube(0.7, (-2, 0, 0)), cube(0.7, (2, 0, 0)))
    if off is None:
        raise ValueError(f"{name} requires --input path/to/triangular.off")
    return read_off(off)
