"""Fixed-capacity, device-count variants of optional dual kernels."""

from functools import lru_cache

import warp as wp

from . import _bvh
from ._cell import Cell64, Scratch64, clip, init_cell, plane_at
from ._tetrahedra import (
    box_distance2,
    key_hash,
    may_clip,
    power_residual,
    power_tolerance,
    read4,
    sorted4,
)

wp.set_module_options({"enable_backward": False, "fast_math": False})


@lru_cache(maxsize=None)
def make_cells(queue_capacity=128):
    queue_size = wp.constant(queue_capacity)

    @wp.kernel(enable_backward=False, module="unique", module_options={"fast_math": False})
    def build_cells(
        points: wp.array[wp.vec3],
        weights: wp.array[float],
        tree: _bvh.Tree,
        jobs: wp.array[int],
        first: wp.array[int],
        job_count: wp.array[int],
        bounds: wp.array[wp.vec3],
        scratch: Scratch64,
        status: wp.array[int],
        candidates: wp.array2d[int],
        candidate_count: wp.array[int],
    ):
        slot = wp.tid()
        job = first[0] + slot
        if job >= job_count[0]:
            return
        i = jobs[job]
        if status[i] == 0 or status[i] == 3:
            return
        if tree.owners[i] != i:
            status[i] = 3
            return
        p = wp.vec3d(points[i])
        wi = wp.float64(weights[i])
        cell = init_cell(scratch, slot, points[i], bounds[0], bounds[1], Cell64())
        nodes = wp.zeros(shape=queue_size, dtype=int)
        distances = wp.zeros(shape=queue_size, dtype=wp.float64)
        nodes[0] = 1
        distances[0] = wp.float64(0.0)
        size = int(1)
        while size > 0 and cell.status == 0:
            nearest = int(0)
            for j in range(1, size):
                if distances[j] < distances[nearest]:
                    nearest = j
            node = nodes[nearest]
            size -= 1
            nodes[nearest] = nodes[size]
            distances[nearest] = distances[size]
            if not may_clip(tree, node, p, wi, scratch, cell):
                continue
            if tree.count[node] > 0:
                for k in range(tree.start[node], tree.start[node] + tree.count[node]):
                    j = tree.ids[k]
                    if tree.owners[j] == j:
                        cell = clip(
                            scratch,
                            cell,
                            i,
                            j,
                            wp.vec3d(points[j]) - p,
                            weights[i],
                            weights[j],
                            False,
                        )
                        if cell.status != 0:
                            break
            else:
                for side in range(2):
                    child = tree.left[node]
                    if side == 1:
                        child = tree.right[node]
                    if size == queue_size:
                        cell.status = 4
                        break
                    nodes[size] = child
                    distances[size] = box_distance2(tree, child, p)
                    size += 1
        status[i] = cell.status
        if cell.status != 0:
            return
        # A true finite dual vertex is incident to three site bisectors. Box
        # planes have neighbor -1 and must never become tetrahedron vertices.
        for k in range(cell.nv):
            v = scratch.vertices[cell.base + 32 * k]
            a = scratch.neighbors[cell.base + 32 * plane_at(v, 0)]
            b = scratch.neighbors[cell.base + 32 * plane_at(v, 1)]
            c = scratch.neighbors[cell.base + 32 * plane_at(v, 2)]
            if i < a and i < b and i < c:
                tet = sorted4(wp.vec4i(i, a, b, c))
                at = wp.atomic_add(candidate_count, 0, 1)
                if at < candidates.shape[0]:
                    for j in range(4):
                        candidates[at, j] = tet[j]

    return build_cells


@wp.kernel(enable_backward=False)
def unique_candidates(
    candidates: wp.array2d[int],
    table: wp.array[int],
    active: wp.array[int],
):
    i = wp.tid()
    if i >= active[0]:
        return
    v = read4(candidates, i)
    slot = int(key_hash(v) & wp.uint32(table.shape[0] - 1))
    while True:
        other = wp.atomic_cas(table, slot, -1, i)
        if other == -1 or read4(candidates, other) == v:
            break
        slot = (slot + 1) & (table.shape[0] - 1)


@wp.kernel(enable_backward=False)
def validate_power_spheres(
    points: wp.array[wp.vec3],
    weights: wp.array[float],
    tree: _bvh.Tree,
    tets: wp.array2d[int],
    errors: wp.array[int],
    active: wp.array[int],
):
    i = wp.tid()
    if i >= active[0]:
        return
    v = read4(tets, i)
    p = wp.vec3d(points[v[0]])
    a = wp.vec3d(points[v[1]]) - p
    b = wp.vec3d(points[v[2]]) - p
    c = wp.vec3d(points[v[3]]) - p
    wi = wp.float64(weights[v[0]])
    da = (wp.dot(a, a) + wi - wp.float64(weights[v[1]])) * wp.float64(0.5)
    db = (wp.dot(b, b) + wi - wp.float64(weights[v[2]])) * wp.float64(0.5)
    dc = (wp.dot(c, c) + wi - wp.float64(weights[v[3]])) * wp.float64(0.5)
    center = (da * wp.cross(b, c) + db * wp.cross(c, a) + dc * wp.cross(a, b)) / wp.dot(
        a, wp.cross(b, c)
    )
    stack = wp.zeros(shape=128, dtype=int)
    stack[0] = 1
    size = int(1)
    while size > 0:
        size -= 1
        node = stack[size]
        lo = wp.vec3d(tree.lower[node]) - p
        hi = wp.vec3d(tree.upper[node]) - p
        q = wp.min(wp.max(center, lo), hi)
        delta = wi - wp.float64(tree.weight[node])
        if power_residual(q, center, delta) > power_tolerance(q, center, delta):
            continue
        if tree.count[node] > 0:
            for k in range(tree.start[node], tree.start[node] + tree.count[node]):
                j = tree.ids[k]
                if tree.owners[j] == j and j != v[0] and j != v[1] and j != v[2] and j != v[3]:
                    q = wp.vec3d(points[j]) - p
                    delta = wi - wp.float64(weights[j])
                    residual = power_residual(q, center, delta)
                    tolerance = power_tolerance(q, center, delta)
                    if residual < -tolerance:
                        wp.atomic_add(errors, 0, 1)
                    elif wp.abs(residual) <= tolerance:
                        wp.atomic_add(errors, 1, 1)
        else:
            if size + 2 > 128:
                wp.atomic_add(errors, 5, 1)
                return
            stack[size] = tree.left[node]
            stack[size + 1] = tree.right[node]
            size += 2


@wp.kernel(enable_backward=False)
def make_faces(
    tets: wp.array2d[int],
    faces: wp.array2d[int],
    active: wp.array[int],
):
    f = wp.tid()
    if f >= 4 * active[0]:
        return
    i = f // 4
    opposite = f % 4
    v = wp.vec4i(-1)
    at = int(0)
    for k in range(4):
        if k != opposite:
            v[at] = tets[i, k]
            at += 1
    v[3] = -1
    v = sorted4(v)
    for k in range(3):
        faces[f, k] = v[k + 1]
    faces[f, 3] = 0


@wp.kernel(enable_backward=False)
def index_faces(
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    faces: wp.array2d[int],
    table: wp.array[int],
    counts: wp.array[int],
    sides: wp.array[int],
    active: wp.array[int],
):
    f = wp.tid()
    if f >= 4 * active[0]:
        return
    v = read4(faces, f)
    p = wp.vec3d(points[v[0]])
    n = wp.cross(wp.vec3d(points[v[1]]) - p, wp.vec3d(points[v[2]]) - p)
    opposite = wp.vec3d(points[tets[f // 4, f % 4]]) - p
    side = int(1)
    if wp.dot(n, opposite) < wp.float64(0.0):
        side = -1
    slot = int(key_hash(v) & wp.uint32(table.shape[0] - 1))
    while True:
        other = wp.atomic_cas(table, slot, -1, f)
        if other == -1 or read4(faces, other) == v:
            wp.atomic_add(counts, slot, 1)
            wp.atomic_add(sides, slot, side)
            break
        slot = (slot + 1) & (table.shape[0] - 1)
