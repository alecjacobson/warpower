"""Opt-in full-hull dual extraction and validation, entirely in Warp.

This module is deliberately separate from the adjacency construction kernels.
It reconstructs cells in float64, expands the clipping domain when necessary,
and checks the resulting tetrahedra before exposing them to callers.
"""

from functools import lru_cache

import numpy as np
import warp as wp

from . import _bvh
from ._canonical import mix
from ._cell import Cell64, Scratch64, allocate_scratch, clip, init_cell, plane_at, position
from ._construct import collect_failures

wp.set_module_options({"enable_backward": False, "fast_math": False})


@wp.func(inline=True)
def sorted4(v: wp.vec4i):
    for i in range(1, 4):
        for j in range(i, 0, -1):
            if v[j] < v[j - 1]:
                a = v[j]
                v[j] = v[j - 1]
                v[j - 1] = a
    return v


@wp.func(inline=True)
def key_hash(v: wp.vec4i):
    h = wp.uint32(0)
    for k in range(4):
        h = mix(h ^ wp.uint32(v[k]))
    return h


@wp.func(inline=True)
def read4(a: wp.array2d[int], i: int):
    return wp.vec4i(a[i, 0], a[i, 1], a[i, 2], a[i, 3])


@wp.func(inline=True)
def box_distance2(tree: _bvh.Tree, node: int, p: wp.vec3d):
    d = wp.max(
        wp.max(wp.vec3d(tree.lower[node]) - p, p - wp.vec3d(tree.upper[node])), wp.vec3d(0.0)
    )
    return wp.dot(d, d)


@wp.func(inline=True)
def may_clip(tree: _bvh.Tree, node: int, p: wp.vec3d, wi: wp.float64, s: Scratch64, cell: Cell64):
    lower = wp.vec3d(tree.lower[node]) - p
    upper = wp.vec3d(tree.upper[node]) - p
    delta = wi - wp.float64(tree.weight[node])
    # Conservative minimization over the entire node box, in seed coordinates.
    for k in range(cell.nv):
        x = position(s, cell.base + 32 * k)
        q = wp.min(wp.max(x, lower), upper)
        products = wp.cw_mul(x, q)
        q2 = wp.dot(q, q)
        value = q2 - wp.float64(2.0) * (products[0] + products[1] + products[2]) + delta
        scale = (
            q2
            + wp.float64(2.0) * (wp.abs(products[0]) + wp.abs(products[1]) + wp.abs(products[2]))
            + wp.abs(delta)
        )
        if value <= wp.float64(1.0e-12) * (scale + wp.float64(1.0e-100)):
            return True
    return False


@lru_cache(maxsize=None)
def make_cells(queue_capacity=128):
    queue_size = wp.constant(queue_capacity)

    @wp.kernel(enable_backward=False, module="unique", module_options={"fast_math": False})
    def build_cells(
        points: wp.array[wp.vec3],
        weights: wp.array[float],
        tree: _bvh.Tree,
        jobs: wp.array[int],
        first: int,
        lower: wp.vec3,
        upper: wp.vec3,
        scratch: Scratch64,
        status: wp.array[int],
        candidates: wp.array2d[int],
        candidate_count: wp.array[int],
    ):
        slot = wp.tid()
        i = jobs[first + slot]
        if tree.owners[i] != i:
            status[i] = 3
            return
        p = wp.vec3d(points[i])
        wi = wp.float64(weights[i])
        cell = init_cell(scratch, slot, points[i], lower, upper, Cell64())
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
def unique_candidates(candidates: wp.array2d[int], table: wp.array[int]):
    i = wp.tid()
    v = read4(candidates, i)
    slot = int(key_hash(v) & wp.uint32(table.shape[0] - 1))
    while True:
        other = wp.atomic_cas(table, slot, -1, i)
        if other == -1 or read4(candidates, other) == v:
            break
        slot = (slot + 1) & (table.shape[0] - 1)


@wp.func(inline=True)
def determinant_scale(a: wp.vec3d, b: wp.vec3d, c: wp.vec3d):
    # The norm product is a simple conservative scale for the determinant.
    return wp.length(a) * wp.length(b) * wp.length(c)


@wp.kernel(enable_backward=False)
def orient_unique(
    points: wp.array[wp.vec3],
    candidates: wp.array2d[int],
    table: wp.array[int],
    output: wp.array2d[int],
    count: wp.array[int],
    errors: wp.array[int],
):
    slot = wp.tid()
    source = table[slot]
    if source < 0:
        return
    v = read4(candidates, source)
    p = wp.vec3d(points[v[0]])
    a = wp.vec3d(points[v[1]]) - p
    b = wp.vec3d(points[v[2]]) - p
    c = wp.vec3d(points[v[3]]) - p
    det = wp.dot(a, wp.cross(b, c))
    if wp.abs(det) <= wp.float64(1.0e-13) * determinant_scale(a, b, c):
        wp.atomic_add(errors, 2, 1)
        return
    if det < wp.float64(0.0):
        j = v[2]
        v[2] = v[3]
        v[3] = j
    at = wp.atomic_add(count, 0, 1)
    for k in range(4):
        output[at, k] = v[k]


@wp.func(inline=True)
def power_residual(q: wp.vec3d, center: wp.vec3d, delta: wp.float64):
    return wp.dot(q, q) - wp.float64(2.0) * wp.dot(center, q) + delta


@wp.func(inline=True)
def power_tolerance(q: wp.vec3d, center: wp.vec3d, delta: wp.float64):
    v = wp.cw_mul(center, q)
    return wp.float64(1.0e-12) * (
        wp.dot(q, q)
        + wp.float64(2.0) * (wp.abs(v[0]) + wp.abs(v[1]) + wp.abs(v[2]))
        + wp.abs(delta)
        + wp.float64(1.0e-100)
    )


@wp.kernel(enable_backward=False)
def validate_power_spheres(
    points: wp.array[wp.vec3],
    weights: wp.array[float],
    tree: _bvh.Tree,
    tets: wp.array2d[int],
    errors: wp.array[int],
):
    i = wp.tid()
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
def make_faces(tets: wp.array2d[int], faces: wp.array2d[int]):
    f = wp.tid()
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
):
    f = wp.tid()
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


@wp.kernel(enable_backward=False)
def validate_faces(
    points: wp.array[wp.vec3],
    tree: _bvh.Tree,
    faces: wp.array2d[int],
    table: wp.array[int],
    counts: wp.array[int],
    sides: wp.array[int],
    errors: wp.array[int],
):
    slot = wp.tid()
    count = counts[slot]
    if count == 0:
        return
    if count > 2 or (count == 2 and sides[slot] != 0):
        wp.atomic_add(errors, 3, 1)
        return
    if count == 2:
        return
    v = read4(faces, table[slot])
    p = wp.vec3d(points[v[0]])
    n = wp.cross(wp.vec3d(points[v[1]]) - p, wp.vec3d(points[v[2]]) - p) * wp.float64(-sides[slot])
    n = wp.normalize(n)
    stack = wp.zeros(shape=128, dtype=int)
    stack[0] = 1
    size = int(1)
    while size > 0:
        size -= 1
        node = stack[size]
        lo = wp.vec3d(tree.lower[node]) - p
        hi = wp.vec3d(tree.upper[node]) - p
        q = lo
        for axis in range(3):
            if n[axis] > wp.float64(0.0):
                q[axis] = hi[axis]
        tolerance = wp.float64(1.0e-12) * (wp.length(q) + wp.float64(1.0e-100))
        if wp.dot(n, q) <= tolerance:
            continue
        if tree.count[node] > 0:
            for k in range(tree.start[node], tree.start[node] + tree.count[node]):
                q = wp.vec3d(points[tree.ids[k]]) - p
                if wp.dot(n, q) > wp.float64(1.0e-12) * (wp.length(q) + wp.float64(1.0e-100)):
                    wp.atomic_add(errors, 4, 1)
                    return
        else:
            if size + 2 > 128:
                wp.atomic_add(errors, 5, 1)
                return
            stack[size] = tree.left[node]
            stack[size + 1] = tree.right[node]
            size += 2


def _extract(points, weights, tree, lower, upper, chunk_size):
    from ._api import _chunk_size

    n, device = len(points), points.device
    output_capacity = max(8 * n, 16)
    while True:
        candidates = wp.empty((output_capacity, 4), dtype=int, device=device)
        candidate_count = wp.zeros(1, dtype=int, device=device)
        status = wp.full(n, 1, dtype=int, device=device)
        jobs = tree.ids
        for capacity in (128, 256, 512):
            size = _chunk_size(len(jobs), capacity, device, chunk_size, bytes_per_slot=100)
            scratch = allocate_scratch(size, capacity, device, robust=True)
            kernel = make_cells(capacity)
            for start in range(0, len(jobs), size):
                wp.launch(
                    kernel,
                    min(size, len(jobs) - start),
                    inputs=[
                        points,
                        weights,
                        tree,
                        jobs,
                        start,
                        lower,
                        upper,
                        scratch,
                        status,
                        candidates,
                        candidate_count,
                    ],
                    block_dim=32,
                    device=device,
                )
            failed_jobs = wp.empty(n, dtype=int, device=device)
            failed_count = wp.zeros(1, dtype=int, device=device)
            wp.launch(
                collect_failures, n, inputs=[status, failed_jobs, failed_count], device=device
            )
            failed = int(failed_count.numpy()[0])
            del scratch
            if failed == 0:
                break
            jobs = failed_jobs[:failed]
        if failed:
            raise ValueError(
                f"Tetrahedralization failed to construct {failed} cells in float64; no partial mesh was returned"
            )
        count = int(candidate_count.numpy()[0])
        if count <= output_capacity:
            return candidates[:count]
        output_capacity = count


def build(points, weights, *, chunk_size=None, leaf_size=4):
    """Return a validated full-hull tetrahedralization or raise on ambiguity."""
    n, device = len(points), points.device
    if n < 4:
        return wp.empty((0, 4), dtype=int, device=device), {
            "tetrahedra_count": 0,
            "tetrahedra_expansions": 0,
        }
    hierarchy = _bvh.build(points, weights, leaf_size)
    bounds = hierarchy.bounds.numpy().astype(np.float64)
    span = float(np.max(bounds[1] - bounds[0]))
    if span == 0:
        return wp.empty((0, 4), dtype=int, device=device), {
            "tetrahedra_count": 0,
            "tetrahedra_expansions": 0,
        }
    padding = max(span * 0.5, np.finfo(np.float32).tiny)
    for expansion in range(12):
        lower = (bounds[0] - padding).astype(np.float32)
        upper = (bounds[1] + padding).astype(np.float32)
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            break
        candidates = _extract(
            points, weights, hierarchy.tree, wp.vec3(lower), wp.vec3(upper), chunk_size
        )
        count = len(candidates)
        if count:
            table = wp.full(1 << (2 * count - 1).bit_length(), -1, dtype=int, device=device)
            wp.launch(unique_candidates, count, inputs=[candidates, table], device=device)
            output = wp.empty((count, 4), dtype=int, device=device)
            actual = wp.zeros(1, dtype=int, device=device)
            errors = wp.zeros(6, dtype=int, device=device)
            wp.launch(
                orient_unique,
                len(table),
                inputs=[points, candidates, table, output, actual, errors],
                device=device,
            )
            m = int(actual.numpy()[0])
            output = output[:m]
            if m:
                wp.launch(
                    validate_power_spheres,
                    m,
                    inputs=[points, weights, hierarchy.tree, output, errors],
                    block_dim=32,
                    device=device,
                )
                faces = wp.empty((4 * m, 4), dtype=int, device=device)
                table = wp.full(1 << (8 * m - 1).bit_length(), -1, dtype=int, device=device)
                counts = wp.zeros(len(table), dtype=int, device=device)
                sides = wp.zeros(len(table), dtype=int, device=device)
                wp.launch(make_faces, 4 * m, inputs=[output, faces], device=device)
                wp.launch(
                    index_faces,
                    4 * m,
                    inputs=[points, output, faces, table, counts, sides],
                    device=device,
                )
                wp.launch(
                    validate_faces,
                    len(table),
                    inputs=[points, hierarchy.tree, faces, table, counts, sides, errors],
                    block_dim=32,
                    device=device,
                )
            problems = errors.numpy()
            if problems[1] or problems[2]:
                raise ValueError(
                    "Tetrahedralization is degenerate or numerically ambiguous (cospherical/copower or near-coplanar sites); no partial mesh was returned"
                )
            if problems[0] or problems[3] or problems[5]:
                raise ValueError(
                    f"Tetrahedralization failed power-sphere/topology validation: {problems.tolist()}; no partial mesh was returned"
                )
            if m and problems[4] == 0:
                return output, {
                    "tetrahedra_count": m,
                    "tetrahedra_expansions": expansion,
                    "tetrahedra_padding": padding,
                    "tetrahedra_validation": "positive orientation, empty power spheres, paired interior faces, supporting convex-hull boundary",
                }
        padding *= 8
    raise ValueError(
        "Could not certify a complete 3D tetrahedralization after domain expansion; sites may be lower-dimensional or numerically ill-conditioned. No partial mesh was returned"
    )
