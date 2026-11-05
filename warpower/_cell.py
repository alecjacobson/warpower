"""Per-cell convex clipping, adapted from Paragram under Apache-2.0.

Topology stores oriented triples of incident planes. Scratch is interleaved
across 32 independent cells, so each vertex/plane loop uses coalesced loads.
All planes and vertices use coordinates relative to the cell's seed.
"""

from typing import Any

import warp as wp

wp.set_module_options({"enable_backward": False})


@wp.struct
class Scratch:
    planes: wp.array[wp.vec4]
    vertices: wp.array[wp.uint32]
    positions: wp.array[wp.vec4]
    next: wp.array[int]
    neighbors: wp.array[int]
    capacity: int


@wp.struct
class Scratch64:
    planes: wp.array[wp.vec4d]
    vertices: wp.array[wp.uint32]
    positions: wp.array[wp.vec4d]
    cartesian: wp.array[wp.vec3d]
    next: wp.array[int]
    neighbors: wp.array[int]
    capacity: int


@wp.struct
class Cell:
    base: int
    nv: int
    np: int
    status: int
    lo: wp.vec3
    hi: wp.vec3
    r2: float


@wp.struct
class Cell64:
    base: int
    nv: int
    np: int
    status: int
    lo: wp.vec3d
    hi: wp.vec3d
    r2: wp.float64


@wp.func(inline=True)
def set_position(s: Scratch, at: int, h: wp.vec4):
    s.positions[at] = h


@wp.func(inline=True)
def set_position(s: Scratch64, at: int, h: wp.vec4d):  # noqa: F811
    s.positions[at] = h
    inverse_w = wp.float64(1.0) / h[3]
    s.cartesian[at] = wp.vec3d(h[0], h[1], h[2]) * inverse_w


@wp.func(inline=True)
def position(s: Scratch, at: int):
    h = s.positions[at]
    return wp.vec3(h[0], h[1], h[2]) * (1.0 / h[3])


@wp.func(inline=True)
def position(s: Scratch64, at: int):  # noqa: F811
    return s.cartesian[at]


@wp.func(inline=True)
def swap_positions(s: Scratch, a: int, b: int):
    h = s.positions[a]
    s.positions[a] = s.positions[b]
    s.positions[b] = h


@wp.func(inline=True)
def swap_positions(s: Scratch64, a: int, b: int):  # noqa: F811
    h = s.positions[a]
    p = s.cartesian[a]
    s.positions[a] = s.positions[b]
    s.cartesian[a] = s.cartesian[b]
    s.positions[b] = h
    s.cartesian[b] = p


@wp.func(inline=True)
def encode(a: int, b: int, c: int):
    return wp.uint32(a) | (wp.uint32(b) << wp.uint32(10)) | (wp.uint32(c) << wp.uint32(20))


@wp.func(inline=True)
def plane_at(v: wp.uint32, i: int):
    return int((v >> wp.uint32(10 * i)) & wp.uint32(1023))


@wp.func(inline=True)
def coordinates(s: Any, base: int, v: wp.uint32):
    a = s.planes[base + 32 * plane_at(v, 0)]
    b = s.planes[base + 32 * plane_at(v, 1)]
    c = s.planes[base + 32 * plane_at(v, 2)]
    bc0 = b[1] * c[2] - b[2] * c[1]
    bc1 = b[2] * c[0] - b[0] * c[2]
    bc2 = b[0] * c[1] - b[1] * c[0]
    x0 = -a[3] * bc0 - b[3] * (c[1] * a[2] - c[2] * a[1]) - c[3] * (a[1] * b[2] - a[2] * b[1])
    x1 = -a[3] * bc1 - b[3] * (c[2] * a[0] - c[0] * a[2]) - c[3] * (a[2] * b[0] - a[0] * b[2])
    x2 = -a[3] * bc2 - b[3] * (c[0] * a[1] - c[1] * a[0]) - c[3] * (a[0] * b[1] - a[1] * b[0])
    return type(a)(x0, x1, x2, a[0] * bc0 + a[1] * bc1 + a[2] * bc2)


@wp.func(inline=True)
def init_cell(s: Any, slot: int, seed: wp.vec3, lower: wp.vec3, upper: wp.vec3, cell: Any):
    cell.base = (slot // 32) * (32 * s.capacity) + slot % 32
    cell.nv = 8
    cell.np = 6
    cell.status = 0
    cell.lo = type(cell.lo)(lower) - type(cell.lo)(seed)
    cell.hi = type(cell.hi)(upper) - type(cell.hi)(seed)
    cell.r2 = cell.lo.dtype(0.0)
    for axis in range(3):
        low = s.planes.dtype(s.planes.dtype.dtype(0.0))
        high = s.planes.dtype(s.planes.dtype.dtype(0.0))
        low[axis] = low.dtype(1.0)
        low[3] = -cell.lo[axis]
        high[axis] = high.dtype(-1.0)
        high[3] = cell.hi[axis]
        s.planes[cell.base + 32 * (2 * axis)] = low
        s.planes[cell.base + 32 * (2 * axis + 1)] = high
        d = wp.max(wp.abs(cell.lo[axis]), wp.abs(cell.hi[axis]))
        cell.r2 += d * d
    s.vertices[cell.base + 32 * 0] = encode(2, 5, 0)
    s.vertices[cell.base + 32 * 1] = encode(5, 3, 0)
    s.vertices[cell.base + 32 * 2] = encode(1, 5, 2)
    s.vertices[cell.base + 32 * 3] = encode(5, 1, 3)
    s.vertices[cell.base + 32 * 4] = encode(4, 2, 0)
    s.vertices[cell.base + 32 * 5] = encode(4, 0, 3)
    s.vertices[cell.base + 32 * 6] = encode(2, 4, 1)
    s.vertices[cell.base + 32 * 7] = encode(4, 3, 1)
    for i in range(6):
        s.neighbors[cell.base + 32 * i] = -1
    for i in range(8):
        at = cell.base + 32 * i
        set_position(s, at, coordinates(s, cell.base, s.vertices[at]))
    return cell


@wp.func(inline=True)
def update_bounds(s: Any, cell: Any):
    lo = type(cell.lo)(cell.lo.dtype(1.0e30))
    hi = type(cell.hi)(cell.hi.dtype(-1.0e30))
    r2 = cell.lo.dtype(0.0)
    for i in range(cell.nv):
        p = position(s, cell.base + 32 * i)
        lo = wp.min(lo, p)
        hi = wp.max(hi, p)
        r2 = wp.max(r2, wp.dot(p, p))
    cell.lo = lo
    cell.hi = hi
    cell.r2 = r2
    return cell


@wp.func(inline=True)
def collect_planes(s: Any, cell: Any):
    for i in range(cell.np):
        s.next[cell.base + 32 * i] = -1
    for i in range(cell.nv):
        v = s.vertices[cell.base + 32 * i]
        for j in range(3):
            s.next[cell.base + 32 * plane_at(v, j)] = 0
    count = int(0)
    for i in range(cell.np):
        at = cell.base + 32 * i
        if s.next[at] == 0:
            to = cell.base + 32 * count
            s.planes[to] = s.planes[at]
            s.neighbors[to] = s.neighbors[at]
            s.next[at] = count
            count += 1
    for i in range(cell.nv):
        at = cell.base + 32 * i
        v = s.vertices[at]
        a = s.next[cell.base + 32 * plane_at(v, 0)]
        b = s.next[cell.base + 32 * plane_at(v, 1)]
        c = s.next[cell.base + 32 * plane_at(v, 2)]
        s.vertices[at] = encode(a, b, c)
    cell.np = count
    return cell


@wp.func(inline=True)
def outside(h: Any, plane: Any):
    # Homogeneous determinants may have either orientation.
    value = wp.dot(h, plane)
    return (value > h.dtype(0.0) and h[3] < h.dtype(0.0)) or (
        value < h.dtype(0.0) and h[3] > h.dtype(0.0)
    )


@wp.func(inline=True)
def clip(
    s: Any,
    cell: Any,
    i: int,
    j: int,
    q: Any,
    wi: float,
    wj: float,
    check_duplicate: bool = True,
):
    if cell.status != 0 or i == j:
        return cell
    d2 = wp.dot(q, q)
    if d2 == q.dtype(0.0):
        # A deterministic owner for exactly coincident equal-weight sites.
        if wj > wi or (wj == wi and j < i):
            cell.status = 3
            cell.nv = 0
        return cell
    numerator = d2 + q.dtype(wi) - q.dtype(wj)
    if numerator > q.dtype(0.0) and numerator * numerator > q.dtype(4.0) * cell.r2 * d2:
        return cell
    if numerator < q.dtype(0.0) and numerator * numerator > q.dtype(4.0) * cell.r2 * d2:
        cell.status = 3
        cell.nv = 0
        return cell
    inv_length = q.dtype(1.0) / wp.sqrt(d2)
    plane = s.planes.dtype(-q[0], -q[1], -q[2], q.dtype(0.5) * numerator) * inv_length
    # Do not insert duplicate hint planes, including roundoff re-clips.
    if check_duplicate:
        for k in range(cell.np):
            if s.neighbors[cell.base + 32 * k] == j:
                return cell
    removed = int(0)
    k = int(0)
    while k < cell.nv:
        at = cell.base + 32 * k
        if outside(s.positions[at], plane):
            cell.nv -= 1
            last = cell.base + 32 * cell.nv
            v = s.vertices[at]
            s.vertices[at] = s.vertices[last]
            s.vertices[last] = v
            swap_positions(s, at, last)
            removed += 1
        else:
            k += 1
    if removed == 0:
        return cell
    if cell.nv == 0:
        cell.status = 3
        return cell
    if cell.np == s.capacity:
        # Cannot collect here: removed vertices still reference old planes.
        cell.status = 1
        return cell
    new_plane = cell.np
    s.planes[cell.base + 32 * new_plane] = plane
    s.neighbors[cell.base + 32 * new_plane] = j
    cell.np += 1
    for k in range(cell.np):
        s.next[cell.base + 32 * k] = -1

    first = int(-1)
    vertex = cell.nv
    iterations = int(0)
    while removed > 0:
        iterations += 1
        if iterations > s.capacity * s.capacity:
            cell.status = 5
            return cell
        v = s.vertices[cell.base + 32 * vertex]
        border = wp.vec3i(0)
        opposite = wp.vec3i(0)
        for e in range(3):
            a = plane_at(v, e)
            b = plane_at(v, (e + 1) % 3)
            border[e] = int(s.next[cell.base + 32 * a] != -1)
            opposite[e] = int(s.next[cell.base + 32 * b] == a)
        simple = bool(True)
        for e in range(3):
            if opposite[e] == 0 and opposite[(e + 1) % 3] == 0 and border[(e + 1) % 3] != 0:
                simple = False
        if opposite[0] == 0 and opposite[1] == 0 and opposite[2] == 0:
            if first == -1:
                for e in range(3):
                    s.next[cell.base + 32 * plane_at(v, e)] = plane_at(v, (e + 1) % 3)
                first = plane_at(v, 0)
            else:
                simple = False
        if not simple:
            vertex += 1
            if vertex == cell.nv + removed:
                vertex = cell.nv
        else:
            for e in range(3):
                if opposite[e] == 0:
                    s.next[cell.base + 32 * plane_at(v, e)] = plane_at(v, (e + 1) % 3)
            for e in range(3):
                if opposite[e] != 0 and opposite[(e + 1) % 3] != 0:
                    p = plane_at(v, (e + 1) % 3)
                    if first == p:
                        first = s.next[cell.base + 32 * p]
                    s.next[cell.base + 32 * p] = -1
            at = cell.base + 32 * vertex
            last = cell.base + 32 * (cell.nv + removed - 1)
            s.vertices[at] = s.vertices[last]
            s.vertices[last] = v
            removed -= 1
            vertex = cell.nv
    if first == -1:
        cell.status = 5
        return cell
    edge = first
    iterations = int(0)
    while True:
        if cell.nv >= s.capacity:
            cell.status = 2
            return cell
        following = s.next[cell.base + 32 * edge]
        if following < 0 or following >= cell.np or iterations >= s.capacity:
            cell.status = 5
            return cell
        v = encode(new_plane, edge, following)
        at = cell.base + 32 * cell.nv
        s.vertices[at] = v
        set_position(s, at, coordinates(s, cell.base, v))
        cell.nv += 1
        edge = following
        iterations += 1
        if edge == first:
            break
    cell = update_bounds(s, cell)
    if cell.np >= s.capacity * 85 // 100:
        cell = collect_planes(s, cell)
    return cell


@wp.func(inline=True)
def emit(
    s: Any,
    cell: Any,
    index: int,
    neighbors: wp.array2d[int],
    counts: wp.array[int],
    status: wp.array[int],
):
    status[index] = cell.status
    count = int(0)
    if cell.status == 0:
        for k in range(cell.np):
            s.next[cell.base + 32 * k] = 0
        for k in range(cell.nv):
            v = s.vertices[cell.base + 32 * k]
            for e in range(3):
                s.next[cell.base + 32 * plane_at(v, e)] = 1
        for k in range(cell.np):
            at = cell.base + 32 * k
            j = s.neighbors[at]
            if s.next[at] != 0 and j >= 0:
                if count >= neighbors.shape[1]:
                    status[index] = 1
                    counts[index] = 0
                    return
                neighbors[index, count] = j
                count += 1
    elif cell.status != 3:
        # Valid candidate IDs remain useful even if floating-point topology
        # failed. They seed a higher-precision retry, never a partial CSR row.
        for k in range(neighbors.shape[1]):
            j = int(-1)
            if k < cell.np:
                j = s.neighbors[cell.base + 32 * k]
            neighbors[index, k] = j
    counts[index] = count


@wp.kernel(enable_backward=False)
def clip_candidates(
    points: wp.array[wp.vec3],
    weights: wp.array[float],
    candidates: wp.array2d[int],
    lower: wp.vec3,
    upper: wp.vec3,
    scratch: Scratch,
    neighbors: wp.array2d[int],
    counts: wp.array[int],
    status: wp.array[int],
):
    i = wp.tid()
    p = points[i]
    cell = init_cell(scratch, i, p, lower, upper, Cell())
    for k in range(candidates.shape[1]):
        j = candidates[i, k]
        if j >= 0:
            cell = clip(scratch, cell, i, j, points[j] - p, weights[i], weights[j])
        if cell.status != 0:
            break
    emit(scratch, cell, i, neighbors, counts, status)


def allocate_scratch(n, capacity=64, device="cuda:0", robust=False):
    scratch = Scratch64() if robust else Scratch()
    vector = wp.vec4d if robust else wp.vec4
    scratch.capacity = capacity
    size = ((n + 31) // 32) * 32 * capacity
    scratch.planes = wp.empty(size, dtype=vector, device=device)
    scratch.positions = wp.empty(size, dtype=vector, device=device)
    if robust:
        scratch.cartesian = wp.empty(size, dtype=wp.vec3d, device=device)
    scratch.vertices = wp.empty(size, dtype=wp.uint32, device=device)
    scratch.next = wp.empty(size, dtype=int, device=device)
    scratch.neighbors = wp.empty(size, dtype=int, device=device)
    return scratch
