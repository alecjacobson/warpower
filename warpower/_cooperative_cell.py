"""Cooperative clipping with independent arithmetic and storage choices.

One 32-thread block owns a cell with 128, 256, or 512 vertex/plane slots. Tile operations
classify and compact vertices; the block reconstructs the clipping boundary
and checks that it is one cycle before emitting a row. Capacity or topology
failures retain their resource requirements for subsequent retries. FP32
decisions near the clipping plane request FP64 recovery. Adapted from the
Paragram clipping representation under Apache-2.0.
"""

from functools import lru_cache
from typing import Any

import warp as wp

from ._cell import intersect_planes_fp32

wp.set_module_options({"enable_backward": False})


@wp.struct
class Scratch:
    lane: int
    meta: wp.array2d[int]
    planes: wp.array[wp.vec4]
    vertices: wp.array[wp.uint32]
    positions: wp.array[wp.vec4]
    cartesian: wp.array[wp.vec3]
    next: wp.array[int]
    neighbors: wp.array[int]
    capacity: int


@wp.struct
class Cell:
    capacity: int
    base: int
    nv: int
    np: int
    status: int
    lo: wp.vec3
    hi: wp.vec3
    r2: float


@wp.struct
class Scratch64:
    lane: int
    meta: wp.array2d[int]
    planes: wp.array[wp.vec4d]
    vertices: wp.array[wp.uint32]
    positions: wp.array[wp.vec4d]
    cartesian: wp.array[wp.vec3d]
    next: wp.array[int]
    neighbors: wp.array[int]
    capacity: int


@wp.struct
class Cell64:
    capacity: int
    base: int
    nv: int
    np: int
    status: int
    lo: wp.vec3d
    hi: wp.vec3d
    r2: wp.float64


@wp.func(inline=True)
def set_position(s: Any, at: int, h: Any):  # noqa: F811
    s.positions[at] = h
    inverse_w = h.dtype(1.0) / h[3]
    s.cartesian[at] = s.cartesian.dtype(h[0], h[1], h[2]) * inverse_w


@wp.func(inline=True)
def position(s: Any, at: int):  # noqa: F811
    return s.cartesian[at]


@wp.func(inline=True)
def encode(a: int, b: int, c: int):
    return wp.uint32(a) | (wp.uint32(b) << wp.uint32(10)) | (wp.uint32(c) << wp.uint32(20))


@wp.func(inline=True)
def plane_at(v: wp.uint32, i: int):
    return int((v >> wp.uint32(10 * i)) & wp.uint32(1023))


@wp.func(inline=True)
def coordinates(s: Any, base: int, v: wp.uint32):
    a = s.planes[base + plane_at(v, 0)]
    b = s.planes[base + plane_at(v, 1)]
    c = s.planes[base + plane_at(v, 2)]
    bc0 = b[1] * c[2] - b[2] * c[1]
    bc1 = b[2] * c[0] - b[0] * c[2]
    bc2 = b[0] * c[1] - b[1] * c[0]
    x0 = -a[3] * bc0 - b[3] * (c[1] * a[2] - c[2] * a[1]) - c[3] * (a[1] * b[2] - a[2] * b[1])
    x1 = -a[3] * bc1 - b[3] * (c[2] * a[0] - c[0] * a[2]) - c[3] * (a[2] * b[0] - a[0] * b[2])
    x2 = -a[3] * bc2 - b[3] * (c[0] * a[1] - c[1] * a[0]) - c[3] * (a[0] * b[1] - a[1] * b[0])
    return type(a)(x0, x1, x2, a[0] * bc0 + a[1] * bc1 + a[2] * bc2)


@wp.func(inline=True)
def coordinates(s: Scratch, base: int, v: wp.uint32):  # noqa: F811
    return intersect_planes_fp32(
        s.planes[base + plane_at(v, 0)],
        s.planes[base + plane_at(v, 1)],
        s.planes[base + plane_at(v, 2)],
    )


@wp.func(inline=True)
def collect_planes(s: Any, cell: Any):
    for i in range(cell.np):
        s.next[cell.base + i] = -1
    for i in range(cell.nv):
        v = s.vertices[cell.base + i]
        for j in range(3):
            s.next[cell.base + plane_at(v, j)] = 0
    count = int(0)
    for i in range(cell.np):
        at = cell.base + i
        if s.next[at] == 0:
            to = cell.base + count
            s.planes[to] = s.planes[at]
            s.neighbors[to] = s.neighbors[at]
            s.next[at] = count
            count += 1
    for i in range(cell.nv):
        at = cell.base + i
        v = s.vertices[at]
        a = s.next[cell.base + plane_at(v, 0)]
        b = s.next[cell.base + plane_at(v, 1)]
        c = s.next[cell.base + plane_at(v, 2)]
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
            s.next[cell.base + k] = 0
        for k in range(cell.nv):
            v = s.vertices[cell.base + k]
            for e in range(3):
                s.next[cell.base + plane_at(v, e)] = 1
        for k in range(cell.np):
            at = cell.base + k
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
                j = s.neighbors[cell.base + k]
            neighbors[index, k] = j
    counts[index] = count


def allocate_scratch(n, capacity=128, device="cuda:0", robust=True):
    """Allocate one independent 32-lane clipping workspace per cell."""
    if capacity not in (128, 256, 512):
        raise ValueError("cooperative capacity must be 128, 256, or 512")
    scratch = Scratch64() if robust else Scratch()
    vector = wp.vec4d if robust else wp.vec4
    cartesian = wp.vec3d if robust else wp.vec3
    scratch.capacity = capacity
    size = n * capacity
    scratch.planes = wp.empty(size, dtype=vector, device=device)
    scratch.positions = wp.empty(size, dtype=vector, device=device)
    scratch.cartesian = wp.empty(size, dtype=cartesian, device=device)
    scratch.vertices = wp.empty(size, dtype=wp.uint32, device=device)
    scratch.next = wp.empty(size, dtype=int, device=device)
    scratch.neighbors = wp.empty(size, dtype=int, device=device)
    scratch.meta = wp.empty((n, 3), dtype=int, device=device)
    return scratch


@wp.func
def barrier(value: int):
    shared = wp.tile_from_thread(shape=(1,), value=value, thread_idx=0, storage="shared")
    return wp.tile_extract(shared, 0)


@wp.func
def init_cell(s: Any, slot: int, seed: wp.vec3, lower: wp.vec3, upper: wp.vec3, cell: Any):
    lane = s.lane
    cell.base = slot * s.capacity
    cell.capacity = s.capacity
    cell.nv = 8
    cell.np = 6
    cell.status = 0
    cell.lo = s.cartesian.dtype(lower) - s.cartesian.dtype(seed)
    cell.hi = s.cartesian.dtype(upper) - s.cartesian.dtype(seed)
    cell.r2 = cell.lo.dtype(0.0)
    for axis in range(3):
        d = wp.max(wp.abs(cell.lo[axis]), wp.abs(cell.hi[axis]))
        cell.r2 += d * d
    if lane < 3:
        low = s.planes.dtype(cell.lo.dtype(0.0))
        high = s.planes.dtype(cell.lo.dtype(0.0))
        low[lane] = cell.lo.dtype(1.0)
        low[3] = -cell.lo[lane]
        high[lane] = cell.lo.dtype(-1.0)
        high[3] = cell.hi[lane]
        s.planes[cell.base + 2 * lane] = low
        s.planes[cell.base + 2 * lane + 1] = high
    # Tiles read all capacity slots; initialize inactive entries as well.
    for k in range(lane, s.capacity, 32):
        s.neighbors[cell.base + k] = -1
        s.vertices[cell.base + k] = wp.uint32(0)
        s.positions[cell.base + k] = s.planes.dtype(cell.lo.dtype(0.0))
        s.cartesian[cell.base + k] = s.cartesian.dtype(cell.lo.dtype(0.0))
    cell.status = barrier(0)
    if lane < 8:
        v = wp.uint32(0)
        if lane == 0:
            v = encode(2, 5, 0)
        elif lane == 1:
            v = encode(5, 3, 0)
        elif lane == 2:
            v = encode(1, 5, 2)
        elif lane == 3:
            v = encode(5, 1, 3)
        elif lane == 4:
            v = encode(4, 2, 0)
        elif lane == 5:
            v = encode(4, 0, 3)
        elif lane == 6:
            v = encode(2, 4, 1)
        else:
            v = encode(4, 3, 1)
        s.vertices[cell.base + lane] = v
        set_position(s, cell.base + lane, coordinates(s, cell.base, v))
    cell.status = barrier(0)
    return cell


@wp.func
def duplicate_flag(k: int, neighbor: int, np: int, candidate: int):
    return int(k < np and neighbor == candidate)


@wp.func
def removed_flag(k: int, h: Any, nv: int, plane: Any):
    return int(k < nv and outside(h, plane))


@wp.func
def partition_index(k: int, removed: int, prefix: int, nv: int, count: int, base: int):
    if k >= nv:
        return -1
    if removed != 0:
        return base + nv - count + prefix - 1
    return base + k - prefix


@wp.func
def lower_position(k: int, p: Any, nv: int, axis: int):
    if k < nv:
        return p[axis]
    return p.dtype(1.0e30)


@wp.func
def upper_position(k: int, p: Any, nv: int, axis: int):
    if k < nv:
        return p[axis]
    return p.dtype(-1.0e30)


@wp.func
def squared_position(k: int, p: Any, nv: int):
    if k < nv:
        return wp.dot(p, p)
    return p.dtype(0.0)


@wp.func
def classify_cartesian(k: int, position: wp.vec3, nv: int, plane: wp.vec4):
    if k >= nv:
        return 0
    terms = wp.cw_mul(position, wp.vec3(plane[0], plane[1], plane[2]))
    value = terms[0] + terms[1] + terms[2] + plane[3]
    scale = wp.abs(terms[0]) + wp.abs(terms[1]) + wp.abs(terms[2]) + wp.abs(plane[3])
    if wp.abs(value) <= 1.0e-6 * scale:
        return 2
    return int(value < 0.0)


@lru_cache(maxsize=None)
def make_clip(capacity=128, robust=True):
    tile_capacity = wp.constant(capacity)

    @wp.func
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
        lane = s.lane
        slot = cell.base // s.capacity
        d2 = wp.dot(q, q)
        if d2 == q.dtype(0.0):
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
        indices = wp.tile_arange(tile_capacity, dtype=int)
        if check_duplicate:
            neighbors = wp.tile_load(s.neighbors, shape=tile_capacity, offset=cell.base)
            matches = wp.tile_map(duplicate_flag, indices, neighbors, cell.np, j)
            duplicate = wp.tile_extract(wp.tile_max(matches), 0)
            if duplicate != 0:
                return cell
        if wp.static(robust):
            plane = s.planes.dtype(-q[0], -q[1], -q[2], q.dtype(0.5) * numerator) / wp.sqrt(d2)
        else:
            plane = s.planes.dtype(-q[0], -q[1], -q[2], q.dtype(0.5) * numerator) * (
                q.dtype(1.0) / wp.sqrt(d2)
            )
        positions = wp.tile_load(s.positions, shape=tile_capacity, offset=cell.base)
        vertices = wp.tile_load(s.vertices, shape=tile_capacity, offset=cell.base)
        cartesian = wp.tile_load(s.cartesian, shape=tile_capacity, offset=cell.base)
        if wp.static(not robust):
            removed = wp.tile_map(classify_cartesian, indices, cartesian, cell.nv, plane)
            if wp.tile_extract(wp.tile_max(removed), 0) == 2:
                cell.status = 6
                return cell
        else:
            removed = wp.tile_map(removed_flag, indices, positions, cell.nv, plane)
        scan = wp.tile_scan_inclusive(removed)
        count = wp.tile_extract(scan, tile_capacity - 1)
        if count == 0:
            return cell
        if count == cell.nv:
            cell.status = 3
            cell.nv = 0
            return cell
        if cell.np == cell.capacity:
            if cell.capacity == s.capacity:
                cell.status = 1
                return cell
            cell.capacity = wp.min(s.capacity, 2 * cell.capacity)
        destinations = wp.tile_map(
            partition_index, indices, removed, scan, cell.nv, count, cell.base
        )
        wp.tile_store_indexed(s.positions, destinations, positions, axis=0, offset=(0,))
        wp.tile_store_indexed(s.vertices, destinations, vertices, axis=0, offset=(0,))
        wp.tile_store_indexed(s.cartesian, destinations, cartesian, axis=0, offset=(0,))
        cell.nv -= count
        new_plane = cell.np
        cell.np += 1
        if lane == 0:
            s.planes[cell.base + new_plane] = plane
            s.neighbors[cell.base + new_plane] = j
            s.meta[slot, 0] = 0
            s.meta[slot, 1] = 0
            s.meta[slot, 2] = s.capacity
        for k in range(lane, cell.np, 32):
            s.next[cell.base + k] = -1
        cell.status = barrier(0)
        triangles = wp.tile_load(
            s.vertices, shape=tile_capacity, offset=cell.base, storage="shared"
        )
        for edge_id in range(lane, 3 * count, 32):
            triangle_id = cell.nv + edge_id // 3
            v = wp.tile_extract(triangles, triangle_id)
            edge = edge_id % 3
            a = plane_at(v, edge)
            b = plane_at(v, (edge + 1) % 3)
            opposite = int(0)
            for other in range(cell.nv, cell.nv + count):
                if other != triangle_id:
                    u = wp.tile_extract(triangles, other)
                    ua, ub, uc = plane_at(u, 0), plane_at(u, 1), plane_at(u, 2)
                    if (a == ua or a == ub or a == uc) and (b == ua or b == ub or b == uc):
                        opposite += 1
            if opposite == 0:
                previous = wp.atomic_cas(s.next, cell.base + a, -1, b)
                if previous != -1:
                    wp.atomic_max(s.meta, slot, 1, 5)
                wp.atomic_min(s.meta, slot, 2, a)
                at = cell.nv + wp.atomic_add(s.meta, slot, 0, 1)
                if at >= s.capacity:
                    wp.atomic_max(s.meta, slot, 1, 2)
                else:
                    new_vertex = encode(new_plane, a, b)
                    s.vertices[cell.base + at] = new_vertex
                    homogeneous = coordinates(s, cell.base, new_vertex)
                    if homogeneous[3] == q.dtype(0.0):
                        wp.atomic_max(s.meta, slot, 1, 5)
                    else:
                        xyz = s.cartesian.dtype(homogeneous[0], homogeneous[1], homogeneous[2]) * (
                            q.dtype(1.0) / homogeneous[3]
                        )
                        if not (
                            wp.isfinite(xyz[0]) and wp.isfinite(xyz[1]) and wp.isfinite(xyz[2])
                        ):
                            wp.atomic_max(s.meta, slot, 1, 5)
                        s.positions[cell.base + at] = homogeneous
                        s.cartesian[cell.base + at] = xyz
            elif opposite != 1:
                wp.atomic_max(s.meta, slot, 1, 5)
        cell.status = barrier(0)
        created = s.meta[slot, 0]
        error = s.meta[slot, 1]
        # Queue retries can start below the allocated tile capacity. Grow the
        # logical clipping limit only when needed, reusing the current cell.
        while cell.nv + created > cell.capacity and cell.capacity < s.capacity:
            cell.capacity = wp.min(s.capacity, 2 * cell.capacity)
        if lane == 0 and error == 0:
            first = s.meta[slot, 2]
            cursor = first
            if created < 3 or first >= cell.np:
                error = 5
            else:
                for k in range(created):
                    if cursor < 0 or cursor >= cell.np:
                        error = 5
                        break
                    cursor = s.next[cell.base + cursor]
                    if cursor == first and k != created - 1:
                        error = 5
                if cursor != first:
                    error = 5
        cell.status = barrier(error)
        if cell.status != 0:
            return cell
        cell.nv += created
        new_positions = wp.tile_load(s.cartesian, shape=tile_capacity, offset=cell.base)
        for axis in range(3):
            lower = wp.tile_map(lower_position, indices, new_positions, cell.nv, axis)
            upper = wp.tile_map(upper_position, indices, new_positions, cell.nv, axis)
            cell.lo[axis] = wp.tile_extract(wp.tile_min(lower), 0)
            cell.hi[axis] = wp.tile_extract(wp.tile_max(upper), 0)
        radius = wp.tile_map(squared_position, indices, new_positions, cell.nv)
        cell.r2 = wp.tile_extract(wp.tile_max(radius), 0)
        if cell.np >= cell.capacity * 85 // 100:
            if lane == 0:
                cell = collect_planes(s, cell)
            cell.np = barrier(cell.np)
        return cell

    return clip
