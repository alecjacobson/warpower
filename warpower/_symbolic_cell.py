"""Combinatorial Voronoi clipping with original-coordinate symbolic predicates.

Topology stores only incident plane IDs. Optional conservative bounds on the
unweighted vertex limits reject strictly redundant geometric tests; uncertain
cases retain the original-coordinate symbolic predicates.
"""

import warp as wp

from . import _filtered_predicates as fp
from ._cell import Scratch64, encode, plane_at
from ._delaunay_certificate import symbolic_sphere
from ._halfspace_predicates import classify

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.struct
class SymbolicCell:
    base: int
    nv: int
    np: int
    status: int
    lo: wp.vec3d
    hi: wp.vec3d
    changed: int


@wp.func(inline=True)
def init_symbolic(s: Scratch64, slot: int):
    cell = SymbolicCell()
    cell.base = (slot // 32) * (32 * s.capacity) + slot % 32
    cell.nv = 8
    cell.np = 6
    cell.status = 0
    for k in range(6):
        s.neighbors[cell.base + 32 * k] = -(k + 1)
    s.vertices[cell.base + 32 * 0] = encode(2, 5, 0)
    s.vertices[cell.base + 32 * 1] = encode(5, 3, 0)
    s.vertices[cell.base + 32 * 2] = encode(1, 5, 2)
    s.vertices[cell.base + 32 * 3] = encode(5, 1, 3)
    s.vertices[cell.base + 32 * 4] = encode(4, 2, 0)
    s.vertices[cell.base + 32 * 5] = encode(4, 0, 3)
    s.vertices[cell.base + 32 * 6] = encode(2, 4, 1)
    s.vertices[cell.base + 32 * 7] = encode(4, 3, 1)
    return cell


@wp.func(inline=True)
def collect_symbolic(s: Scratch64, cell: SymbolicCell):
    for k in range(cell.np):
        s.next[cell.base + 32 * k] = -1
    for k in range(cell.nv):
        v = s.vertices[cell.base + 32 * k]
        for j in range(3):
            s.next[cell.base + 32 * plane_at(v, j)] = 0
    count = int(0)
    for k in range(cell.np):
        at = cell.base + 32 * k
        if s.next[at] == 0:
            s.neighbors[cell.base + 32 * count] = s.neighbors[at]
            s.next[at] = count
            count += 1
    for k in range(cell.nv):
        at = cell.base + 32 * k
        v = s.vertices[at]
        a = s.next[cell.base + 32 * plane_at(v, 0)]
        b = s.next[cell.base + 32 * plane_at(v, 1)]
        c = s.next[cell.base + 32 * plane_at(v, 2)]
        s.vertices[at] = encode(a, b, c)
    cell.np = count
    return cell


@wp.func(inline=False)
def site_relation(points: wp.array[wp.vec3], i: int, ia: int, ib: int, ic: int, j: int):
    orientation = fp.orient3d(points[i], points[ia], points[ib], points[ic])
    relation = symbolic_sphere(points, wp.vec4i(i, ia, ib, ic), j)
    relation.sign *= -orientation.sign
    relation.valid = int(relation.valid != 0 and orientation.valid != 0 and orientation.sign != 0)
    return relation


@wp.func(inline=False)
def clip_symbolic(
    s: Scratch64,
    cell: SymbolicCell,
    i: int,
    j: int,
    points: wp.array[wp.vec3],
    bounds: wp.array[wp.vec3],
):
    cell.changed = 0
    if cell.status != 0 or i == j:
        return cell
    removed = int(0)
    k = int(0)
    while k < cell.nv:
        at = cell.base + 32 * k
        planes = s.vertices[at]
        ia = s.neighbors[cell.base + 32 * plane_at(planes, 0)]
        ib = s.neighbors[cell.base + 32 * plane_at(planes, 1)]
        ic = s.neighbors[cell.base + 32 * plane_at(planes, 2)]
        relation = fp.predicate(fp.scalar(wp.float64(0.0)))
        if ia >= 0 and ib >= 0 and ic >= 0:
            relation = site_relation(points, i, ia, ib, ic, j)
        else:
            relation = classify(points, bounds, i, wp.vec4i(ia, ib, ic, j))
        if relation.valid == 0:
            cell.status = 6
            return cell
        if relation.sign < 0:
            cell.nv -= 1
            last = cell.base + 32 * cell.nv
            v = s.vertices[at]
            s.vertices[at] = s.vertices[last]
            s.vertices[last] = v
            removed += 1
        else:
            k += 1
    if removed == 0:
        return cell
    cell.changed = 1
    if cell.nv == 0:
        cell.status = 3
        return cell
    if cell.np == s.capacity:
        # Cannot collect here: removed vertices still reference old planes.
        cell.status = 1
        return cell
    new_plane = cell.np
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
        cell.nv += 1
        edge = following
        iterations += 1
        if edge == first:
            break
    if cell.np >= s.capacity * 85 // 100:
        cell = collect_symbolic(s, cell)
    return cell
