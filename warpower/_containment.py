"""Conservative exact containment for rare material-boundary recovery.

Input surface topology/contact validation and tet orientation validation are
prerequisites. Plane support plus a strict material witness prove that an entire
tet is in the closed material. Rejection is inconclusive, not proof of exterior.
All decisions use original float32 coordinates. Certified binary64 bounds
resolve unambiguous signs; uncertain signs retain the dyadic-integer fallback.
Calling kernels must disable fast math and floating-point contraction. For
finite binary32 inputs the degree-three products and error terms stay normal
in binary64, including when the original coordinates are subnormal.
"""

import warp as wp

from . import _filtered_predicates as predicates
from . import _integer as integer
from ._domain import next_edge
from ._surface_bvh import SurfaceTree

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.func(inline=False)
def exact_triangle(p: wp.vec3, q0: wp.vec3, q1: wp.vec3, q2: wp.vec3, direction: wp.vec3):
    base = wp.min(
        wp.min(integer.vector_base(p), integer.vector_base(direction)),
        wp.min(integer.vector_base(q0), wp.min(integer.vector_base(q1), integer.vector_base(q2))),
    )
    a, b, c = (
        integer.relative(q0, p, base),
        integer.relative(q1, p, base),
        integer.relative(q2, p, base),
    )
    d = integer.relative(direction, wp.vec3(), base)
    n = integer.det3(a, b, c)
    u, v, w = integer.det3(d, b, c), integer.det3(d, c, a), integer.det3(d, a, b)
    denominator = integer.plus(integer.plus(u, v), w)
    if n.valid == 0 or u.valid == 0 or v.valid == 0 or w.valid == 0 or denominator.valid == 0:
        return wp.vec2i(0, 2)
    return triangle_event(n.sign, u.sign, v.sign, w.sign, denominator.sign)


@wp.func
def triangle_event(n: int, u: int, v: int, w: int, denominator: int):
    if denominator == 0:
        if n == 0:
            return wp.vec2i(0, 1)
        return wp.vec2i(0, 0)
    if n == 0 or n != denominator:
        return wp.vec2i(0, 0)
    if (
        (u == 0 or u == denominator)
        and (v == 0 or v == denominator)
        and (w == 0 or w == denominator)
    ):
        if u == 0 or v == 0 or w == 0:
            return wp.vec2i(0, 1)
        return wp.vec2i(denominator, 0)
    return wp.vec2i(0, 0)


@wp.func
def triangle_bounds(p: wp.vec3, q0: wp.vec3, q1: wp.vec3, q2: wp.vec3, direction: wp.vec3):
    ax, ay, az = (
        predicates.difference(q0[0], p[0]),
        predicates.difference(q0[1], p[1]),
        predicates.difference(q0[2], p[2]),
    )
    bx, by, bz = (
        predicates.difference(q1[0], p[0]),
        predicates.difference(q1[1], p[1]),
        predicates.difference(q1[2], p[2]),
    )
    cx, cy, cz = (
        predicates.difference(q2[0], p[0]),
        predicates.difference(q2[1], p[1]),
        predicates.difference(q2[2], p[2]),
    )
    dx, dy, dz = (
        predicates.scalar(wp.float64(direction[0])),
        predicates.scalar(wp.float64(direction[1])),
        predicates.scalar(wp.float64(direction[2])),
    )
    n = predicates.det3(ax, ay, az, bx, by, bz, cx, cy, cz)
    u = predicates.det3(dx, dy, dz, bx, by, bz, cx, cy, cz)
    v = predicates.det3(dx, dy, dz, cx, cy, cz, ax, ay, az)
    w = predicates.det3(dx, dy, dz, ax, ay, az, bx, by, bz)
    denominator = predicates.plus(predicates.plus(u, v), w)
    return n, u, v, w, denominator


@wp.func(inline=False)
def triangle(p: wp.vec3, q0: wp.vec3, q1: wp.vec3, q2: wp.vec3, direction: wp.vec3):
    n, u, v, w, denominator = triangle_bounds(p, q0, q1, q2, direction)
    if (
        predicates.certain(n)
        and predicates.certain(u)
        and predicates.certain(v)
        and predicates.certain(w)
        and predicates.certain(denominator)
    ):
        return triangle_event(
            predicates.predicate(n).sign,
            predicates.predicate(u).sign,
            predicates.predicate(v).sign,
            predicates.predicate(w).sign,
            predicates.predicate(denominator).sign,
        )
    return exact_triangle(p, q0, q1, q2, direction)


@wp.func(inline=False)
def ray_count(tree: SurfaceTree, p: wp.vec3, direction: wp.vec3):
    # For a strict query, forward t = det(a,b,c) / dot(d, cross(b-a,c-a)).
    # u/v/w are its barycentric numerators. Only strict triangle-interior
    # crossings count; an edge hit or coplanar ray requests another direction.
    count = int(0)
    for f in range(tree.n):
        q0, q1, q2 = (
            tree.points[tree.indices[3 * f]],
            tree.points[tree.indices[3 * f + 1]],
            tree.points[tree.indices[3 * f + 2]],
        )
        result = triangle(p, q0, q1, q2, direction)
        if result[1] != 0:
            return result
        count += result[0]
    return wp.vec2i(count, 0)


@wp.func(inline=False)
def strict_material_side(tree: SurfaceTree, p: wp.vec3):
    # The caller proves p is off the entire surface. Ambiguity is inconclusive.
    d0, d1, d2 = wp.vec3(11.0, 23.0, 37.0), wp.vec3(-17.0, 29.0, 43.0), wp.vec3(31.0, -47.0, 19.0)
    r = ray_count(tree, p, d0)
    if r[1] == 1:
        r = ray_count(tree, p, d1)
    if r[1] == 1:
        r = ray_count(tree, p, d2)
    if r[1] != 0 or (r[0] != 0 and r[0] != 1):
        return 128
    return r[0]


@wp.func(inline=False)
def contained(tree: SurfaceTree, points: wp.array[wp.vec3], ids: wp.vec4i):
    lo, hi = wp.vec3(points[ids[0]]), wp.vec3(points[ids[0]])
    for k in range(1, 4):
        lo, hi = wp.min(lo, points[ids[k]]), wp.max(hi, points[ids[k]])
    # Overlapping triangle planes must support the whole tet on the material
    # side. Thus no input triangle can cross its interior. A strict vertex
    # witnesses which connected complement region contains that interior.
    strict = int(15)
    for f in range(tree.n):
        a, b, c = (
            tree.points[tree.indices[3 * f]],
            tree.points[tree.indices[3 * f + 1]],
            tree.points[tree.indices[3 * f + 2]],
        )
        qlo, qhi = wp.min(a, wp.min(b, c)), wp.max(a, wp.max(b, c))
        if (
            qhi[0] < lo[0]
            or qhi[1] < lo[1]
            or qhi[2] < lo[2]
            or qlo[0] > hi[0]
            or qlo[1] > hi[1]
            or qlo[2] > hi[2]
        ):
            continue
        for k in range(4):
            side = predicates.orient3d(a, b, c, points[ids[k]])
            if side.valid == 0 or side.sign > 0:
                return 0
            if side.sign == 0:
                strict &= ~(1 << k)
    for k in range(4):
        if (strict & (1 << k)) != 0:
            membership = strict_material_side(tree, points[ids[k]])
            if membership != 128:
                return int(membership == 1)
    return 0


@wp.func
def bad_edge(
    a: int,
    b: int,
    indices: wp.array[int],
    table: wp.array[int],
    occurrences: wp.array[int],
    balance: wp.array[int],
):
    lo, hi = wp.min(a, b), wp.max(a, b)
    slot = int(
        (wp.uint32(lo) * wp.uint32(73856093) ^ wp.uint32(hi) * wp.uint32(19349663))
        & wp.uint32(table.shape[0] - 1)
    )
    for _ in range(table.shape[0]):
        e = table[slot]
        if e < 0:
            return False
        if (
            wp.min(indices[e], indices[next_edge(e)]) == lo
            and wp.max(indices[e], indices[next_edge(e)]) == hi
        ):
            return occurrences[slot] != 2 or balance[slot] != 0
        slot = (slot + 1) & (table.shape[0] - 1)
    return False


@wp.kernel
def add_contained(
    tree: SurfaceTree,
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    count: wp.array[int],
    keep: wp.array[int],
    indices: wp.array[int],
    table: wp.array[int],
    occurrences: wp.array[int],
    balance: wp.array[int],
    additions: wp.array[int],
    total: wp.array[int],
):
    t = wp.tid()
    if t >= count[0] or keep[t] != 0:
        return
    ids = wp.vec4i(tets[t, 0], tets[t, 1], tets[t, 2], tets[t, 3])
    touches = bool(False)
    for i in range(4):
        for j in range(i + 1, 4):
            if bad_edge(ids[i], ids[j], indices, table, occurrences, balance):
                touches = True
    if touches and contained(tree, points, ids) != 0:
        keep[t] = 1
        wp.atomic_add(additions, 0, 1)
        wp.atomic_add(total, 0, 1)


@wp.kernel
def begin(
    error: wp.array[int],
    diagnostics: wp.array[int],
    rounds: wp.array[int],
    total: wp.array[int],
    ready: wp.array[int],
):
    rounds[0] = 0
    total[0] = 0
    ready[0] = int(error[0] == 0 and diagnostics[0] == 2)


@wp.kernel
def next_round(
    error: wp.array[int],
    diagnostics: wp.array[int],
    additions: wp.array[int],
    rounds: wp.array[int],
    ready: wp.array[int],
):
    rounds[0] += 1
    ready[0] = int(error[0] == 0 and diagnostics[0] == 2 and additions[0] > 0 and rounds[0] < 4)
