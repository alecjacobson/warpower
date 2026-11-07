"""Bounded, untrusted float32 wall candidates for private VTM reconstruction.

Wall transitions and artificial initial triangles derive from Wuheng Gao's
Local-DeWall f6292bb41215437bee4ce77a69baa785713d6140 (copyright 2026 Wuheng Gao,
MIT; see LICENSE-Local-DeWall). This implementation uses the current site BVH,
fixed Warp buffers and two kernels instead of lane shuffles. Approximate sphere
queries and finite artificial vertices may omit or propose invalid cells; every
proposal needs the original-coordinate symbolic seed filter and complete hull
certificate. No candidate status or closure flag is a geometric proof.
"""

import warp as wp

from ._bvh import Tree

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.func
def normalized_position(points: wp.array[wp.vec3], tree: Tree, i: int):
    lo, hi = wp.vec3(tree.lower[1]), wp.vec3(tree.upper[1])
    span = wp.max(hi[0] - lo[0], wp.max(hi[1] - lo[1], hi[2] - lo[2]))
    if i < points.shape[0]:
        return (wp.vec3(points[i]) - (lo + hi) * wp.float32(0.5)) / span
    j = i - points.shape[0]
    p = wp.vec3(0.0)
    if j == 0:
        p = wp.vec3(-1.0, -1.0, -1.0)
    elif j == 1:
        p = wp.vec3(1.0, -1.0, -1.0)
    elif j == 2:
        p = wp.vec3(0.0, 1.0, -1.0)
    elif j == 3:
        p = wp.vec3(-1.0, 1.0, 1.0)
    elif j == 4:
        p = wp.vec3(1.0, 1.0, 1.0)
    else:
        p = wp.vec3(0.0, -1.0, 1.0)
    return p * wp.float32(1024.0)


@wp.kernel
def normalize(points: wp.array[wp.vec3], tree: Tree, normalized: wp.array[wp.vec3]):
    i = wp.tid()
    normalized[i] = normalized_position(points, tree, i)


@wp.func
def position(points: wp.array[wp.vec3], tree: Tree, i: int):
    return points[i]


@wp.func
def greater(p: wp.vec3, q: wp.vec3):
    return q[0] > p[0] or (q[0] == p[0] and (q[1] > p[1] or (q[1] == p[1] and q[2] > p[2])))


@wp.func
def orient(a: wp.vec3, b: wp.vec3, c: wp.vec3, d: wp.vec3):
    return wp.dot(wp.cross(b - a, c - a), d - a)


@wp.func
def projected(a: wp.vec3, b: wp.vec3, c: wp.vec3, p: wp.vec3):
    x, y, z = a - p, b - p, c - p
    u, v, w = x[0] * y[1] - x[1] * y[0], y[0] * z[1] - y[1] * z[0], z[0] * x[1] - z[1] * x[0]
    return (u >= wp.float32(0.0) and v >= wp.float32(0.0) and w >= wp.float32(0.0)) or (
        u <= wp.float32(0.0) and v <= wp.float32(0.0) and w <= wp.float32(0.0)
    )


@wp.struct
class Search:
    point: int
    nodes: int
    failed: int


@wp.func
def search(points: wp.array[wp.vec3], tree: Tree, site: int, tri: wp.vec3i):
    out = Search()
    out.point = -1
    p = position(points, tree, site)
    a, b, c = (
        position(points, tree, tri[0]) - p,
        position(points, tree, tri[1]) - p,
        position(points, tree, tri[2]) - p,
    )
    determinant = wp.dot(a, wp.cross(b, c))
    if determinant == wp.float32(0.0) or not wp.isfinite(determinant):
        out.failed = 1
        return out
    center = (
        wp.dot(a, a) * wp.cross(b, c)
        + wp.dot(b, b) * wp.cross(c, a)
        + wp.dot(c, c) * wp.cross(a, b)
    ) / (wp.float32(2.0) * determinant)
    if not wp.isfinite(center[0]) or not wp.isfinite(center[1]) or not wp.isfinite(center[2]):
        out.failed = 1
        return out
    lo, hi = wp.vec3(tree.lower[1]), wp.vec3(tree.upper[1])
    span = wp.max(hi[0] - lo[0], wp.max(hi[1] - lo[1], hi[2] - lo[2]))
    origin = (lo + hi) * wp.float32(0.5)
    stack = wp.zeros(shape=64, dtype=int)
    stack[0] = 1
    size = int(1)
    while size > 0 and out.nodes < 2 * tree.n:
        size -= 1
        node = stack[size]
        out.nodes += 1
        qlo = (wp.vec3(tree.lower[node]) - origin) / span - p
        qhi = (wp.vec3(tree.upper[node]) - origin) / span - p
        # Untrusted interval overestimate for a candidate-only sphere query.
        closest = wp.max(qlo, wp.min(center, qhi))
        positive = wp.float32(2.0) * wp.dot(center, closest)
        squared = wp.dot(closest, closest)
        pad = wp.float32(1.0e-13) * (wp.abs(positive) + squared + wp.float32(1.0))
        if positive - squared < -pad:
            continue
        if tree.count[node] > 0:
            for k in range(tree.start[node], tree.start[node] + tree.count[node]):
                j = tree.ids[k]
                if j == site or j == tri[0] or j == tri[1] or j == tri[2] or tree.owners[j] != j:
                    continue
                q = position(points, tree, j) - p
                positive = wp.float32(2.0) * wp.dot(center, q)
                squared = wp.dot(q, q)
                pad = wp.float32(1.0e-13) * (wp.abs(positive) + squared + wp.float32(1.0))
                if positive - squared > pad:
                    out.point = j
                    return out
        else:
            if size + 2 > 64:
                out.failed = 1
                return out
            left, right = int(tree.left[node]), int(tree.right[node])
            left_center = (wp.vec3(tree.lower[left]) + wp.vec3(tree.upper[left])) * wp.float32(0.5)
            right_center = (wp.vec3(tree.lower[right]) + wp.vec3(tree.upper[right])) * wp.float32(
                0.5
            )
            realp = origin + span * points[site]
            if wp.length_sq(left_center - realp) > wp.length_sq(right_center - realp):
                temporary = int(left)
                left = int(right)
                right = temporary
            stack[size] = right
            stack[size + 1] = left
            size += 2
    if size != 0:
        out.failed = 1
    return out


@wp.kernel
def initialize(
    points: wp.array[wp.vec3],
    tree: Tree,
    seeds: wp.array[wp.vec3i],
    status: wp.array[int],
    costs: wp.array2d[int],
):
    lane = wp.tid()
    site, side = lane // 2, lane % 2
    status[lane] = 0
    costs[lane, 0] = 0
    costs[lane, 1] = 0
    tri = wp.vec3i(tree.n + 3 * side, tree.n + 3 * side + 1, tree.n + 3 * side + 2)
    if tree.owners[site] != site:
        status[lane] = 8
        seeds[lane] = tri
        return
    p = position(points, tree, site)
    completed = bool(False)
    for iteration in range(256):
        result = search(points, tree, site, tri)
        costs[lane, 0] += 1
        costs[lane, 1] += result.nodes
        if result.failed != 0:
            status[lane] = 1
            break
        if result.point < 0:
            completed = True
            break
        q = position(points, tree, result.point)
        if projected(q, position(points, tree, tri[1]), position(points, tree, tri[2]), p):
            tri[0] = result.point
        elif projected(position(points, tree, tri[0]), q, position(points, tree, tri[2]), p):
            tri[1] = result.point
        elif projected(position(points, tree, tri[0]), position(points, tree, tri[1]), q, p):
            tri[2] = result.point
        else:
            status[lane] = 2
            break
    if not completed:
        status[lane] = status[lane] | 4
    for k in range(3):
        a, b = tri[k], tri[(k + 1) % 3]
        if not greater(p, position(points, tree, a)) and greater(p, position(points, tree, b)):
            tri = wp.vec3i(a, b, tri[(k + 2) % 3])
            break
    if greater(p, position(points, tree, tri[0])) or not greater(p, position(points, tree, tri[1])):
        status[lane] = status[lane] | 16
    seeds[lane] = tri


@wp.func
def emit(
    points: wp.array[wp.vec3],
    tree: Tree,
    site: int,
    tri: wp.vec3i,
    tets: wp.array2d[int],
    count: wp.array[int],
):
    if (
        tri[0] < tree.n
        and tri[1] < tree.n
        and tri[2] < tree.n
        and greater(position(points, tree, site), position(points, tree, tri[2]))
    ):
        at = wp.atomic_add(count, 0, 1)
        if at < tets.shape[0]:
            tets[at, 0] = tri[0]
            tets[at, 1] = tri[1]
            tets[at, 2] = tri[2]
            tets[at, 3] = site


@wp.kernel
def walk(
    points: wp.array[wp.vec3],
    tree: Tree,
    seeds: wp.array[wp.vec3i],
    seed_status: wp.array[int],
    status: wp.array[int],
    costs: wp.array2d[int],
    tets: wp.array2d[int],
    count: wp.array[int],
):
    lane = wp.tid()
    site = lane // 2
    partner = lane ^ 1
    status[lane] = seed_status[lane] | seed_status[partner]
    costs[lane, 0] = 0
    costs[lane, 1] = 0
    if status[lane] != 0:
        return
    p = position(points, tree, site)
    tri = seeds[lane]
    aim = seeds[partner]
    old = int(aim[0])
    emit(points, tree, site, tri, tets, count)
    completed = bool(False)
    advance = bool(True)
    seen_a = wp.zeros(shape=64, dtype=int)
    seen_b = wp.zeros(shape=64, dtype=int)
    seen_count = int(0)
    for iteration in range(256):
        if advance:
            if greater(p, position(points, tree, tri[2])):
                tri[1] = tri[2]
            else:
                tri[0] = tri[2]
            tri[2] = old
            # Source orient3dfast uses the opposite sign of this repository.
            if orient(
                position(points, tree, tri[0]),
                position(points, tree, tri[1]),
                position(points, tree, tri[2]),
                p,
            ) <= wp.float32(0.0):
                tri[2] = aim[0]
                if orient(
                    position(points, tree, tri[0]),
                    position(points, tree, tri[1]),
                    position(points, tree, tri[2]),
                    p,
                ) <= wp.float32(0.0):
                    tri[2] = aim[1]
            old = tri[2]
        if (tri[0] == aim[0] or (tri[0] >= tree.n and aim[0] >= tree.n)) and (
            tri[1] == aim[1] or (tri[1] >= tree.n and aim[1] >= tree.n)
        ):
            completed = True
            break
        if orient(
            position(points, tree, tri[0]),
            position(points, tree, tri[1]),
            position(points, tree, tri[2]),
            p,
        ) <= wp.float32(0.0):
            status[lane] = 128
            break
        if advance:
            duplicate = bool(False)
            for k in range(seen_count):
                if seen_a[k] == tri[0] and seen_b[k] == tri[1]:
                    duplicate = True
            if duplicate or seen_count >= 64:
                status[lane] = 256
                break
            seen_a[seen_count] = tri[0]
            seen_b[seen_count] = tri[1]
            seen_count += 1
        result = search(points, tree, site, tri)
        costs[lane, 0] += 1
        costs[lane, 1] += result.nodes
        if result.failed != 0:
            status[lane] = 32
            break
        advance = result.point < 0
        if advance:
            emit(points, tree, site, tri, tets, count)
        else:
            old = tri[2]
            tri[2] = result.point
    if not completed:
        status[lane] = status[lane] | 64
