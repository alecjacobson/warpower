"""Exact original-coordinate certificate for optional Delaunay recovery.

Each accepted positive-volume cell must be a strict lower facet after symbolic
lifting perturbation. This establishes nonoverlap even for cospherical inputs;
paired interior faces and supporting hull faces then establish completeness.
"""

import warp as wp

from . import _filtered_predicates as fp
from . import _sphere_bounds as sb
from ._bvh import Tree
from ._canonical import mix
from ._domain import _Scan
from ._mesh_topology import face_vertices, identity, index_faces, sorted_face

wp.set_module_options(
    {"enable_backward": False, "fast_math": False, "fuse_fp": False, "block_dim": 32}
)
Rows5 = wp.types.vector(5, wp.int32)
HASH_SCALE = wp.constant(2.0**-23)


@wp.func
def direction(owner: int):
    return float(mix(wp.uint32(owner + 1)) & wp.uint32(0xFFFFFF)) * HASH_SCALE - 1.0


@wp.func(inline=True)
def symbolic_sphere(points: wp.array[wp.vec3], ids: wp.vec4i, query: int):
    a, b, c, d, q = points[ids[0]], points[ids[1]], points[ids[2]], points[ids[3]], points[query]
    result = fp.insphere(a, b, c, d, q)
    if result.valid == 0 or result.sign != 0:
        return result
    # The power determinant is affine in its weights. At an exact unweighted
    # tie, this is precisely the coefficient of the first infinitesimal.
    result = fp.power_sphere(
        a,
        b,
        c,
        d,
        q,
        direction(ids[0]),
        direction(ids[1]),
        direction(ids[2]),
        direction(ids[3]),
        direction(query),
    )
    if result.valid == 0 or result.sign != 0:
        return result
    # Remaining ties use successively smaller positive weights in site-ID order.
    # Cofactor of weight i in the negative xyz/lift/1 determinant is (-1)^i
    # times orient3d of the other four rows in their original order.
    rows = Rows5(ids[0], ids[1], ids[2], ids[3], query)
    last = int(-1)
    for _ in range(5):
        best, row = int(2147483647), int(-1)
        for k in range(5):
            if rows[k] > last and rows[k] < best:
                best, row = rows[k], k
        others = wp.vec4i(0)
        at = int(0)
        for k in range(5):
            if k != row:
                others[at] = rows[k]
                at += 1
        result = fp.orient3d(
            points[others[0]], points[others[1]], points[others[2]], points[others[3]]
        )
        if result.valid == 0:
            return result
        if result.sign != 0:
            result.sign *= 1 - 2 * (row % 2)
            result.estimate *= wp.float64(1 - 2 * (row % 2))
            return result
        last = best
    result.valid = 0  # Impossible for a positive-volume defining tetrahedron.
    return result


@wp.struct
class Sphere:
    origin: wp.vec3
    nx: fp.Ball
    ny: fp.Ball
    nz: fp.Ball
    determinant: fp.Ball


@wp.func
def cofactor(
    u: fp.Ball,
    v: fp.Ball,
    w: fp.Ball,
    uy: fp.Ball,
    uz: fp.Ball,
    vy: fp.Ball,
    vz: fp.Ball,
    wy: fp.Ball,
    wz: fp.Ball,
):
    return fp.plus(
        fp.plus(fp.times(u, fp.cross2(vy, vz, wy, wz)), fp.times(v, fp.cross2(wy, wz, uy, uz))),
        fp.times(w, fp.cross2(uy, uz, vy, vz)),
    )


@wp.func
def sphere(a: wp.vec3, b: wp.vec3, c: wp.vec3, d: wp.vec3):
    ux, uy, uz = fp.difference(b[0], a[0]), fp.difference(b[1], a[1]), fp.difference(b[2], a[2])
    vx, vy, vz = fp.difference(c[0], a[0]), fp.difference(c[1], a[1]), fp.difference(c[2], a[2])
    wx, wy, wz = fp.difference(d[0], a[0]), fp.difference(d[1], a[1]), fp.difference(d[2], a[2])
    u, v, w = (
        fp.lift(ux, uy, uz, 0.0, 0.0),
        fp.lift(vx, vy, vz, 0.0, 0.0),
        fp.lift(wx, wy, wz, 0.0, 0.0),
    )
    result = Sphere()
    result.origin = a
    result.nx = cofactor(u, v, w, uy, uz, vy, vz, wy, wz)
    result.ny = cofactor(u, v, w, uz, ux, vz, vx, wz, wx)
    result.nz = cofactor(u, v, w, ux, uy, vx, vy, wx, wy)
    result.determinant = fp.det3(ux, uy, uz, vx, vy, vz, wx, wy, wz)
    return result


@wp.func
def interval(lo: float, hi: float):
    result = fp.Ball()
    a, b = wp.float64(lo), wp.float64(hi)
    result.value = wp.float64(0.5) * (a + b)
    result.error = (wp.float64(0.5) * (b - a) + fp.EPS * (wp.abs(a) + wp.abs(b))) * fp.ADD_PAD
    return result


@wp.func
def outside(value: Sphere, lo: wp.vec3, hi: wp.vec3):
    x = fp.minus(interval(lo[0], hi[0]), fp.scalar(wp.float64(value.origin[0])))
    y = fp.minus(interval(lo[1], hi[1]), fp.scalar(wp.float64(value.origin[1])))
    z = fp.minus(interval(lo[2], hi[2]), fp.scalar(wp.float64(value.origin[2])))
    result = fp.minus(
        fp.plus(fp.plus(fp.times(x, value.nx), fp.times(y, value.ny)), fp.times(z, value.nz)),
        fp.times(value.determinant, fp.lift(x, y, z, 0.0, 0.0)),
    )
    return wp.isfinite(result.value) and wp.isfinite(result.error) and result.value < -result.error


@wp.kernel
def select_positive(
    points: wp.array[wp.vec3],
    input_tets: wp.array2d[int],
    count: wp.array[int],
    owners: wp.array[int],
    selected: wp.array[int],
    error: wp.array[int],
):
    i = wp.tid()
    selected[i] = 0
    if count[0] <= 0 or count[0] > input_tets.shape[0]:
        wp.atomic_or(error, 0, 1)
        return
    if i >= count[0]:
        return
    ids = wp.vec4i(0)
    for k in range(4):
        v = input_tets[i, k]
        if v < 0 or v >= points.shape[0]:
            wp.atomic_or(error, 0, 1)
            return
        ids[k] = owners[v]
        for axis in range(3):
            if not wp.isfinite(points[v][axis]):
                wp.atomic_or(error, 0, 1)
                return
        for j in range(k):
            if ids[j] == ids[k]:
                wp.atomic_or(error, 0, 1)
                return
    sign = fp.orient3d(points[ids[0]], points[ids[1]], points[ids[2]], points[ids[3]])
    if sign.valid == 0:
        wp.atomic_or(error, 0, 2)
    elif sign.sign < 0:
        wp.atomic_or(error, 0, 4)
    elif sign.sign > 0:
        selected[i] = 1


@wp.kernel
def compact(
    input_tets: wp.array2d[int],
    selected: wp.array[int],
    prefix: wp.array[int],
    owners: wp.array[int],
    output: wp.array2d[int],
    count: wp.array[int],
    used: wp.array[int],
):
    i = wp.tid()
    if i == selected.shape[0] - 1:
        count[0] = prefix[i]
    if selected[i] != 0:
        for k in range(4):
            v = owners[input_tets[i, k]]
            output[prefix[i] - 1, k] = v
            wp.atomic_or(used, v, 1)


@wp.kernel
def ready(error: wp.array[int], condition: wp.array[int]):
    condition[0] = int(error[0] == 0)


@wp.kernel
def check_spheres(
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    count: wp.array[int],
    tree: Tree,
    error: wp.array[int],
    statistics: wp.array[int],
):
    i = wp.tid()
    if i >= count[0]:
        return
    ids = wp.vec4i(tets[i, 0], tets[i, 1], tets[i, 2], tets[i, 3])
    shape = sphere(points[ids[0]], points[ids[1]], points[ids[2]], points[ids[3]])
    box = sb.bounds(shape.origin, shape.nx, shape.ny, shape.nz, shape.determinant)
    stack = wp.zeros(shape=128, dtype=int)
    size = int(1)
    stack[0] = 1
    queries, visits = int(0), int(0)
    while size > 0:
        size -= 1
        node = stack[size]
        visits += 1
        if sb.outside(box, tree.lower[node], tree.upper[node]):
            continue
        if outside(shape, tree.lower[node], tree.upper[node]):
            continue
        if tree.count[node] > 0:
            for k in range(tree.start[node], tree.start[node] + tree.count[node]):
                q = tree.ids[k]
                if tree.owners[q] != q or q == ids[0] or q == ids[1] or q == ids[2] or q == ids[3]:
                    continue
                queries += 1
                # The node's interval can straddle the sphere while an individual
                # leaf point is provably outside. Reuse its cached cofactors;
                # uncertain signs and exact ties retain the symbolic predicate.
                if points.shape[0] >= 32 and outside(shape, points[q], points[q]):
                    continue
                result = symbolic_sphere(points, ids, q)
                if result.valid == 0:
                    wp.atomic_or(error, 0, 2)
                elif result.sign >= 0:
                    wp.atomic_or(error, 0, 8)
        else:
            if size > 125:
                wp.atomic_or(error, 0, 16)
                return
            stack[size] = tree.left[node]
            stack[size + 1] = tree.right[node]
            size += 2
    wp.atomic_add(statistics, 0, queries)
    wp.atomic_add(statistics, 1, visits)


@wp.struct
class Halfspace:
    origin: wp.vec3
    nx: fp.Ball
    ny: fp.Ball
    nz: fp.Ball


@wp.func
def halfspace(a: wp.vec3, b: wp.vec3, c: wp.vec3):
    ux, uy, uz = fp.difference(b[0], a[0]), fp.difference(b[1], a[1]), fp.difference(b[2], a[2])
    vx, vy, vz = fp.difference(c[0], a[0]), fp.difference(c[1], a[1]), fp.difference(c[2], a[2])
    result = Halfspace()
    result.origin = a
    result.nx = fp.cross2(uy, uz, vy, vz)
    result.ny = fp.cross2(uz, ux, vz, vx)
    result.nz = fp.cross2(ux, uy, vx, vy)
    return result


@wp.func
def inside_halfspace(plane: Halfspace, lo: wp.vec3, hi: wp.vec3):
    x = fp.minus(interval(lo[0], hi[0]), fp.scalar(wp.float64(plane.origin[0])))
    y = fp.minus(interval(lo[1], hi[1]), fp.scalar(wp.float64(plane.origin[1])))
    z = fp.minus(interval(lo[2], hi[2]), fp.scalar(wp.float64(plane.origin[2])))
    value = fp.plus(fp.plus(fp.times(plane.nx, x), fp.times(plane.ny, y)), fp.times(plane.nz, z))
    return wp.isfinite(value.value) and wp.isfinite(value.error) and value.value <= -value.error


@wp.kernel
def collect_boundary_faces(
    table: wp.array[int],
    occurrences: wp.array[int],
    balance: wp.array[int],
    boundary: wp.array[int],
    count: wp.array[int],
    error: wp.array[int],
):
    slot = wp.tid()
    n = occurrences[slot]
    if n == 0:
        return
    if n == 2 and balance[slot] == 0:
        return
    if n != 1:
        wp.atomic_or(error, 0, 32)
        return
    # There are at most four occupied face slots per input tetrahedron.
    at = wp.atomic_add(count, 0, 1)
    boundary[at] = table[slot]


@wp.kernel
def check_faces(
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    boundary: wp.array[int],
    count: wp.array[int],
    tree: Tree,
    error: wp.array[int],
    depth: int,
    workers: int,
):
    tid = wp.tid()
    pieces = 1 << depth
    worker, lane = tid // pieces, tid % pieces
    # Partition the radix tree by path bits. An early leaf belongs only to
    # the lane whose unused bits are zero, so every subtree is covered once.
    root = int(1)
    live = bool(True)
    for level in range(depth):
        if tree.count[root] > 0:
            live = (lane >> level) == 0
            break
        if ((lane >> level) & 1) == 0:
            root = tree.left[root]
        else:
            root = tree.right[root]
    if not live:
        return
    # Fixed launch size, device count, and a strided tail cover even malformed
    # candidates exposing all four faces per tetrahedron.
    for slot in range(worker, count[0], workers):
        ids = face_vertices(tets, boundary[slot])
        a, b, c = points[ids[0]], points[ids[1]], points[ids[2]]
        plane = halfspace(a, b, c)
        stack = wp.zeros(shape=128, dtype=int)
        stack[0] = root
        size = int(1)
        while size > 0:
            size -= 1
            node = stack[size]
            if inside_halfspace(plane, tree.lower[node], tree.upper[node]):
                continue
            if tree.count[node] > 0:
                for k in range(tree.start[node], tree.start[node] + tree.count[node]):
                    q = tree.ids[k]
                    # Coincident sites have exactly the same plane sign.
                    if tree.owners[q] != q or q == ids[0] or q == ids[1] or q == ids[2]:
                        continue
                    result = fp.orient3d(a, b, c, points[q])
                    if result.valid == 0:
                        wp.atomic_or(error, 0, 2)
                    elif result.sign > 0:
                        wp.atomic_or(error, 0, 64)
            else:
                if size > 125:
                    wp.atomic_or(error, 0, 16)
                    return
                stack[size] = tree.left[node]
                stack[size + 1] = tree.right[node]
                size += 2


@wp.func
def anchor_halfspace(plane: Halfspace, lo: wp.vec3, hi: wp.vec3, strict: bool):
    x = fp.minus(interval(lo[0], hi[0]), fp.scalar(wp.float64(plane.origin[0])))
    y = fp.minus(interval(lo[1], hi[1]), fp.scalar(wp.float64(plane.origin[1])))
    z = fp.minus(interval(lo[2], hi[2]), fp.scalar(wp.float64(plane.origin[2])))
    value = fp.plus(fp.plus(fp.times(plane.nx, x), fp.times(plane.ny, y)), fp.times(plane.nz, z))
    finite = wp.isfinite(value.value) and wp.isfinite(value.error)
    if strict:
        return finite and value.value < -value.error
    return finite and value.value <= -value.error


@wp.kernel
def check_faces_with_anchors(
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    boundary: wp.array[int],
    count: wp.array[int],
    tree: Tree,
    error: wp.array[int],
    depth: int,
    workers: int,
    anchors: wp.array[int],
):
    tid = wp.tid()
    pieces = 1 << depth
    worker, lane = tid // pieces, tid % pieces
    # Partition the radix tree by path bits. An early leaf belongs only to
    # the lane whose unused bits are zero, so every subtree is covered once.
    root = int(1)
    live = bool(True)
    for level in range(depth):
        if tree.count[root] > 0:
            live = (lane >> level) == 0
            break
        if ((lane >> level) & 1) == 0:
            root = tree.left[root]
        else:
            root = tree.right[root]
    if not live:
        return
    # Fixed launch size, device count, and a strided tail cover even malformed
    # candidates exposing all four faces per tetrahedron.
    for slot in range(worker, count[0], workers):
        ids = face_vertices(tets, boundary[slot])
        a, b, c = points[ids[0]], points[ids[1]], points[ids[2]]
        plane = halfspace(a, b, c)
        # Each lane may independently abandon strict pruning after finding a
        # coplanar witness. The shared flag is only OR-written, avoiding a
        # contended atomic read at every visited node.
        strict = bool(slot < anchors.shape[0])
        stack = wp.zeros(shape=128, dtype=int)
        stack[0] = root
        size = int(1)
        while size > 0:
            size -= 1
            node = stack[size]
            if anchor_halfspace(plane, tree.lower[node], tree.upper[node], strict):
                continue
            if tree.count[node] > 0:
                for k in range(tree.start[node], tree.start[node] + tree.count[node]):
                    q = tree.ids[k]
                    # Coincident sites have exactly the same plane sign.
                    if tree.owners[q] != q or q == ids[0] or q == ids[1] or q == ids[2]:
                        continue
                    result = fp.orient3d(a, b, c, points[q])
                    if result.valid == 0:
                        wp.atomic_or(error, 0, 2)
                    elif result.sign > 0:
                        wp.atomic_or(error, 0, 64)
                    elif result.sign == 0 and strict:
                        wp.atomic_or(anchors, slot, 1)
                        strict = False
            else:
                if size > 125:
                    wp.atomic_or(error, 0, 16)
                    return
                stack[size] = tree.left[node]
                stack[size + 1] = tree.right[node]
                size += 2


@wp.kernel
def select_certificate(
    anchors: wp.array[int],
    boundary_count: wp.array[int],
    error: wp.array[int],
    local: wp.array[int],
    global_check: wp.array[int],
):
    found = bool(False)
    for i in range(wp.min(anchors.shape[0], boundary_count[0])):
        found = found or anchors[i] == 0
    local[0] = int(found and error[0] == 0)
    global_check[0] = 1 - local[0]


@wp.kernel
def check_local_spheres(
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    count: wp.array[int],
    table: wp.array[int],
    error: wp.array[int],
    statistics: wp.array[int],
):
    face = wp.tid()
    if face >= 4 * count[0]:
        return
    i = face // 4
    ids = wp.vec4i(tets[i, 0], tets[i, 1], tets[i, 2], tets[i, 3])
    key = sorted_face(face_vertices(tets, face))
    slot = int(
        (
            wp.uint32(key[0]) * wp.uint32(73856093)
            ^ wp.uint32(key[1]) * wp.uint32(19349663)
            ^ wp.uint32(key[2]) * wp.uint32(83492791)
        )
        & wp.uint32(table.shape[0] - 1)
    )
    other = int(-1)
    for _ in range(table.shape[0]):
        other = table[slot]
        if other < 0:
            wp.atomic_or(error, 0, 32)
            return
        if sorted_face(face_vertices(tets, other)) == key:
            break
        slot = (slot + 1) & (table.shape[0] - 1)
    # Boundary faces and the representative of each paired face skip.
    # The other incident tetrahedron tests the representative's opposite site.
    if other != face:
        query = tets[other // 4, other % 4]
        result = symbolic_sphere(points, ids, query)
        wp.atomic_add(statistics, 0, 1)
        if result.valid == 0:
            wp.atomic_or(error, 0, 2)
        elif result.sign >= 0:
            wp.atomic_or(error, 0, 8)


@wp.kernel
def coverage(
    owners: wp.array[int], used: wp.array[int], count: wp.array[int], error: wp.array[int]
):
    i = wp.tid()
    if count[0] == 0 or (owners[i] == i and used[i] == 0):
        wp.atomic_or(error, 0, 128)


class DelaunayCertificate:
    """Internal fixed-buffer check; caller supplies a current original-site BVH."""

    def __init__(self, points, input_tets, input_count, tree):
        self.points, self.input_tets, self.input_count = points, input_tets, input_count
        self.tree, self.device = tree, points.device
        m, n = len(input_tets), len(points)
        with wp.ScopedDevice(self.device):
            self.tetrahedra = wp.empty_like(input_tets)
            self.tetrahedra_count = wp.zeros(1, dtype=int)
            self.error = wp.zeros(1, dtype=int)
            self.statistics = wp.zeros(2, dtype=int)
            self._selected = wp.empty(m, dtype=int)
            self._prefix = wp.empty_like(self._selected)
            self._scan = _Scan(m, self.device)
            self._used = wp.empty(n, dtype=int)
            self._table = wp.empty(1 << (8 * m - 1).bit_length(), dtype=int)
            self._occurrences = wp.empty_like(self._table)
            self._balance = wp.empty_like(self._table)
            self._parents = wp.empty(m, dtype=int)
            self._boundary = wp.empty(4 * m, dtype=int)
            self._boundary_count = wp.zeros(1, dtype=int)
            self._face_depth = 5
            self._face_workers = min(max(n, 128), 4096)
            self._ready = wp.empty(1, dtype=int)
            self._anchors = wp.zeros(16, dtype=int) if n >= 128 else None
            self._local = wp.zeros(1, dtype=int) if n >= 128 else None
            self._global = wp.zeros(1, dtype=int) if n >= 128 else None
            wp.load_module(module=__name__, device=self.device)

    def _launch(self, kernel, dim, args):
        wp.launch(kernel, dim, inputs=args, device=self.device, block_dim=32)

    def run(self, tree=None):
        tree = self.tree if tree is None else tree
        self.error.zero_()
        self.statistics.zero_()
        self._used.zero_()
        self._launch(
            select_positive,
            len(self.input_tets),
            [
                self.points,
                self.input_tets,
                self.input_count,
                tree.owners,
                self._selected,
                self.error,
            ],
        )
        self._scan.run(self._selected, self._prefix)
        self._launch(
            compact,
            len(self.input_tets),
            [
                self.input_tets,
                self._selected,
                self._prefix,
                tree.owners,
                self.tetrahedra,
                self.tetrahedra_count,
                self._used,
            ],
        )
        self._launch(ready, 1, [self.error, self._ready])
        wp.capture_if(self._ready, lambda: self._check(tree))
        self._launch(
            coverage,
            len(self.points),
            [tree.owners, self._used, self.tetrahedra_count, self.error],
        )

    def _global_spheres(self, tree):
        self._launch(
            check_spheres,
            len(self.tetrahedra),
            [
                self.points,
                self.tetrahedra,
                self.tetrahedra_count,
                tree,
                self.error,
                self.statistics,
            ],
        )

    def _check(self, tree):
        if self._anchors is None:
            self._global_spheres(tree)
        else:
            self._anchors.zero_()
        self._table.fill_(-1)
        self._occurrences.zero_()
        self._balance.zero_()
        self._launch(identity, len(self._parents), [self._parents])
        self._launch(
            index_faces,
            4 * len(self.tetrahedra),
            [
                self.tetrahedra,
                self.tetrahedra_count,
                self._table,
                self._occurrences,
                self._balance,
                self._parents,
            ],
        )
        self._boundary_count.zero_()
        self._launch(
            collect_boundary_faces,
            len(self._table),
            [
                self._table,
                self._occurrences,
                self._balance,
                self._boundary,
                self._boundary_count,
                self.error,
            ],
        )
        self._launch(
            check_faces if self._anchors is None else check_faces_with_anchors,
            self._face_workers * (1 << self._face_depth),
            [
                self.points,
                self.tetrahedra,
                self._boundary,
                self._boundary_count,
                tree,
                self.error,
                self._face_depth,
                self._face_workers,
            ]
            + ([] if self._anchors is None else [self._anchors]),
        )
        if self._anchors is not None:
            self._launch(
                select_certificate,
                1,
                [self._anchors, self._boundary_count, self.error, self._local, self._global],
            )
            wp.capture_if(
                self._local,
                lambda: self._launch(
                    check_local_spheres,
                    4 * len(self.tetrahedra),
                    [
                        self.points,
                        self.tetrahedra,
                        self.tetrahedra_count,
                        self._table,
                        self.error,
                        self.statistics,
                    ],
                ),
            )
            wp.capture_if(self._global, lambda: self._global_spheres(tree))
