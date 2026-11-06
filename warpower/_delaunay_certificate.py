"""Exact original-coordinate certificate for optional Delaunay recovery.

Each accepted positive-volume cell must be a strict lower facet after symbolic
lifting perturbation. This establishes nonoverlap even for cospherical inputs;
paired interior faces and supporting hull faces then establish completeness.
"""

import warp as wp

from . import _filtered_predicates as fp
from ._bvh import Tree
from ._canonical import mix
from ._domain import _Scan
from ._mesh_topology import face_vertices, identity, index_faces

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
    stack = wp.zeros(shape=128, dtype=int)
    size = int(1)
    stack[0] = 1
    queries, visits = int(0), int(0)
    while size > 0:
        size -= 1
        node = stack[size]
        visits += 1
        if outside(shape, tree.lower[node], tree.upper[node]):
            continue
        if tree.count[node] > 0:
            for k in range(tree.start[node], tree.start[node] + tree.count[node]):
                q = tree.ids[k]
                if tree.owners[q] != q or q == ids[0] or q == ids[1] or q == ids[2] or q == ids[3]:
                    continue
                queries += 1
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


@wp.func
def inside_halfspace(a: wp.vec3, b: wp.vec3, c: wp.vec3, lo: wp.vec3, hi: wp.vec3):
    ux, uy, uz = fp.difference(b[0], a[0]), fp.difference(b[1], a[1]), fp.difference(b[2], a[2])
    vx, vy, vz = fp.difference(c[0], a[0]), fp.difference(c[1], a[1]), fp.difference(c[2], a[2])
    x = fp.minus(interval(lo[0], hi[0]), fp.scalar(wp.float64(a[0])))
    y = fp.minus(interval(lo[1], hi[1]), fp.scalar(wp.float64(a[1])))
    z = fp.minus(interval(lo[2], hi[2]), fp.scalar(wp.float64(a[2])))
    value = fp.det3(ux, uy, uz, vx, vy, vz, x, y, z)
    return wp.isfinite(value.value) and wp.isfinite(value.error) and value.value <= -value.error


@wp.kernel
def check_faces(
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    table: wp.array[int],
    occurrences: wp.array[int],
    balance: wp.array[int],
    tree: Tree,
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
    ids = face_vertices(tets, table[slot])
    a, b, c = points[ids[0]], points[ids[1]], points[ids[2]]
    stack = wp.zeros(shape=128, dtype=int)
    stack[0] = 1
    size = int(1)
    while size > 0:
        size -= 1
        node = stack[size]
        if inside_halfspace(a, b, c, tree.lower[node], tree.upper[node]):
            continue
        if tree.count[node] > 0:
            for k in range(tree.start[node], tree.start[node] + tree.count[node]):
                q = tree.ids[k]
                if q == ids[0] or q == ids[1] or q == ids[2]:
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
            self._ready = wp.empty(1, dtype=int)
            wp.load_module(module=__name__, device=self.device)

    def _launch(self, kernel, dim, args):
        wp.launch(kernel, dim, inputs=args, device=self.device, block_dim=32)

    def run(self):
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
                self.tree.owners,
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
                self.tree.owners,
                self.tetrahedra,
                self.tetrahedra_count,
                self._used,
            ],
        )
        self._launch(ready, 1, [self.error, self._ready])
        wp.capture_if(self._ready, self._check)
        self._launch(
            coverage,
            len(self.points),
            [self.tree.owners, self._used, self.tetrahedra_count, self.error],
        )

    def _check(self):
        self._launch(
            check_spheres,
            len(self.tetrahedra),
            [
                self.points,
                self.tetrahedra,
                self.tetrahedra_count,
                self.tree,
                self.error,
                self.statistics,
            ],
        )
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
        self._launch(
            check_faces,
            len(self._table),
            [
                self.points,
                self.tetrahedra,
                self._table,
                self._occurrences,
                self._balance,
                self.tree,
                self.error,
            ],
        )
