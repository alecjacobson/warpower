"""Device facet extension of a globally certified Delaunay subset.

The caller supplies a rebuilt hierarchy and canonical owners for finite
original sites. Every iteration uses the full
original-coordinate certificate, including the final result.
"""

import operator

import warp as wp

from . import _filtered_predicates as fp
from ._delaunay_certificate import DelaunayCertificate, symbolic_sphere
from ._mesh_topology import face_vertices

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.kernel
def begin(count: wp.array[int], capacity: int, error: wp.array[int], running: wp.array[int]):
    error[0] = wp.where(count[0] <= 0 or count[0] > capacity, 1, 0)
    running[0] = int(error[0] == 0)


@wp.kernel
def decide(
    certificate: wp.array[int],
    count: wp.array[int],
    rounds: wp.array[int],
    maximum: int,
    error: wp.array[int],
    running: wp.array[int],
):
    running[0] = 0
    if error[0] != 0 or certificate[0] == 0:
        return
    if (certificate[0] & (~192)) != 0 or count[0] <= 0:
        error[0] = 1
    elif rounds[0] >= maximum:
        error[0] = 16
    else:
        running[0] = 1


@wp.func
def sorted_tet(v: wp.vec4i):
    r = v
    for i in range(1, 4):
        for j in range(i, 0, -1):
            if r[j] < r[j - 1]:
                a = int(r[j])
                r[j] = r[j - 1]
                r[j - 1] = a
    return r


@wp.kernel
def prepare_pool(
    tets: wp.array2d[int], count: wp.array[int], keys: wp.array[wp.vec4i], live: wp.array[int]
):
    i = wp.tid()
    live[i] = int(i < count[0])
    if i < count[0]:
        keys[i] = sorted_tet(wp.vec4i(tets[i, 0], tets[i, 1], tets[i, 2], tets[i, 3]))


@wp.kernel
def search_faces(
    points: wp.array[wp.vec3],
    owners: wp.array[int],
    tets: wp.array2d[int],
    faces: wp.array[int],
    count: wp.array[int],
    candidates: wp.array[int],
    workers: int,
    error: wp.array[int],
):
    tid = wp.tid()
    i, lane = tid // workers, tid % workers
    candidates[tid] = -1
    if i >= count[0]:
        return
    face = face_vertices(tets, faces[i])
    a, b, c = points[face[0]], points[face[1]], points[face[2]]
    best = int(-1)
    for q in range(lane, points.shape[0], workers):
        if owners[q] != q or q == face[0] or q == face[1] or q == face[2]:
            continue
        orientation = fp.orient3d(a, b, c, points[q])
        if orientation.valid == 0:
            wp.atomic_or(error, 0, 2)
            return
        if orientation.sign <= 0:
            continue
        if best < 0:
            best = q
        else:
            result = symbolic_sphere(points, wp.vec4i(face[0], face[1], face[2], best), q)
            if result.valid == 0 or result.sign == 0:
                wp.atomic_or(error, 0, 2)
                return
            if result.sign > 0:
                best = q
    candidates[tid] = best


@wp.kernel
def extend_faces(
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    faces: wp.array[int],
    count: wp.array[int],
    candidates: wp.array[int],
    workers: int,
    keys: wp.array[wp.vec4i],
    live: wp.array[int],
    offset: int,
    error: wp.array[int],
):
    i = wp.tid()
    live[offset + i] = 0
    if i >= count[0]:
        return
    face = face_vertices(tets, faces[i])
    best = int(-1)
    for lane in range(workers):
        q = candidates[i * workers + lane]
        if q < 0:
            continue
        if best < 0:
            best = q
        else:
            result = symbolic_sphere(points, wp.vec4i(face[0], face[1], face[2], best), q)
            if result.valid == 0 or result.sign == 0:
                wp.atomic_or(error, 0, 2)
                return
            if result.sign > 0:
                best = q
    if best >= 0:
        keys[offset + i] = sorted_tet(wp.vec4i(face[0], face[1], face[2], best))
        live[offset + i] = 1


@wp.kernel
def unique_pool(
    points: wp.array[wp.vec3],
    keys: wp.array[wp.vec4i],
    live: wp.array[int],
    table: wp.array[int],
    output: wp.array2d[int],
    count: wp.array[int],
    error: wp.array[int],
):
    i = wp.tid()
    if live[i] == 0:
        return
    key = keys[i]
    slot = int(
        (
            wp.uint32(key[0]) * wp.uint32(73856093)
            ^ wp.uint32(key[1]) * wp.uint32(19349663)
            ^ wp.uint32(key[2]) * wp.uint32(83492791)
            ^ wp.uint32(key[3]) * wp.uint32(2654435761)
        )
        & wp.uint32(table.shape[0] - 1)
    )
    for _ in range(table.shape[0]):
        old = wp.atomic_cas(table, slot, -1, i)
        if old == -1:
            at = wp.atomic_add(count, 0, 1)
            if at >= output.shape[0]:
                wp.atomic_or(error, 0, 4)
                return
            sign = fp.orient3d(points[key[0]], points[key[1]], points[key[2]], points[key[3]])
            if sign.valid == 0 or sign.sign == 0:
                wp.atomic_or(error, 0, 2)
                return
            if sign.sign < 0:
                a = int(key[0])
                key[0] = key[1]
                key[1] = a
            for k in range(4):
                output[at, k] = key[k]
            return
        if keys[old] == key:
            return
        slot = (slot + 1) & (table.shape[0] - 1)
    wp.atomic_or(error, 0, 4)


@wp.kernel
def progress(
    previous: wp.array[int],
    count: wp.array[int],
    rounds: wp.array[int],
    error: wp.array[int],
    running: wp.array[int],
):
    rounds[0] += 1
    if error[0] == 0 and count[0] <= previous[0]:
        error[0] = 8
    running[0] = int(error[0] == 0)


@wp.kernel
def finish(error: wp.array[int], certificate: wp.array[int], count: wp.array[int]):
    if error[0] == 0 and certificate[0] != 0:
        error[0] = 32
    if error[0] != 0:
        count[0] = 0


class FacetCompletion:
    """Internal bounded completion; never expose a partial result as successful."""

    def __init__(
        self, points, seeds, seed_count, tree, *, capacity=None, max_rounds=64, query_block_dim=256
    ):
        self.points, self.seeds, self.seed_count = points, seeds, seed_count
        self.tree, self.device = tree, points.device
        self._query_block_dim = query_block_dim
        self.capacity = len(seeds) if capacity is None else operator.index(capacity)
        self.max_rounds = operator.index(max_rounds)
        if self.capacity < len(seeds) or self.capacity < 1 or self.max_rounds < 0:
            raise ValueError("Require capacity >= seed capacity > 0 and max_rounds >= 0")
        with wp.ScopedDevice(self.device):
            self._work = wp.empty((self.capacity, 4), dtype=int)
            self._count = wp.zeros(1, dtype=int)
            self._certificate = DelaunayCertificate(
                points, self._work, self._count, tree, point_parallel=len(points) <= 2048
            )
            self._certificate._anchors = None
            self.tetrahedra = self._certificate.tetrahedra
            self.tetrahedra_count = self._certificate.tetrahedra_count
            self.certificate_error = self._certificate.error
            self.error = wp.zeros(1, dtype=int)
            self.rounds = wp.zeros(1, dtype=int)
            self._running = wp.zeros(1, dtype=int)
            self._workers = 32
            self._candidates = wp.empty(4 * self.capacity * self._workers, dtype=int)
            self._keys = wp.empty(5 * self.capacity, dtype=wp.vec4i)
            self._live = wp.empty(5 * self.capacity, dtype=int)
            self._table = wp.empty(1 << (10 * self.capacity - 1).bit_length(), dtype=int)
            wp.load_module(module=__name__, device=self.device)
            if self._query_block_dim != 256:
                wp.load_module(
                    module=__name__, device=self.device, block_dim=self._query_block_dim
                )

    def _launch(self, kernel, dim, args, *, block_dim=256):
        wp.launch(kernel, dim, inputs=args, device=self.device, block_dim=block_dim)

    def run(self, tree=None):
        tree = self.tree if tree is None else tree
        self.rounds.zero_()
        self.tetrahedra_count.zero_()
        self.certificate_error.zero_()
        wp.copy(self._work, self.seeds, count=self.seeds.size)
        wp.copy(self._count, self.seed_count)
        self._launch(begin, 1, [self._count, len(self.seeds), self.error, self._running])
        wp.capture_if(self._running, lambda: self._certify(tree))
        wp.capture_while(self._running, lambda: self._round(tree))
        self._launch(finish, 1, [self.error, self.certificate_error, self.tetrahedra_count])
        return self

    def _certify(self, tree):
        self._certificate.run(tree)
        self._launch(
            decide,
            1,
            [
                self.certificate_error,
                self.tetrahedra_count,
                self.rounds,
                self.max_rounds,
                self.error,
                self._running,
            ],
        )

    def _round(self, tree):
        c = self._certificate
        self._launch(
            prepare_pool,
            self.capacity,
            [self.tetrahedra, self.tetrahedra_count, self._keys, self._live],
        )
        self._launch(
            search_faces,
            4 * self.capacity * self._workers,
            [
                self.points,
                tree.owners,
                self.tetrahedra,
                c._boundary,
                c._boundary_count,
                self._candidates,
                self._workers,
                self.error,
            ],
            block_dim=self._query_block_dim,
        )
        self._launch(
            extend_faces,
            4 * self.capacity,
            [
                self.points,
                self.tetrahedra,
                c._boundary,
                c._boundary_count,
                self._candidates,
                self._workers,
                self._keys,
                self._live,
                self.capacity,
                self.error,
            ],
            block_dim=self._query_block_dim,
        )
        self._table.fill_(-1)
        self._count.zero_()
        self._launch(
            unique_pool,
            5 * self.capacity,
            [self.points, self._keys, self._live, self._table, self._work, self._count, self.error],
        )
        self._launch(
            progress,
            1,
            [self.tetrahedra_count, self._count, self.rounds, self.error, self._running],
        )
        wp.capture_if(self._running, lambda: self._certify(tree))
