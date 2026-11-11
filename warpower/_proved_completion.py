"""Private reuse of exact cell proofs within one unchanged-input completion.

Seed filtering, every completion round and proof consumption are recorded as one
ordered operation. Points and canonical owners cannot change between them.
Empty-sphere and immediately preceding hull-support results may be reused;
every newly exposed face is globally checked before its result is consumed.
No proof survives a call to run(), and no caller-supplied trust mask is accepted.
"""

import warp as wp

from . import _bounded_predicates as af
from . import _filtered_predicates as fp
from . import _sphere_bounds as sb
from ._bvh import Tree
from ._delaunay_certificate import (
    DelaunayCertificate,
    Halfspace,
    halfspace,
    inside_halfspace,
    outside,
    sphere,
    symbolic_sphere,
)
from ._facet_completion import FacetCompletion
from ._mesh_topology import face_vertices

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.kernel
def compact_proofs(
    source: wp.array[int],
    selected: wp.array[int],
    prefix: wp.array[int],
    output: wp.array[int],
):
    i = wp.tid()
    if selected[i] != 0:
        output[prefix[i] - 1] = source[i]


@wp.kernel
def check_unproved_spheres(
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    count: wp.array[int],
    tree: Tree,
    error: wp.array[int],
    statistics: wp.array[int],
    proved: wp.array[int],
    proof_statistics: wp.array[int],
):
    tid = wp.tid()
    i, lane = tid // 32, tid % 32
    if i >= count[0]:
        return
    if proved[i] != 0:
        if lane == 0:
            wp.atomic_add(proof_statistics, 0, 1)
        return
    if lane == 0:
        wp.atomic_add(proof_statistics, 1, 1)
    ids = wp.vec4i(tets[i, 0], tets[i, 1], tets[i, 2], tets[i, 3])
    shape = sphere(points[ids[0]], points[ids[1]], points[ids[2]], points[ids[3]])
    box = sb.bounds(shape.origin, shape.nx, shape.ny, shape.nz, shape.determinant)
    queries = int(0)
    for q in range(lane, points.shape[0], 32):
        if tree.owners[q] != q or q == ids[0] or q == ids[1] or q == ids[2] or q == ids[3]:
            continue
        p = points[q]
        if sb.outside(box, p, p) or outside(shape, p, p):
            continue
        queries += 1
        result = symbolic_sphere(points, ids, q)
        if result.valid == 0:
            wp.atomic_or(error, 0, 2)
        elif result.sign >= 0:
            wp.atomic_or(error, 0, 8)
    wp.atomic_add(statistics, 0, queries)


@wp.kernel
def unique_proved_pool(
    points: wp.array[wp.vec3],
    keys: wp.array[wp.vec4i],
    live: wp.array[int],
    table: wp.array[int],
    output: wp.array2d[int],
    count: wp.array[int],
    error: wp.array[int],
    previous_prefix: int,
    proved: wp.array[int],
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
            # Trust the winning source, not a hash slot or a compacted index.
            # A new duplicate winning over an old cell safely loses its proof.
            proved[at] = int(i < previous_prefix)
            return
        if keys[old] == key:
            return
        slot = (slot + 1) & (table.shape[0] - 1)
    wp.atomic_or(error, 0, 4)


@wp.func
def axis_supported(a: wp.vec3, b: wp.vec3, c: wp.vec3, opposite: wp.vec3, lo: wp.vec3, hi: wp.vec3):
    # The caller just certified a positive tetrahedron. Its oriented face has
    # the opposite vertex strictly on its interior side. Exact coordinate
    # equality therefore identifies both the plane and the interior direction.
    # Current original-site bounds prove support even at exact coplanarity.
    for axis in range(3):
        plane = a[axis]
        if plane != b[axis] or plane != c[axis] or plane == opposite[axis]:
            continue
        if not wp.isfinite(plane) or not wp.isfinite(opposite[axis]):
            continue
        if not wp.isfinite(lo[axis]) or not wp.isfinite(hi[axis]) or lo[axis] > hi[axis]:
            continue
        if opposite[axis] > plane:
            if lo[axis] >= plane:
                return True
        else:
            if hi[axis] <= plane:
                return True
    return False


@wp.func
def nonnegative(value: fp.Ball):
    if (
        not wp.isfinite(value.value)
        or not wp.isfinite(value.error)
        or value.error < wp.float64(0.0)
    ):
        return -1
    if value.value >= value.error:
        return 1
    if value.value <= -value.error:
        return 0
    return -1


@wp.func
def maximum_corner(plane: Halfspace):
    # Certified normal signs identify a corner maximizing the exact plane value.
    # Uncertain components retain the original box test.
    x, y, z = nonnegative(plane.nx), nonnegative(plane.ny), nonnegative(plane.nz)
    if x < 0 or y < 0 or z < 0:
        return -1
    return x + 2 * y + 4 * z


@wp.func
def bounded_box_state(a: wp.vec3, b: wp.vec3, c: wp.vec3, mask: int, lo: wp.vec3, hi: wp.vec3):
    if mask < 0 or mask > 7 or not af.bounded(lo) or not af.bounded(hi):
        return 0
    if lo[0] > hi[0] or lo[1] > hi[1] or lo[2] > hi[2]:
        return 0
    q = wp.vec3(
        wp.where((mask & 1) != 0, hi[0], lo[0]),
        wp.where((mask & 2) != 0, hi[1], lo[1]),
        wp.where((mask & 4) != 0, hi[2], lo[2]),
    )
    value = af.orientation_filter(a, b, c, q)
    if value.value < -value.error:
        return 1
    if value.value > value.error:
        # This rejects box containment only. Its sites still need traversal.
        return -1
    return 0


@wp.kernel
def record_face_support(
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    boundary: wp.array[int],
    count: wp.array[int],
    tree: Tree,
    error: wp.array[int],
    depth: int,
    workers: int,
    supported: wp.array[int],
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
        # A different partition may already have disproved this same face.
        # Atomic reads match the monotone atomic clears below.
        if wp.atomic_add(supported, slot, 0) == 0:
            continue
        ids = face_vertices(tets, boundary[slot])
        a, b, c = points[ids[0]], points[ids[1]], points[ids[2]]
        face = boundary[slot]
        opposite = points[tets[face // 4, face % 4]]
        if axis_supported(a, b, c, opposite, tree.lower[1], tree.upper[1]):
            continue
        plane = halfspace(a, b, c)
        mask = maximum_corner(plane)
        stack = wp.zeros(shape=128, dtype=int)
        stack[0] = root
        size = int(1)
        while size > 0:
            if wp.atomic_add(supported, slot, 0) == 0:
                break
            size -= 1
            node = stack[size]
            state = bounded_box_state(a, b, c, mask, tree.lower[node], tree.upper[node])
            if state == 1:
                continue
            if state == 0 and inside_halfspace(plane, tree.lower[node], tree.upper[node]):
                continue
            if tree.count[node] > 0:
                for k in range(tree.start[node], tree.start[node] + tree.count[node]):
                    q = tree.ids[k]
                    # Coincident sites have exactly the same plane sign.
                    if tree.owners[q] != q or q == ids[0] or q == ids[1] or q == ids[2]:
                        continue
                    result = af.orient3d(a, b, c, points[q])
                    if result.valid == 0:
                        wp.atomic_or(error, 0, 2)
                    elif result.sign > 0:
                        wp.atomic_or(error, 0, 64)
                        wp.atomic_and(supported, slot, 0)
                        # One exact exterior witness disproves this face's
                        # support. End only this subtree/face, not the thread:
                        # the strided outer loop must still check later slots.
                        size = 0
                        break
            else:
                if size > 125:
                    wp.atomic_or(error, 0, 16)
                    return
                stack[size] = tree.left[node]
                stack[size + 1] = tree.right[node]
                size += 2


@wp.kernel
def search_unproved_faces(
    points: wp.array[wp.vec3],
    owners: wp.array[int],
    tets: wp.array2d[int],
    faces: wp.array[int],
    count: wp.array[int],
    candidates: wp.array[int],
    workers: int,
    error: wp.array[int],
    supported: wp.array[int],
    statistics: wp.array[int],
):
    tid = wp.tid()
    i, lane = tid // workers, tid % workers
    candidates[tid] = -1
    if i >= count[0]:
        return
    if lane == 0:
        wp.atomic_add(statistics, wp.where(supported[i] != 0, 0, 1), 1)
    if supported[i] != 0:
        return
    face = face_vertices(tets, faces[i])
    a, b, c = points[face[0]], points[face[1]], points[face[2]]
    best = int(-1)
    for q in range(lane, points.shape[0], workers):
        if owners[q] != q or q == face[0] or q == face[1] or q == face[2]:
            continue
        orientation = af.orient3d(a, b, c, points[q])
        if orientation.valid == 0:
            wp.atomic_or(error, 0, 2)
            return
        if orientation.sign <= 0:
            continue
        if best < 0:
            best = q
        else:
            result = af.symbolic_sphere(points, wp.vec4i(face[0], face[1], face[2], best), q)
            if result.valid == 0 or result.sign == 0:
                wp.atomic_or(error, 0, 2)
                return
            if result.sign > 0:
                best = q
    candidates[tid] = best


class ProvedCertificate(DelaunayCertificate):
    """Completion-only certificate, with private same-input proof provenance."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with wp.ScopedDevice(self.device):
            self._input_proved = wp.empty(len(self.input_tets), dtype=int)
            self._proved = wp.empty_like(self._input_proved)
            # Cell checks skipped/performed, accumulated over one completion.
            self.proof_statistics = wp.zeros(2, dtype=int)
            self._face_supported = wp.empty_like(self._boundary)
            self.face_statistics = wp.zeros(2, dtype=int)
            self._prepared_unproved = None
            self._hull_cache = None
            wp.load_module(module=__name__, device=self.device)
            wp.load_module(module=__name__, device=self.device, block_dim=32)

    def _check(self, tree):
        self._launch(
            compact_proofs,
            len(self.input_tets),
            [self._input_proved, self._selected, self._prefix, self._proved],
        )
        super()._check(tree)

    def _global_spheres(self, tree):
        if self._prepared_unproved is not None and self.point_parallel:
            self._prepared_unproved.run(tree)
            return
        if not (self.point_parallel or 128 <= len(self.points) <= 512):
            # Large clouds retain the original complete BVH check.
            super()._global_spheres(tree)
            return
        self._launch(
            check_unproved_spheres,
            32 * len(self.tetrahedra),
            [
                self.points,
                self.tetrahedra,
                self.tetrahedra_count,
                tree,
                self.error,
                self.statistics,
                self._proved,
                self.proof_statistics,
            ],
        )

    def _hull_support(self, tree):
        if self._hull_cache is not None:
            self._hull_cache.check(self, tree)
            return
        # This array belongs to the current boundary ordering only. A new
        # certificate always resets and recomputes it before any search.
        self._face_supported.fill_(1)
        self._launch(
            record_face_support,
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
                self._face_supported,
            ],
        )


class SeededFacetCompletion(FacetCompletion):
    """Run the real seed filter and immediately consume its current cell proofs."""

    _certificate_type = ProvedCertificate

    def __init__(self, seed_filter, *, cache_hull_support=False, **kwargs):
        self._seed_filter = seed_filter
        super().__init__(
            seed_filter.points,
            seed_filter.tetrahedra,
            seed_filter.tetrahedra_count,
            seed_filter.tree,
            **kwargs,
        )
        if cache_hull_support:
            from ._completion_hull_cache import HullSupportCache

            c = self._certificate
            c._hull_cache = HullSupportCache(len(c.points), len(c._boundary), c.device)
        seed = seed_filter._prepared
        if seed is not None and 4096 < len(seed_filter.points) <= 16384:
            from ._stackless_seeds import StacklessSeedQueries
            from ._unproved_queries import PreparedUnprovedQueries

            c = self._certificate
            if (
                isinstance(seed, StacklessSeedQueries)
                and len(c.tetrahedra) <= len(seed.shapes)
                and len(c.tetrahedra) <= len(seed.boxes)
            ):
                # Seed queries finish before completion begins. Their geometry
                # is dead scratch; every later consumer prepares its own current
                # cells again. Keep direct array references, without a cycle.
                c._prepared_unproved = PreparedUnprovedQueries(
                    c.points,
                    c.tetrahedra,
                    c.tetrahedra_count,
                    c.tree,
                    c.error,
                    c.statistics,
                    c._proved,
                    c.proof_statistics,
                    segment=seed.segment,
                    shapes=seed.shapes,
                    boxes=seed.boxes,
                )

    def run(self):
        if self._certificate._hull_cache is not None:
            self._certificate._hull_cache.reset()
        self._seed_filter.run()
        self._certificate._input_proved.fill_(1)
        self._certificate.proof_statistics.zero_()
        self._certificate.face_statistics.zero_()
        return super().run(self._seed_filter.tree)

    def _deduplicate(self):
        # Entered only after the preceding certificate proved every retained
        # cell. Coverage/hull-support failures may remain; other errors stop it.
        self._table.fill_(-1)
        self._count.zero_()
        self._launch(
            unique_proved_pool,
            5 * self.capacity,
            [
                self.points,
                self._keys,
                self._live,
                self._table,
                self._work,
                self._count,
                self.error,
                self.capacity,
                self._certificate._input_proved,
            ],
        )

    def _search_faces(self, tree):
        # decide() admits this round only if the entire preceding certificate
        # has no geometric errors. Thus every live BVH piece finished checking
        # these exact faces against these unchanged sites and canonical owners.
        c = self._certificate
        self._launch(
            search_unproved_faces,
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
                c._face_supported,
                c.face_statistics,
            ],
            block_dim=self._query_block_dim,
        )
