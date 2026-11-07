"""Private reuse of exact cell proofs within one unchanged-input completion.

Seed filtering, every completion round and proof consumption are recorded as one
ordered operation. Points and canonical owners cannot change between them. Only
empty-sphere predicates are reused; all other certificate checks still execute.
No proof survives a call to run(), and no caller-supplied trust mask is accepted.
"""

import warp as wp

from . import _filtered_predicates as fp
from . import _sphere_bounds as sb
from ._bvh import Tree
from ._delaunay_certificate import DelaunayCertificate, outside, sphere, symbolic_sphere
from ._facet_completion import FacetCompletion

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


class ProvedCertificate(DelaunayCertificate):
    """Completion-only certificate, with private same-input proof provenance."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with wp.ScopedDevice(self.device):
            self._input_proved = wp.empty(len(self.input_tets), dtype=int)
            self._proved = wp.empty_like(self._input_proved)
            # Cell checks skipped/performed, accumulated over one completion.
            self.proof_statistics = wp.zeros(2, dtype=int)
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


class SeededFacetCompletion(FacetCompletion):
    """Run the real seed filter and immediately consume its current cell proofs."""

    _certificate_type = ProvedCertificate

    def __init__(self, seed_filter, **kwargs):
        self._seed_filter = seed_filter
        super().__init__(
            seed_filter.points,
            seed_filter.tetrahedra,
            seed_filter.tetrahedra_count,
            seed_filter.tree,
            **kwargs,
        )

    def run(self):
        self._seed_filter.run()
        self._certificate._input_proved.fill_(1)
        self._certificate.proof_statistics.zero_()
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
