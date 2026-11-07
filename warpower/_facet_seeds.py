"""Device filtering of uncertified cells to strict original-coordinate lower facets.

Internal stage: the caller supplies finite original sites and current canonical
owners. Malformed individual cells are skipped; invalid counts or predicate
failure reject the batch. A downstream completeness certificate remains required.
"""

import warp as wp

from . import _filtered_predicates as fp
from . import _sphere_bounds as sb
from ._bvh import Tree
from ._delaunay_certificate import outside, sphere, symbolic_sphere
from ._facet_completion import sorted_tet, unique_pool

wp.set_module_options(
    {"enable_backward": False, "fast_math": False, "fuse_fp": False, "block_dim": 32}
)


@wp.kernel
def seed_keys(
    points: wp.array[wp.vec3],
    source: wp.array2d[int],
    count: wp.array[int],
    owners: wp.array[int],
    keys: wp.array[wp.vec4i],
    live: wp.array[int],
    error: wp.array[int],
):
    i = wp.tid()
    live[i] = 0
    if count[0] < 0 or count[0] > source.shape[0]:
        wp.atomic_or(error, 0, 4)
        return
    if i >= count[0]:
        return
    v = wp.vec4i(0)
    for k in range(4):
        j = source[i, k]
        if j < 0 or j >= points.shape[0]:
            return
        v[k] = owners[j]
        for previous in range(k):
            if v[previous] == v[k]:
                return
    o = fp.orient3d(points[v[0]], points[v[1]], points[v[2]], points[v[3]])
    if o.valid == 0:
        wp.atomic_or(error, 0, 2)
        return
    if o.sign == 0:
        return
    keys[i] = sorted_tet(v)
    live[i] = 1


@wp.kernel
def reset_live(count: wp.array[int], live: wp.array[int]):
    i = wp.tid()
    live[i] = int(i < count[0])


@wp.kernel
def check_seeds(
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    count: wp.array[int],
    tree: Tree,
    live: wp.array[int],
    error: wp.array[int],
):
    tid = wp.tid()
    i, lane = tid // 32, tid % 32
    if error[0] != 0 or i >= count[0]:
        return
    ids = wp.vec4i(tets[i, 0], tets[i, 1], tets[i, 2], tets[i, 3])
    shape = sphere(points[ids[0]], points[ids[1]], points[ids[2]], points[ids[3]])
    box = sb.bounds(shape.origin, shape.nx, shape.ny, shape.nz, shape.determinant)
    for q in range(lane, points.shape[0], 32):
        if tree.owners[q] != q or q == ids[0] or q == ids[1] or q == ids[2] or q == ids[3]:
            continue
        p = points[q]
        if sb.outside(box, p, p) or outside(shape, p, p):
            continue
        result = symbolic_sphere(points, ids, q)
        if result.valid == 0:
            wp.atomic_or(error, 0, 2)
            wp.atomic_and(live, i, 0)
        elif result.sign >= 0:
            wp.atomic_and(live, i, 0)


@wp.kernel
def compact(
    source: wp.array2d[int],
    live: wp.array[int],
    output: wp.array2d[int],
    count: wp.array[int],
    error: wp.array[int],
):
    i = wp.tid()
    if live[i] == 0 or error[0] != 0:
        return
    at = wp.atomic_add(count, 0, 1)
    for k in range(4):
        output[at, k] = source[i, k]


class CertifiedSeeds:
    """Preallocated exact seed filter; input is a device-count prefix of cells."""

    def __init__(self, points, source, count, tree, *, block_dim=256, prepare_geometry=False):
        self.points, self.source, self.count, self.tree = points, source, count, tree
        self.device = points.device
        self._block_dim = block_dim
        n = len(source)
        if n < 1:
            raise ValueError("Candidate capacity must be positive")
        with wp.ScopedDevice(points.device):
            self.keys = wp.empty(n, dtype=wp.vec4i)
            self.live = wp.empty(n, dtype=int)
            self.table = wp.empty(1 << (2 * n - 1).bit_length(), dtype=int)
            self.unique = wp.empty_like(source)
            self.unique_count = wp.zeros(1, dtype=int)
            self.tetrahedra = wp.empty_like(source)
            self.tetrahedra_count = wp.zeros(1, dtype=int)
            self.error = wp.zeros(1, dtype=int)
            wp.load_module(module=__name__, device=points.device)
            if self._block_dim != 256:
                wp.load_module(module=__name__, device=points.device, block_dim=self._block_dim)
            wp.load_module(module="warpower._facet_completion", device=points.device)
            self._prepared = None
            if prepare_geometry:
                from ._prepared_seeds import PreparedSeedQueries

                self._prepared = PreparedSeedQueries(
                    self.points,
                    self.unique,
                    self.unique_count,
                    self.tree,
                    self.live,
                    self.error,
                    block_dim=self._block_dim,
                )

    def run(self):
        n = len(self.source)
        self.error.zero_()
        self.unique_count.zero_()
        self.tetrahedra_count.zero_()
        self.table.fill_(-1)
        wp.launch(
            seed_keys,
            n,
            device=self.device,
            inputs=[
                self.points,
                self.source,
                self.count,
                self.tree.owners,
                self.keys,
                self.live,
                self.error,
            ],
        )
        wp.launch(
            unique_pool,
            n,
            device=self.device,
            inputs=[
                self.points,
                self.keys,
                self.live,
                self.table,
                self.unique,
                self.unique_count,
                self.error,
            ],
        )
        wp.launch(reset_live, n, inputs=[self.unique_count, self.live], device=self.device)
        if self._prepared is None:
            wp.launch(
                check_seeds,
                32 * n,
                device=self.device,
                inputs=[
                    self.points,
                    self.unique,
                    self.unique_count,
                    self.tree,
                    self.live,
                    self.error,
                ],
                block_dim=self._block_dim,
            )
        else:
            self._prepared.run()
        wp.launch(
            compact,
            n,
            device=self.device,
            inputs=[self.unique, self.live, self.tetrahedra, self.tetrahedra_count, self.error],
        )
