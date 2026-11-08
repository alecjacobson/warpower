"""Private #233 split prototype: preserve query counts, cache exact fallback.

The caller supplies finite original sites, canonical owners, valid cell indices,
and the rebuilt segment/permutation, as required by private facet completion.
No production installation or scratch sharing is provided in this first probe.
"""

import warp as wp

from warpower import _sphere_bounds as sb
from warpower import _bounded_predicates as low
from warpower._analytic_predicates import strict
from warpower._prepared_seeds import point_value
from warpower._bvh import Tree
from warpower._delaunay_certificate import Sphere, outside, sphere, symbolic_sphere
from warpower._stackless_seeds import successor

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.kernel
def prepare_unproved(
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    count: wp.array[int],
    proved: wp.array[int],
    shapes: wp.array[Sphere],
    boxes: wp.array[sb.Box],
):
    i = wp.tid()
    if i >= count[0] or proved[i] != 0:
        return
    ids = wp.vec4i(tets[i, 0], tets[i, 1], tets[i, 2], tets[i, 3])
    shape = sphere(points[ids[0]], points[ids[1]], points[ids[2]], points[ids[3]])
    shapes[i] = shape
    boxes[i] = sb.bounds(shape.origin, shape.nx, shape.ny, shape.nz, shape.determinant)


@wp.kernel
def strict_unproved(
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    count: wp.array[int],
    tree: Tree,
    error: wp.array[int],
    statistics: wp.array[int],
    proved: wp.array[int],
    proof_statistics: wp.array[int],
    shapes: wp.array[Sphere],
    boxes: wp.array[sb.Box],
    segment: Tree,
    pending: wp.array[wp.uint32],
    unresolved: wp.array2d[int],
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
    a, b, c, d = points[ids[0]], points[ids[1]], points[ids[2]], points[ids[3]]
    first = int(-1)
    shape = shapes[i]
    box = boxes[i]
    queries = int(0)
    group_base = wp.max(1, segment.leaf_base // 32)
    node = int(1)
    visits = int(0)
    while node > 0 and visits < 2 * group_base - 1:
        current = node
        visits += 1
        if sb.outside(box, segment.lower[current], segment.upper[current]):
            node = successor(current)
            continue
        if current < group_base:
            node = 2 * current
            continue
        node = successor(current)
        rank = 32 * (current - group_base) + lane
        if rank >= points.shape[0]:
            continue
        q = tree.ids[rank]
        if tree.owners[q] != q or q == ids[0] or q == ids[1] or q == ids[2] or q == ids[3]:
            continue
        p = points[q]
        if sb.outside(box, p, p) or outside(shape, p, p):
            continue
        queries += 1
        fast = low.sphere_filter(a, b, c, d, p)
        if strict(fast):
            if fast.value > wp.float64(0.0):
                wp.atomic_or(error, 0, 8)
            continue
        value = point_value(shape, p)
        if strict(value):
            if value.value > wp.float64(0.0):
                wp.atomic_or(error, 0, 8)
            continue
        if first == -1:
            first = q
        else:
            first = -2
    if node != 0:
        # A valid current segment cannot exhaust its full-tree visit budget.
        wp.atomic_or(error, 0, 2)
    wp.atomic_add(statistics, 0, queries)

    if first != -1:
        unresolved[i, lane] = first
        wp.atomic_or(pending, i, wp.uint32(1) << wp.uint32(lane))


@wp.kernel
def exact_unproved(
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    count: wp.array[int],
    tree: Tree,
    error: wp.array[int],
    statistics: wp.array[int],
    proved: wp.array[int],
    proof_statistics: wp.array[int],
    shapes: wp.array[Sphere],
    boxes: wp.array[sb.Box],
    segment: Tree,
    pending: wp.array[wp.uint32],
    unresolved: wp.array2d[int],
):
    tid = wp.tid()
    i, lane = tid // 32, tid % 32
    if i >= count[0]:
        return
    if proved[i] != 0 or (pending[i] & (wp.uint32(1) << wp.uint32(lane))) == wp.uint32(0):
        return
    ids = wp.vec4i(tets[i, 0], tets[i, 1], tets[i, 2], tets[i, 3])
    saved = unresolved[i, lane]
    if saved >= 0:
        if saved >= points.shape[0]:
            wp.atomic_or(error, 0, 2)
            return
        result = symbolic_sphere(points, ids, saved)
        if result.valid == 0:
            wp.atomic_or(error, 0, 2)
        elif result.sign >= 0:
            wp.atomic_or(error, 0, 8)
        return
    if saved != -2:
        wp.atomic_or(error, 0, 2)
        return
    a, b, c, d = points[ids[0]], points[ids[1]], points[ids[2]], points[ids[3]]
    shape = shapes[i]
    box = boxes[i]
    group_base = wp.max(1, segment.leaf_base // 32)
    node = int(1)
    visits = int(0)
    while node > 0 and visits < 2 * group_base - 1:
        current = node
        visits += 1
        if sb.outside(box, segment.lower[current], segment.upper[current]):
            node = successor(current)
            continue
        if current < group_base:
            node = 2 * current
            continue
        node = successor(current)
        rank = 32 * (current - group_base) + lane
        if rank >= points.shape[0]:
            continue
        q = tree.ids[rank]
        if tree.owners[q] != q or q == ids[0] or q == ids[1] or q == ids[2] or q == ids[3]:
            continue
        p = points[q]
        if sb.outside(box, p, p) or outside(shape, p, p):
            continue
        fast = low.sphere_filter(a, b, c, d, p)
        if strict(fast):
            if fast.value > wp.float64(0.0):
                wp.atomic_or(error, 0, 8)
            continue
        value = point_value(shape, p)
        if strict(value):
            if value.value > wp.float64(0.0):
                wp.atomic_or(error, 0, 8)
            continue
        result = symbolic_sphere(points, ids, q)
        if result.valid == 0:
            wp.atomic_or(error, 0, 2)
        elif result.sign >= 0:
            wp.atomic_or(error, 0, 8)
    if node != 0:
        # A valid current segment cannot exhaust its full-tree visit budget.
        wp.atomic_or(error, 0, 2)


class PreparedUnprovedQueries:
    def __init__(
        self,
        points,
        tets,
        count,
        tree,
        error,
        statistics,
        proved,
        proof_statistics,
        *,
        segment,
        block_dim=32,
    ):
        if (
            segment.n != len(points)
            or tree.n != len(points)
            or segment.ids.ptr != tree.ids.ptr
            or segment.leaf_size != 1
            or segment.leaf_base != 1 << (len(points) - 1).bit_length()
        ):
            raise ValueError("Query segment must share the current point permutation")
        self.device = points.device
        self.capacity = len(tets)
        self.block_dim = block_dim
        self.shapes = wp.empty(self.capacity, dtype=Sphere, device=self.device)
        self.boxes = wp.empty(self.capacity, dtype=sb.Box, device=self.device)
        self.pending = wp.empty(self.capacity, dtype=wp.uint32, device=self.device)
        self.unresolved = wp.empty((self.capacity, 32), dtype=int, device=self.device)
        self.prepare_inputs = [points, tets, count, proved, self.shapes, self.boxes]
        self.query_inputs = [
            points,
            tets,
            count,
            tree,
            error,
            statistics,
            proved,
            proof_statistics,
            self.shapes,
            self.boxes,
            segment,
            self.pending,
            self.unresolved,
        ]
        wp.load_module(module=__name__, device=self.device, block_dim=block_dim)

    def run(self):
        wp.launch(
            prepare_unproved,
            self.capacity,
            inputs=self.prepare_inputs,
            device=self.device,
            block_dim=self.block_dim,
        )
        self.query()

    def query(self):
        self.pending.zero_()
        wp.launch(
            strict_unproved,
            32 * self.capacity,
            inputs=self.query_inputs,
            device=self.device,
            block_dim=self.block_dim,
        )
        wp.launch(
            exact_unproved,
            32 * self.capacity,
            inputs=self.query_inputs,
            device=self.device,
            block_dim=self.block_dim,
        )
