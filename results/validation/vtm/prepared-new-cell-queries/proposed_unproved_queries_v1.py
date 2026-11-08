"""Share current-cell sphere geometry and traverse current original-site bounds.

Private completion stage: finite sites, canonical owners, valid cell indices and
the rebuilt current segment are required. Scratch is borrowed from the seed
query stage after it finishes; each invocation prepares current geometry anew.
"""

import warp as wp

from warpower import _sphere_bounds as sb
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
def check_prepared_unproved(
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
        result = symbolic_sphere(points, ids, q)
        if result.valid == 0:
            wp.atomic_or(error, 0, 2)
        elif result.sign >= 0:
            wp.atomic_or(error, 0, 8)
    if node != 0:
        # A valid current segment cannot exhaust its full-tree visit budget.
        wp.atomic_or(error, 0, 2)
    wp.atomic_add(statistics, 0, queries)


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
        shapes,
        boxes,
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
        if len(shapes) < self.capacity or len(boxes) < self.capacity:
            raise ValueError("Borrowed sphere geometry must cover every completion cell slot")
        if shapes.device != self.device or boxes.device != self.device:
            raise ValueError("Borrowed sphere geometry must share the query device")
        self.shapes, self.boxes, self.segment = shapes, boxes, segment
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
        ]
        wp.load_module(module=__name__, device=self.device, block_dim=block_dim)

    def run(self, tree=None):
        current = self.query_inputs[3] if tree is None else tree
        if current.ids.ptr != self.segment.ids.ptr or current.n != self.segment.n:
            raise ValueError("Completion query tree changed its segment association")
        wp.launch(
            prepare_unproved,
            self.capacity,
            inputs=self.prepare_inputs,
            device=self.device,
            block_dim=self.block_dim,
        )
        wp.launch(
            check_prepared_unproved,
            32 * self.capacity,
            inputs=[*self.query_inputs[:3], current, *self.query_inputs[4:]],
            device=self.device,
            block_dim=self.block_dim,
        )
