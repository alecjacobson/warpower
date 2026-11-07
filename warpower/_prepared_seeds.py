"""Current-input sphere geometry shared by cooperative seed-query lanes.

Only the private VTM reuse filter opts in. Every run recomputes all live geometry
before querying it; these arrays never carry a proof across input changes.
"""

import warp as wp

from . import _sphere_bounds as sb
from ._bvh import Tree
from ._delaunay_certificate import Sphere, outside, sphere, symbolic_sphere

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.kernel
def prepare(
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    count: wp.array[int],
    error: wp.array[int],
    shapes: wp.array[Sphere],
    boxes: wp.array[sb.Box],
):
    i = wp.tid()
    if error[0] != 0 or i >= count[0]:
        return
    ids = wp.vec4i(tets[i, 0], tets[i, 1], tets[i, 2], tets[i, 3])
    shape = sphere(points[ids[0]], points[ids[1]], points[ids[2]], points[ids[3]])
    shapes[i] = shape
    boxes[i] = sb.bounds(shape.origin, shape.nx, shape.ny, shape.nz, shape.determinant)


@wp.kernel
def check(
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    count: wp.array[int],
    tree: Tree,
    live: wp.array[int],
    error: wp.array[int],
    shapes: wp.array[Sphere],
    boxes: wp.array[sb.Box],
):
    tid = wp.tid()
    i, lane = tid // 32, tid % 32
    if error[0] != 0 or i >= count[0]:
        return
    ids = wp.vec4i(tets[i, 0], tets[i, 1], tets[i, 2], tets[i, 3])
    shape = shapes[i]
    box = boxes[i]
    # Match _facet_seeds.check_seeds, including its exact/symbolic fallback.
    # Keeping the original kernel separate preserves ordinary construction.
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


class PreparedSeedQueries:
    """Fixed setup storage: 80-byte sphere and 28-byte box per candidate slot."""

    def __init__(self, points, tets, count, tree, live, error, *, block_dim):
        self.device, self.capacity, self.block_dim = points.device, len(tets), block_dim
        self.shapes = wp.empty(self.capacity, dtype=Sphere, device=self.device)
        self.boxes = wp.empty(self.capacity, dtype=sb.Box, device=self.device)
        self._prepare_inputs = [points, tets, count, error, self.shapes, self.boxes]
        self._check_inputs = [points, tets, count, tree, live, error, self.shapes, self.boxes]
        wp.load_module(module=__name__, device=self.device, block_dim=self.block_dim)

    def run(self):
        wp.launch(
            prepare,
            self.capacity,
            inputs=self._prepare_inputs,
            block_dim=self.block_dim,
            device=self.device,
        )
        wp.launch(
            check,
            32 * self.capacity,
            inputs=self._check_inputs,
            block_dim=self.block_dim,
            device=self.device,
        )
