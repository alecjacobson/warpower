"""Private prepared-geometry filter with a separate uncertain-sign fallback."""
import warp as wp

from warpower import _filtered_predicates as fp, _sphere_bounds as sb
from warpower._bvh import Tree
from warpower._delaunay_certificate import Sphere, interval, outside, symbolic_sphere
from warpower._facet_seeds import CertifiedSeeds, check_seeds
from seed_filter_prepared import prepare_bounds

wp.set_module_options({'enable_backward': False, 'fast_math': False, 'fuse_fp': False})


@wp.func
def query_ball(value: Sphere, p: wp.vec3):
    x = fp.minus(interval(p[0], p[0]), fp.scalar(wp.float64(value.origin[0])))
    y = fp.minus(interval(p[1], p[1]), fp.scalar(wp.float64(value.origin[1])))
    z = fp.minus(interval(p[2], p[2]), fp.scalar(wp.float64(value.origin[2])))
    return fp.minus(
        fp.plus(fp.plus(fp.times(x, value.nx), fp.times(y, value.ny)), fp.times(z, value.nz)),
        fp.times(value.determinant, fp.lift(x, y, z, 0.0, 0.0)),
    )


@wp.kernel
def bounded_queries(
    points: wp.array[wp.vec3], tets: wp.array2d[int], count: wp.array[int], tree: Tree,
    live: wp.array[int], error: wp.array[int], shapes: wp.array[Sphere],
    boxes: wp.array[sb.Box], pending: wp.array[int],
):
    tid = wp.tid()
    i, lane = tid // 32, tid % 32
    if error[0] != 0 or i >= count[0]:
        return
    ids = wp.vec4i(tets[i, 0], tets[i, 1], tets[i, 2], tets[i, 3])
    shape = shapes[i]
    box = boxes[i]
    for q in range(lane, points.shape[0], 32):
        if tree.owners[q] != q or q == ids[0] or q == ids[1] or q == ids[2] or q == ids[3]:
            continue
        p = points[q]
        if sb.outside(box, p, p):
            continue
        value = query_ball(shape, p)
        if wp.isfinite(value.value) and wp.isfinite(value.error):
            if value.value < -value.error:
                continue
            if value.value > value.error:
                wp.atomic_and(live, i, 0)
                continue
        wp.atomic_or(pending, i, 1)


@wp.kernel
def uncertain_queries(
    points: wp.array[wp.vec3], tets: wp.array2d[int], count: wp.array[int], tree: Tree,
    live: wp.array[int], error: wp.array[int], shapes: wp.array[Sphere],
    boxes: wp.array[sb.Box], pending: wp.array[int],
):
    tid = wp.tid()
    i, lane = tid // 32, tid % 32
    if error[0] != 0 or i >= count[0]:
        return
    if pending[i] == 0:
        return
    ids = wp.vec4i(tets[i, 0], tets[i, 1], tets[i, 2], tets[i, 3])
    shape = shapes[i]
    box = boxes[i]
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


class StagedSeeds(CertifiedSeeds):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shapes = wp.empty(len(self.source), dtype=Sphere, device=self.device)
        self.boxes = wp.empty(len(self.source), dtype=sb.Box, device=self.device)
        self.pending = wp.empty(len(self.source), dtype=int, device=self.device)
        wp.load_module(module=__name__, device=self.device, block_dim=self._block_dim)
        wp.load_module(module='seed_filter_prepared', device=self.device, block_dim=self._block_dim)

    def run(self):
        original = wp.launch

        def launch(kernel, *args, **kwargs):
            if kernel is check_seeds:
                p, t, n, tree, live, error = kwargs['inputs']
                self.pending.zero_()
                original(prepare_bounds, len(self.unique), inputs=[p, t, n, error, self.shapes, self.boxes], block_dim=self._block_dim, device=self.device)
                kwargs = dict(kwargs, inputs=kwargs['inputs'] + [self.shapes, self.boxes, self.pending])
                original(bounded_queries, *args, **kwargs)
                return original(uncertain_queries, *args, **kwargs)
            return original(kernel, *args, **kwargs)

        wp.launch = launch
        try:
            super().run()
        finally:
            wp.launch = original
