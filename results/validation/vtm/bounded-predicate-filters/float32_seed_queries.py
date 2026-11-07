"""Private #180 first-stage binary32 filter before prepared seed arithmetic."""
import warp as wp
import bounded_float32_filters as low
from warpower import _sphere_bounds as sb
from warpower._bvh import Tree
from warpower._delaunay_certificate import Sphere, symbolic_sphere
from warpower._analytic_predicates import strict
from warpower._prepared_seeds import point_value, prepare
from warpower._grouped_seeds import GroupedSeedQueries, group_bounds

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})

@wp.kernel
def check_float32_groups(
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    count: wp.array[int],
    tree: Tree,
    live: wp.array[int],
    error: wp.array[int],
    shapes: wp.array[Sphere],
    boxes: wp.array[sb.Box],
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
):
    tid = wp.tid()
    i, lane = tid // 32, tid % 32
    if error[0] != 0 or i >= count[0]:
        return
    ids = wp.vec4i(tets[i, 0], tets[i, 1], tets[i, 2], tets[i, 3])
    a, b, c, d = points[ids[0]], points[ids[1]], points[ids[2]], points[ids[3]]
    shape = shapes[i]
    box = boxes[i]
    for group in range(lower.shape[0]):
        # All 32 lanes see the same group box. An invalid sphere enclosure
        # cannot skip any group; all surviving queries retain the exact policy.
        if sb.outside(box, lower[group], upper[group]):
            continue
        rank = 32 * group + lane
        if rank >= points.shape[0]:
            continue
        q = tree.ids[rank]
        if tree.owners[q] != q or q == ids[0] or q == ids[1] or q == ids[2] or q == ids[3]:
            continue
        p = points[q]
        if sb.outside(box, p, p):
            continue
        fast = low.sphere_filter(a, b, c, d, p)
        if strict(fast):
            if fast.value > wp.float64(0.0):
                wp.atomic_and(live, i, 0)
            continue
        value = point_value(shape, p)
        if (
            wp.isfinite(value.value)
            and wp.isfinite(value.error)
            and wp.abs(value.value) > value.error
        ):
            if value.value > wp.float64(0.0):
                wp.atomic_and(live, i, 0)
            continue
        result = symbolic_sphere(points, ids, q)
        if result.valid == 0:
            wp.atomic_or(error, 0, 2)
            wp.atomic_and(live, i, 0)
        elif result.sign >= 0:
            wp.atomic_and(live, i, 0)


class Float32GroupedQueries(GroupedSeedQueries):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        wp.load_module(module=__name__, device=self.device, block_dim=self.block_dim)

    def run(self):
        wp.launch(group_bounds, len(self.lower), inputs=self._group_inputs,
                  block_dim=self.block_dim, device=self.device)
        wp.launch(prepare, self.capacity, inputs=self._prepare_inputs,
                  block_dim=self.block_dim, device=self.device)
        wp.launch(check_float32_groups, 32*self.capacity, inputs=self._query_inputs,
                  block_dim=self.block_dim, device=self.device)


def install():
    from warpower import _grouped_seeds
    _grouped_seeds.GroupedSeedQueries = Float32GroupedQueries
