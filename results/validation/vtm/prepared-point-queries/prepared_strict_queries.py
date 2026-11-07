"""Private #170: use the same point enclosure for both strict signs."""
import warp as wp
from warpower import _sphere_bounds as sb
from warpower._bvh import Tree
from warpower._delaunay_certificate import Sphere, symbolic_sphere
from prepared_point_queries import point_value
wp.set_module_options({'enable_backward': False, 'fast_math': False, 'fuse_fp': False})


@wp.kernel
def check_strict(
    points: wp.array[wp.vec3], tets: wp.array2d[int], count: wp.array[int], tree: Tree,
    live: wp.array[int], error: wp.array[int], shapes: wp.array[Sphere], boxes: wp.array[sb.Box],
    stop_rejected: bool,
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
        if stop_rejected and wp.atomic_add(live, i, 0) == 0:
            return
        value = point_value(shape, p)
        if wp.isfinite(value.value) and wp.isfinite(value.error) and wp.abs(value.value) > value.error:
            if value.value > wp.float64(0.0):
                wp.atomic_and(live, i, 0)
            continue
        result = symbolic_sphere(points, ids, q)
        if result.valid == 0:
            wp.atomic_or(error, 0, 2)
            wp.atomic_and(live, i, 0)
        elif result.sign >= 0:
            wp.atomic_and(live, i, 0)
