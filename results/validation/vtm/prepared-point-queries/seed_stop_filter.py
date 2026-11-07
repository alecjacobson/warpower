"""Private #169: stop rejected seed lanes after the cheap sphere box test.

Internal preconditions match CertifiedSeeds: finite original coordinates,
current canonical ownership and positive-volume canonical candidate cells.
Public input validation and per-run live resets remain outside this kernel.
"""
import warp as wp
from warpower import _sphere_bounds as sb
from warpower._bvh import Tree
from warpower._delaunay_certificate import Sphere, outside, symbolic_sphere
wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})

@wp.kernel
def check_stopping(
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
        if sb.outside(box, p, p):
            continue
        # Live is reset for this invocation. A certified rejection ends only
        # this candidate lane; retained cells still examine every query.
        if wp.atomic_add(live, i, 0) == 0:
            return
        if outside(shape, p, p):
            continue
        result = symbolic_sphere(points, ids, q)
        if result.valid == 0:
            wp.atomic_or(error, 0, 2)
            wp.atomic_and(live, i, 0)
        elif result.sign >= 0:
            wp.atomic_and(live, i, 0)

