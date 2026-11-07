"""Private #170 prepared point-expression alternatives; no production change."""
import warp as wp
from warpower import _filtered_predicates as fp, _analytic_predicates as af, _sphere_bounds as sb
from warpower._bvh import Tree
from warpower._delaunay_certificate import Sphere, outside, symbolic_sphere
wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})

@wp.func
def point_value(shape:Sphere,p:wp.vec3):
    x=fp.difference(p[0],shape.origin[0])
    y=fp.difference(p[1],shape.origin[1])
    z=fp.difference(p[2],shape.origin[2])
    return fp.minus(
        fp.plus(fp.plus(fp.times(x,shape.nx),fp.times(y,shape.ny)),fp.times(z,shape.nz)),
        fp.times(shape.determinant,fp.lift(x,y,z,0.0,0.0)),
    )

@wp.func
def point_outside(shape:Sphere,p:wp.vec3):
    value=point_value(shape,p)
    return wp.isfinite(value.value) and wp.isfinite(value.error) and value.value < -value.error

@wp.kernel
def check_point(
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
        if sb.outside(box, p, p) or point_outside(shape, p):
            continue
        result = symbolic_sphere(points, ids, q)
        if result.valid == 0:
            wp.atomic_or(error, 0, 2)
            wp.atomic_and(live, i, 0)
        elif result.sign >= 0:
            wp.atomic_and(live, i, 0)


@wp.kernel
def check_analytic_first(
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
        value = af.sphere_filter(points[ids[0]], points[ids[1]], points[ids[2]], points[ids[3]], p)
        if af.strict(value):
            if value.value > wp.float64(0.0):
                wp.atomic_and(live, i, 0)
            continue
        # Keep the prepared cofactor filter before exact fallback: the two
        # expressions have different conditioning on thin cells/far queries.
        if outside(shape, p, p):
            continue
        result = symbolic_sphere(points, ids, q)
        if result.valid == 0:
            wp.atomic_or(error, 0, 2)
            wp.atomic_and(live, i, 0)
        elif result.sign >= 0:
            wp.atomic_and(live, i, 0)

