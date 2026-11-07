"""Private #172 box-only traversals with one or eight query workers."""
import warp as wp
from warpower import _sphere_bounds as sb
from warpower._bvh import Tree
from warpower._delaunay_certificate import Sphere, symbolic_sphere
from warpower._prepared_seeds import point_value

wp.set_module_options({'enable_backward': False, 'fast_math': False, 'fuse_fp': False})


@wp.kernel
def check_tree_1(
    points: wp.array[wp.vec3], tets: wp.array2d[int], count: wp.array[int], tree: Tree,
    live: wp.array[int], error: wp.array[int], shapes: wp.array[Sphere], boxes: wp.array[sb.Box],
):
    tid = wp.tid()
    i, lane = tid // 1, tid % 1
    if error[0] != 0 or i >= count[0]:
        return
    # The low 0 path bits partition the current hierarchy. An early leaf
    # belongs only to the lane whose unused path bits are zero.
    root = int(1)
    for level in range(0):
        if tree.count[root] > 0:
            if (lane >> level) != 0:
                return
            break
        if ((lane >> level) & 1) == 0:
            root = tree.left[root]
        else:
            root = tree.right[root]
    ids = wp.vec4i(tets[i, 0], tets[i, 1], tets[i, 2], tets[i, 3])
    shape = shapes[i]
    box = boxes[i]
    stack = wp.zeros(shape=128, dtype=int)
    stack[0] = root
    size = int(1)
    while size > 0:
        size -= 1
        node = stack[size]
        if sb.outside(box, tree.lower[node], tree.upper[node]):
            continue
        if tree.count[node] > 0:
            for k in range(tree.start[node], tree.start[node] + tree.count[node]):
                q = tree.ids[k]
                if tree.owners[q] != q or q == ids[0] or q == ids[1] or q == ids[2] or q == ids[3]:
                    continue
                p = points[q]
                if sb.outside(box, p, p):
                    continue
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
        else:
            if size > 125:
                wp.atomic_or(error, 0, 16)
                wp.atomic_and(live, i, 0)
                return
            stack[size] = tree.left[node]
            stack[size + 1] = tree.right[node]
            size += 2


@wp.kernel
def check_tree_8(
    points: wp.array[wp.vec3], tets: wp.array2d[int], count: wp.array[int], tree: Tree,
    live: wp.array[int], error: wp.array[int], shapes: wp.array[Sphere], boxes: wp.array[sb.Box],
):
    tid = wp.tid()
    i, lane = tid // 8, tid % 8
    if error[0] != 0 or i >= count[0]:
        return
    # The low 3 path bits partition the current hierarchy. An early leaf
    # belongs only to the lane whose unused path bits are zero.
    root = int(1)
    for level in range(3):
        if tree.count[root] > 0:
            if (lane >> level) != 0:
                return
            break
        if ((lane >> level) & 1) == 0:
            root = tree.left[root]
        else:
            root = tree.right[root]
    ids = wp.vec4i(tets[i, 0], tets[i, 1], tets[i, 2], tets[i, 3])
    shape = shapes[i]
    box = boxes[i]
    stack = wp.zeros(shape=128, dtype=int)
    stack[0] = root
    size = int(1)
    while size > 0:
        size -= 1
        node = stack[size]
        if sb.outside(box, tree.lower[node], tree.upper[node]):
            continue
        if tree.count[node] > 0:
            for k in range(tree.start[node], tree.start[node] + tree.count[node]):
                q = tree.ids[k]
                if tree.owners[q] != q or q == ids[0] or q == ids[1] or q == ids[2] or q == ids[3]:
                    continue
                p = points[q]
                if sb.outside(box, p, p):
                    continue
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
        else:
            if size > 125:
                wp.atomic_or(error, 0, 16)
                wp.atomic_and(live, i, 0)
                return
            stack[size] = tree.left[node]
            stack[size + 1] = tree.right[node]
            size += 2


