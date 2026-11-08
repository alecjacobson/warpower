"""Private exact proof of an entire tetrahedron's open interior being exterior."""
import warp as wp

from warpower import _integer as integer
from warpower import _filtered_predicates as predicates
from proven_exterior_filter_v2 import triangle
from warpower._surface_bvh import SurfaceTree

wp.set_module_options({'enable_backward': False, 'fast_math': False, 'fuse_fp': False})


@wp.func
def add_rows(a: integer.Row, b: integer.Row):
    r = integer.Row()
    r.x = integer.plus(a.x, b.x)
    r.y = integer.plus(a.y, b.y)
    r.z = integer.plus(a.z, b.z)
    r.w = integer.empty()
    return r


@wp.func(inline=False)
def centroid_relative(q: wp.vec3, p0: wp.vec3, p1: wp.vec3, p2: wp.vec3, p3: wp.vec3, base: int):
    # Exact 4*q - sum(p_i), scaled by 2^-base. No rounded centroid exists.
    a = add_rows(integer.relative(q, p0, base), integer.relative(q, p1, base))
    b = add_rows(integer.relative(q, p2, base), integer.relative(q, p3, base))
    return add_rows(a, b)


@wp.func(inline=False)
def centroid_ray(tree: SurfaceTree, p0: wp.vec3, p1: wp.vec3, p2: wp.vec3, p3: wp.vec3, direction: wp.vec3):
    count = int(0)
    for f in range(tree.n):
        q0 = wp.vec3(tree.points[tree.indices[3*f]])
        q1 = wp.vec3(tree.points[tree.indices[3*f+1]])
        q2 = wp.vec3(tree.points[tree.indices[3*f+2]])
        result = triangle(p0, p1, p2, p3, q0, q1, q2, direction)
        if result[1] != 0:
            return result
        count += result[0]
    return wp.vec2i(count, 0)


@wp.func(inline=False)
def exterior(tree: SurfaceTree, points: wp.array[wp.vec3], ids: wp.vec4i):
    p0 = wp.vec3(points[ids[0]])
    p1 = wp.vec3(points[ids[1]])
    p2 = wp.vec3(points[ids[2]])
    p3 = wp.vec3(points[ids[3]])
    orientation = predicates.orient3d(p0, p1, p2, p3)
    if orientation.valid == 0 or orientation.sign <= 0:
        return 0
    lo = wp.min(wp.min(p0, p1), wp.min(p2, p3))
    hi = wp.max(wp.max(p0, p1), wp.max(p2, p3))
    for f in range(tree.n):
        a = wp.vec3(tree.points[tree.indices[3*f]])
        b = wp.vec3(tree.points[tree.indices[3*f+1]])
        c = wp.vec3(tree.points[tree.indices[3*f+2]])
        qlo, qhi = wp.min(a, wp.min(b, c)), wp.max(a, wp.max(b, c))
        if qhi[0] < lo[0] or qhi[1] < lo[1] or qhi[2] < lo[2] or qlo[0] > hi[0] or qlo[1] > hi[1] or qlo[2] > hi[2]:
            continue
        positive = bool(False)
        negative = bool(False)
        for k in range(4):
            side = predicates.orient3d(a, b, c, wp.vec3(points[ids[k]]))
            if side.valid == 0:
                return 0
            positive = positive or side.sign > 0
            negative = negative or side.sign < 0
        if positive and negative:
            return 0
    # The positively oriented tet has nonzero volume. Separation from every
    # overlapping triangle plane makes its exact centroid a strict witness.
    r = centroid_ray(tree, p0, p1, p2, p3, wp.vec3(11.0, 23.0, 37.0))
    if r[1] == 1:
        r = centroid_ray(tree, p0, p1, p2, p3, wp.vec3(-17.0, 29.0, 43.0))
    if r[1] == 1:
        r = centroid_ray(tree, p0, p1, p2, p3, wp.vec3(31.0, -47.0, 19.0))
    return int(r[1] == 0 and r[0] == 0)


@wp.kernel
def prove(tree: SurfaceTree, points: wp.array[wp.vec3], tets: wp.array2d[int], result: wp.array[int]):
    i = wp.tid()
    result[i] = exterior(tree, points, wp.vec4i(tets[i, 0], tets[i, 1], tets[i, 2], tets[i, 3]))
