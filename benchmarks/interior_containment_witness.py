"""Experimental strict interior witness after conservative support-plane checks.

No production caller imports this module. A rounded centroid is only a proposal:
all four exact sub-tetrahedron orientations must prove strict interior membership.
"""

import warp as wp

from warpower import _filtered_predicates as predicates
from warpower._containment import contained, strict_material_side
from warpower._surface_bvh import SurfaceTree

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.func
def interior_witness(a: wp.vec3, b: wp.vec3, c: wp.vec3, d: wp.vec3):
    proposal = wp.vec3((wp.vec3d(a) + wp.vec3d(b) + wp.vec3d(c) + wp.vec3d(d)) * wp.float64(0.25))
    orientation = predicates.orient3d(a, b, c, d)
    valid = int(orientation.valid != 0 and orientation.sign != 0)
    for k in range(4):
        x, y, z, w = a, b, c, d
        if k == 0:
            x = proposal
        elif k == 1:
            y = proposal
        elif k == 2:
            z = proposal
        else:
            w = proposal
        side = predicates.orient3d(x, y, z, w)
        valid &= int(side.valid != 0 and side.sign == orientation.sign)
    return proposal, valid


@wp.func(inline=False)
def inspect_containment(tree: SurfaceTree, points: wp.array[wp.vec3], ids: wp.vec4i):
    # Columns: contained, terminal stage, surviving strict-vertex mask, membership.
    # Stages: 0 support rejection; 1 vertex query; 2 no strict rounded witness;
    # 3 interior query. A stage is diagnostic, not a new geometry status.
    lo, hi = wp.vec3(points[ids[0]]), wp.vec3(points[ids[0]])
    for k in range(1, 4):
        lo, hi = wp.min(lo, points[ids[k]]), wp.max(hi, points[ids[k]])
    strict = int(15)
    for f in range(tree.n):
        a, b, c = (
            tree.points[tree.indices[3 * f]],
            tree.points[tree.indices[3 * f + 1]],
            tree.points[tree.indices[3 * f + 2]],
        )
        qlo, qhi = wp.min(a, wp.min(b, c)), wp.max(a, wp.max(b, c))
        if (
            qhi[0] < lo[0]
            or qhi[1] < lo[1]
            or qhi[2] < lo[2]
            or qlo[0] > hi[0]
            or qlo[1] > hi[1]
            or qlo[2] > hi[2]
        ):
            continue
        for k in range(4):
            side = predicates.orient3d(a, b, c, points[ids[k]])
            if side.valid == 0 or side.sign > 0:
                return wp.vec4i(0, 0, strict, 128)
            if side.sign == 0:
                strict &= ~(1 << k)
    for k in range(4):
        if (strict & (1 << k)) != 0:
            membership = strict_material_side(tree, points[ids[k]])
            if membership != 128:
                return wp.vec4i(int(membership == 1), 1, strict, membership)
    q, valid = interior_witness(points[ids[0]], points[ids[1]], points[ids[2]], points[ids[3]])
    if valid == 0:
        return wp.vec4i(0, 2, strict, 128)
    membership = strict_material_side(tree, q)
    return wp.vec4i(int(membership == 1), 3, strict, membership)


@wp.kernel
def compare_containment(
    tree: SurfaceTree,
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    count: wp.array[int],
    baseline: wp.array[int],
    candidate: wp.array2d[int],
):
    i = wp.tid()
    baseline[i] = 0
    for k in range(4):
        candidate[i, k] = 0
    if i < count[0]:
        ids = wp.vec4i(tets[i, 0], tets[i, 1], tets[i, 2], tets[i, 3])
        baseline[i] = contained(tree, points, ids)
        result = inspect_containment(tree, points, ids)
        for k in range(4):
            candidate[i, k] = result[k]


@wp.kernel
def inspect_witness(points: wp.array2d[wp.vec3], witness: wp.array[wp.vec3], valid: wp.array[int]):
    i = wp.tid()
    q, ok = interior_witness(points[i, 0], points[i, 1], points[i, 2], points[i, 3])
    witness[i] = q
    valid[i] = ok
