"""One-point density quadrature for the graded VTM update (Alliez et al., Eq8).

The uniform standalone ODT kernel stays separate so optional sizing does not
change its generated code or add field evaluation to its execution path.
"""

import warp as wp

from ._sizing import SizingData, sizing_value

wp.set_module_options({"enable_backward": False, "fast_math": False})


@wp.kernel
def accumulate_graded_tets(
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    count: wp.array[int],
    mesh_error: wp.array[int],
    invalid: wp.array[int],
    mass: wp.array[wp.float64],
    moment: wp.array[wp.vec3d],
    edge: wp.array[wp.float64],
    energy: wp.array[wp.float64],
    volume: wp.array[wp.float64],
    minimum_quality: wp.array[wp.float64],
    directions: bool,
    sizing: SizingData,
):
    t = wp.tid()
    if t >= count[0] or mesh_error[0] != 0:
        return
    i0, i1, i2, i3 = tets[t, 0], tets[t, 1], tets[t, 2], tets[t, 3]
    p = wp.vec3d(points[i0])
    a, b, c = wp.vec3d(points[i1]) - p, wp.vec3d(points[i2]) - p, wp.vec3d(points[i3]) - p
    cross_bc, cross_ca, cross_ab = wp.cross(b, c), wp.cross(c, a), wp.cross(a, b)
    det = wp.dot(a, cross_bc)
    v = det / wp.float64(6.0)
    if v <= wp.float64(0.0) or not wp.isfinite(v):
        wp.atomic_or(invalid, 0, 256)
        return
    squared_edges = (
        wp.dot(a, a)
        + wp.dot(b, b)
        + wp.dot(c, c)
        + wp.dot(a - b, a - b)
        + wp.dot(a - c, a - c)
        + wp.dot(b - c, b - c)
    )
    centroid = p + (a + b + c) / wp.float64(4.0)
    mu = sizing_value(sizing, centroid)
    if mu <= wp.float64(0.0) or not wp.isfinite(mu):
        wp.atomic_or(invalid, 0, 256)
        return
    density = wp.float64(1.0) / (mu * mu * mu)
    weighted_volume = v * density
    e = weighted_volume * squared_edges / wp.float64(20.0)
    if not wp.isfinite(e):
        wp.atomic_or(invalid, 0, 256)
        return
    wp.atomic_add(energy, 0, e)
    wp.atomic_add(volume, 0, v)
    if directions:
        # V * (circumcenter - p), without division by det. Near-flat tets can
        # have huge circumcenters while this product remains well conditioned.
        vc = (
            wp.dot(a, a) * cross_bc + wp.dot(b, b) * cross_ca + wp.dot(c, c) * cross_ab
        ) / wp.float64(12.0)
        indices = wp.vec4i(i0, i1, i2, i3)
        for k in range(4):
            i = indices[k]
            pi = wp.vec3d(points[i])
            wp.atomic_add(mass, i, weighted_volume)
            wp.atomic_add(moment, i, density * (vc + v * (p - pi)))
            shortest = wp.float64(1.0e300)
            for j in range(4):
                if j != k:
                    shortest = wp.min(shortest, wp.length(pi - wp.vec3d(points[indices[j]])))
            wp.atomic_min(edge, i, shortest)
        # 3 * inradius / circumradius. Evaluating the reciprocal radius avoids
        # constructing the potentially very large circumcenter itself.
        twice_area = (
            wp.length(cross_bc)
            + wp.length(cross_ca)
            + wp.length(cross_ab)
            + wp.length(wp.cross(b - a, c - a))
        )
        q = wp.float64(18.0) * v * v / (twice_area * wp.length(vc))
        wp.atomic_min(minimum_quality, 0, wp.min(wp.float64(1.0), q))
