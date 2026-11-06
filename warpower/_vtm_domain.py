"""Material-domain trust regions for otherwise unassigned VTM sites."""

import warp as wp

from ._surface_bvh import SurfaceTree, closest_point, material_side

wp.set_module_options({"enable_backward": False, "fast_math": False})


@wp.kernel
def constrain(
    tree: SurfaceTree,
    points: wp.array[wp.vec3],
    roles: wp.array[int],
    direction: wp.array[wp.vec3d],
    edge: wp.array[wp.float64],
    step_limit: wp.float64,
    damping: wp.float64,
    error: wp.array[int],
    interior: wp.array[int],
):
    i = wp.tid()
    delta = direction[i]
    # Inactive slots and unselected sites have zero directions. Preserve that
    # mask even when an unselected site happens to lie outside the material.
    if roles[i] != 0:
        interior[i] = 0
        return
    if wp.length_sq(delta) == wp.float64(0.0):
        return
    p = wp.vec3d(points[i])
    side = int(1)
    if interior[i] == 0:
        side = material_side(tree, p)
        interior[i] = int(side == 1)
    margin = wp.float64(8.0e-7) * (wp.length(p) + edge[i])
    length = wp.length(delta)
    limit = wp.float64(1.0e150)
    if side == 1:
        # A more distant surface cannot restrict this direction. Expand the
        # cutoff slightly so its rounding cannot skip a limiting triangle.
        limit = wp.min(
            limit,
            wp.max(
                wp.float64(1.0e-150),
                (length * damping / wp.float64(0.9) + margin) * wp.float64(1.000000000001),
            ),
        )
    hit = closest_point(tree, p, limit)
    if side == 1 and hit.face < 0 and limit < wp.float64(1.0e150):
        return
    if hit.face < 0 or not wp.isfinite(hit.distance2):
        wp.atomic_or(error, 0, 64)
        direction[i] = wp.vec3d(0.0)
        return
    if hit.distance2 == wp.float64(0.0):
        direction[i] = wp.vec3d(0.0)
        return
    if side == 1:
        # The open ball up to the nearest surface lies within the same material
        # component. Account for stored float32 positions, then reserve 10% of
        # the remaining distance. Later backtracking only reduces this step.
        maximum = (
            wp.float64(0.9) * wp.max(wp.float64(0.0), wp.sqrt(hit.distance2) - margin) / damping
        )
        if length > maximum:
            delta *= maximum / length
    elif side == 0:
        # Evolving boundary centroids may lie outside a nonconvex material and
        # later lose their quadrature assignment. Return toward the surface,
        # instead of applying an interior-star formula to such a site.
        delta = hit.point - p
        length = wp.length(delta)
        if length > step_limit * edge[i]:
            delta *= step_limit * edge[i] / length
    else:
        wp.atomic_or(error, 0, 64)
        delta = wp.vec3d(0.0)
    direction[i] = delta
