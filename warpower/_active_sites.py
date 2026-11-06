"""Optional active-prefix inputs for the developing captured VTM core."""

import warp as wp

wp.set_module_options({"enable_backward": False, "fast_math": False})


@wp.kernel
def prepare_mesh(points: wp.array[wp.vec3], active: wp.array[int], mesh_points: wp.array[wp.vec3]):
    i = wp.tid()
    value = wp.vec3(0.0)
    if active[0] >= 4 and active[0] <= points.shape[0]:
        # Lowest-index duplicate ownership removes every inactive mesh slot.
        value = points[0]
        if i < active[0]:
            value = points[i]
    mesh_points[i] = value


@wp.kernel
def validate(active: wp.array[int], capacity: int, error: wp.array[int], ready: wp.array[int]):
    error[0] = 0
    if active[0] < 4 or active[0] > capacity:
        error[0] = 8
    ready[0] = int(error[0] == 0)


@wp.kernel
def propose_active(
    original: wp.array[wp.vec3],
    points: wp.array[wp.vec3],
    delta: wp.array[wp.vec3d],
    edge: wp.array[wp.float64],
    alpha: wp.array[wp.float64],
    maximum: wp.array[wp.float64],
    active: wp.array[int],
):
    i = wp.tid()
    if i >= active[0]:
        return
    p = wp.vec3(wp.vec3d(original[i]) + alpha[0] * delta[i])
    points[i] = p
    movement = wp.length(wp.vec3d(p) - wp.vec3d(original[i])) / edge[i]
    if not wp.isfinite(movement):
        movement = wp.float64(1.0e300)
    wp.atomic_max(maximum, 0, movement)
