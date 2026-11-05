"""GPU validation for values and CSR hints."""

import warp as wp


@wp.kernel(enable_backward=False)
def validate(
    points: wp.array[wp.vec3],
    weights: wp.array[float],
    hints: wp.array[int],
    offsets: wp.array[int],
    use_hints: bool,
    error: wp.array[int],
):
    i = wp.tid()
    n = points.shape[0]
    if i < n:
        p = points[i]
        for axis in range(3):
            if not wp.isfinite(p[axis]):
                wp.atomic_or(error, 0, 1)
        if not wp.isfinite(weights[i]):
            wp.atomic_or(error, 0, 2)
    if use_hints:
        if i <= n:
            at = offsets[i]
            if at < 0 or at > hints.shape[0]:
                wp.atomic_or(error, 0, 4)
            if i == 0 and at != 0:
                wp.atomic_or(error, 0, 4)
            if i == n and at != hints.shape[0]:
                wp.atomic_or(error, 0, 4)
            if i < n and at > offsets[i + 1]:
                wp.atomic_or(error, 0, 4)
        if i < hints.shape[0] and (hints[i] < -1 or hints[i] >= n):
            wp.atomic_or(error, 0, 8)


def check(points, weights, hints, offsets, use_hints):
    error = wp.zeros(1, dtype=int, device=points.device)
    wp.launch(
        validate,
        max(len(points) + 1, len(hints)),
        inputs=[points, weights, hints, offsets, use_hints, error],
        device=points.device,
    )
    code = int(error.numpy()[0])
    if code & 1:
        raise ValueError("points must contain only finite values")
    if code & 2:
        raise ValueError("weights must contain only finite values")
    if code & 4:
        raise ValueError("hint offsets must form a valid CSR partition of initial_guesses")
    if code & 8:
        raise ValueError("hint indices must be in [0, N), or -1 for a skipped hint")
