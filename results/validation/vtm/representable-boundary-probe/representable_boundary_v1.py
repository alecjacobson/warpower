"""Private midpoint-only boundary proposal; normal VTM transactions stay intact."""
import types

import warp as wp

wp.set_module_options({'enable_backward': False, 'fast_math': False})


@wp.struct
class Proposal:
    point: wp.vec3
    changed: int


@wp.func
def rounded_boundary(original: wp.vec3, requested: wp.vec3d, target: wp.vec3,
                     cap: wp.float64):
    result = Proposal()
    result.point = wp.vec3(requested)
    result.changed = 0
    candidate = result.point
    for axis in range(3):
        p = wp.float64(original[axis])
        t = wp.float64(target[axis])
        q = requested[axis]
        if (candidate[axis] == original[axis] and p != t
                and wp.isfinite(p) and wp.isfinite(t) and wp.isfinite(q)
                and q - p == t - q):
            candidate[axis] = target[axis]
            result.changed += 1
    if result.changed != 0:
        movement = wp.length(wp.vec3d(candidate) - wp.vec3d(original))
        if wp.isfinite(cap) and cap > wp.float64(0.0) and wp.isfinite(movement) and movement <= cap:
            result.point = candidate
        else:
            result.changed = 0
    return result


@wp.kernel
def freeze_boundary(points: wp.array[wp.vec3], targets: wp.array[wp.vec3],
                    roles: wp.array[int], frozen: wp.array[wp.vec3]):
    i = wp.tid()
    frozen[i] = points[i]
    if roles[i] == 1 or roles[i] == 2:
        frozen[i] = targets[i]


@wp.func
def proposal_at(i: int, original: wp.array[wp.vec3], points: wp.array[wp.vec3],
                delta: wp.array[wp.vec3d], edge: wp.array[wp.float64],
                alpha: wp.array[wp.float64], maximum: wp.array[wp.float64],
                targets: wp.array[wp.vec3], limit: wp.float64,
                flags: wp.array[int], stats: wp.array[int]):
    q = wp.vec3d(original[i]) + alpha[0] * delta[i]
    result = rounded_boundary(original[i], q, targets[i], limit * edge[i])
    points[i] = result.point
    flags[i] = result.changed
    if result.changed != 0:
        wp.atomic_add(stats, 0, result.changed)
        wp.atomic_add(stats, 1, 1)
    movement = wp.length(wp.vec3d(result.point) - wp.vec3d(original[i])) / edge[i]
    if not wp.isfinite(movement):
        movement = wp.float64(1.0e300)
    wp.atomic_max(maximum, 0, movement)


@wp.kernel
def propose(original: wp.array[wp.vec3], points: wp.array[wp.vec3],
            delta: wp.array[wp.vec3d], edge: wp.array[wp.float64],
            alpha: wp.array[wp.float64], maximum: wp.array[wp.float64],
            targets: wp.array[wp.vec3], limit: wp.float64,
            flags: wp.array[int], stats: wp.array[int]):
    proposal_at(wp.tid(), original, points, delta, edge, alpha, maximum, targets, limit, flags, stats)


@wp.kernel
def propose_active(original: wp.array[wp.vec3], points: wp.array[wp.vec3],
                   delta: wp.array[wp.vec3d], edge: wp.array[wp.float64],
                   alpha: wp.array[wp.float64], maximum: wp.array[wp.float64],
                   active: wp.array[int], targets: wp.array[wp.vec3], limit: wp.float64,
                   flags: wp.array[int], stats: wp.array[int]):
    i = wp.tid()
    flags[i] = 0
    if i < active[0]:
        proposal_at(i, original, points, delta, edge, alpha, maximum, targets, limit, flags, stats)


@wp.kernel
def record_accepted(flags: wp.array[int], stats: wp.array[int]):
    i = wp.tid()
    if flags[i] != 0:
        wp.atomic_add(stats, 2, flags[i])
        wp.atomic_add(stats, 3, 1)


def install(core):
    """Patch one preallocated private pilot workspace, outside any capture."""
    assert not wp.get_stream(core.device).is_capturing
    core._tie_targets = wp.empty_like(core.points)
    core._tie_flags = wp.zeros(core.n, dtype=int, device=core.device)
    core._tie_stats = wp.zeros(4, dtype=int, device=core.device)
    core._propose = propose if core.active_count is None else propose_active
    core._propose_args = ([] if core.active_count is None else [core.active_count]) + [
        core._tie_targets, core.step_limit, core._tie_flags, core._tie_stats]
    launch = core._launch
    accept = core._accept

    def augmented_launch(self, kernel, dim, args, *, block_dim=256):
        launch(kernel, dim, args, block_dim=block_dim)
        if kernel is self._directions:
            # Boundary evaluation mutates its targets/roles after each trial.
            # Freeze the starting assignment for every backtracking attempt.
            launch(freeze_boundary, self.n, [self.points, self.boundary.targets,
                   self.boundary.roles, self._tie_targets])

    def augmented_accept(self):
        accept()
        launch(record_accepted, self.n, [self._tie_flags, self._tie_stats])

    core._launch = types.MethodType(augmented_launch, core)
    core._accept = types.MethodType(augmented_accept, core)
    wp.load_module(module=__name__, device=core.device)
    with wp.ScopedCapture(device=core.device) as capture:
        core._tie_stats.zero_()
        core._record()
    core.graph = capture.graph
    return core
