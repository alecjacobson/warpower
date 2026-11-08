"""Optional, bounded partial rollback after a fresh selective quality rejection."""

import weakref

import warp as wp

wp.set_module_options({"enable_backward": False, "fast_math": False})


@wp.func
def proposal_at(
    i: int,
    original: wp.array[wp.vec3],
    points: wp.array[wp.vec3],
    delta: wp.array[wp.vec3d],
    edge: wp.array[wp.float64],
    alpha: wp.array[wp.float64],
    maximum: wp.array[wp.float64],
    blocked: wp.array[int],
):
    if blocked[i] != 0:
        points[i] = original[i]
    else:
        p = wp.vec3(wp.vec3d(original[i]) + alpha[0] * delta[i])
        points[i] = p
        movement = wp.length(wp.vec3d(p) - wp.vec3d(original[i])) / edge[i]
        if not wp.isfinite(movement):
            movement = wp.float64(1.0e300)
        wp.atomic_max(maximum, 0, movement)


@wp.kernel
def propose(
    original: wp.array[wp.vec3],
    points: wp.array[wp.vec3],
    delta: wp.array[wp.vec3d],
    edge: wp.array[wp.float64],
    alpha: wp.array[wp.float64],
    maximum: wp.array[wp.float64],
    blocked: wp.array[int],
):
    proposal_at(wp.tid(), original, points, delta, edge, alpha, maximum, blocked)


@wp.kernel
def propose_active(
    original: wp.array[wp.vec3],
    points: wp.array[wp.vec3],
    delta: wp.array[wp.vec3d],
    edge: wp.array[wp.float64],
    alpha: wp.array[wp.float64],
    maximum: wp.array[wp.float64],
    active: wp.array[int],
    blocked: wp.array[int],
):
    i = wp.tid()
    if i < active[0]:
        proposal_at(i, original, points, delta, edge, alpha, maximum, blocked)


@wp.kernel
def eligible(
    accepted: wp.array[int],
    rejections: wp.array[int],
    before: wp.array[int],
    reason: wp.array[int],
    used: wp.array[int],
    budget: int,
    ready: wp.array[int],
):
    # last_error is sticky. Require a new quality-only rejection of this trial.
    ready[0] = int(
        accepted[0] == 0 and rejections[0] > before[0] and reason[0] == 4 and used[0] < budget
    )


@wp.kernel
def block_bad_vertices(
    tets: wp.array2d[int],
    count: wp.array[int],
    keep: wp.array[int],
    quality: wp.array[wp.float64],
    previous: wp.array[wp.float64],
    original: wp.array[wp.vec3],
    points: wp.array[wp.vec3],
    mask: wp.array[int],
    blocked: wp.array[int],
    any_blocked: wp.array[int],
    statistics: wp.array[int],
):
    i = wp.tid()
    if (
        i < count[0]
        and keep[i] != 0
        and quality[i] < previous[0] * (wp.float64(1.0) - wp.float64(1.0e-10))
    ):
        for k in range(4):
            v = tets[i, k]
            p = wp.vec3(points[v])
            old = wp.vec3(original[v])
            moved = p[0] != old[0] or p[1] != old[1] or p[2] != old[2]
            if mask[v] != 0 and moved:
                if wp.atomic_or(blocked, v, 1) == 0:
                    wp.atomic_or(any_blocked, 0, 1)
                    wp.atomic_add(statistics, 1, 1)


@wp.kernel
def finish_local(used: wp.array[int], rejected: wp.array[int], statistics: wp.array[int]):
    used[0] += 1
    rejected[0] += 1
    statistics[0] += 1


class LocalQualityBacktracking:
    def __init__(self, core, budget):
        self.core, self.device = weakref.proxy(core), core.device
        self.budget = budget
        self.blocked = wp.zeros(core.n, dtype=int, device=self.device)
        self.used = wp.zeros(1, dtype=int, device=self.device)
        self.before = wp.zeros(1, dtype=int, device=self.device)
        self.ready = wp.zeros(1, dtype=int, device=self.device)
        self.any_blocked = wp.zeros(1, dtype=int, device=self.device)
        # Total local retries and blocked vertices in this core invocation.
        self.statistics = wp.zeros(2, dtype=int, device=self.device)
        core._propose = propose if core.active_count is None else propose_active
        core._propose_args.append(self.blocked)
        wp.load_module(module=__name__, device=self.device)

    def _launch(self, kernel, dim, args):
        wp.launch(kernel, dim, inputs=args, device=self.device)

    def reset(self):
        self.statistics.zero_()
        self.start_step()

    def start_step(self):
        self.blocked.zero_()
        self.used.zero_()

    def begin_attempt(self):
        wp.copy(self.before, self.core.selection.rejections)

    def _mark(self):
        c, s = self.core, self.core.selection
        self.any_blocked.zero_()
        self._launch(
            block_bad_vertices,
            len(c.full_tetrahedra),
            [
                c.full_tetrahedra,
                c.full_tetrahedra_count,
                s.keep,
                s.quality,
                s.previous_minimum,
                c._original,
                c.points,
                s.mask,
                self.blocked,
                self.any_blocked,
                self.statistics,
            ],
        )
        wp.copy(self.ready, self.any_blocked)

    def _retry(self):
        self._launch(finish_local, 1, [self.used, self.core.rejected_steps, self.statistics])

    def finish(self, ordinary_finish):
        c, s = self.core, self.core.selection
        self._launch(
            eligible,
            1,
            [
                c._accepted,
                s.rejections,
                self.before,
                s.last_error,
                self.used,
                self.budget,
                self.ready,
            ],
        )
        wp.capture_if(self.ready, self._mark)

        def ordinary():
            # New cells may appear because of sites outside their own corners.
            # If this heuristic cannot advance, restore the full direction and
            # retain the original global backtracking and eventual rollback.
            self.blocked.zero_()
            ordinary_finish()

        wp.capture_if(self.ready, self._retry, ordinary)
