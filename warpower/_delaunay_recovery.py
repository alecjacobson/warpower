"""Optional certified Delaunay construction with device-only finite retries."""

import operator

import warp as wp

from . import _canonical, _validation
from ._delaunay_certificate import DelaunayCertificate, direction
from ._graph import DiagramWorkspace, input_ready
from ._site_bvh import SiteHierarchy

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.kernel
def prepare(
    base_error: wp.array[int],
    certificate_error: wp.array[int],
    ready: wp.array[int],
):
    certificate_error[0] = 0
    ready[0] = int(base_error[0] == 0)


@wp.kernel
def begin_retry(
    base_error: wp.array[int],
    certificate_error: wp.array[int],
    error: wp.array[int],
    running: wp.array[int],
):
    error[0] = 0
    if base_error[0] == 1:
        error[0] = 1
    elif base_error[0] != 0 or certificate_error[0] != 0:
        error[0] = 2
    running[0] = int(error[0] == 2)


@wp.struct
class ReplayState:
    count: wp.array[int]
    error: wp.array[int]
    attempts: wp.array[int]
    recovered: wp.array[int]
    base_attempted: wp.array[int]
    symbolic_attempted: wp.array[int]
    symbolic_recovered: wp.array[int]
    symbolic_error: wp.array[int]
    symbolic_expansions: wp.array[int]
    used: wp.array[int]
    base_error: wp.array[int]
    candidate_error: wp.array[int]
    certificate_error: wp.array[int]
    running: wp.array[int]
    preferred: wp.array[int]
    recovery_first: wp.array[int]
    auxiliary_level: wp.array[int]


@wp.kernel
def reset(state: ReplayState, cache: bool):
    state.count[0] = 0
    state.error[0] = 2
    state.attempts[0] = 0
    state.recovered[0] = 0
    state.base_attempted[0] = 0
    state.symbolic_attempted[0] = 0
    state.symbolic_recovered[0] = 0
    state.symbolic_error[0] = 0
    state.symbolic_expansions[0] = 0
    state.used[0] = 0
    state.base_error[0] = 0
    state.candidate_error[0] = 0
    state.certificate_error[0] = 0
    state.running[0] = 0
    state.recovery_first[0] = int(cache and state.preferred[0] != 0)
    state.auxiliary_level[0] = 0


@wp.kernel
def fallback_ready(error: wp.array[int], attempted: wp.array[int], ready: wp.array[int]):
    ready[0] = int(error[0] == 2 and attempted[0] == 0)


@wp.kernel
def preferred_ready(
    valid: wp.array[int], preferred: wp.array[int], strategy: int, ready: wp.array[int]
):
    ready[0] = int(valid[0] != 0 and preferred[0] == strategy)


@wp.kernel
def finish_symbolic(
    symbolic_error: wp.array[int],
    recovered: wp.array[int],
    symbolic_recovered: wp.array[int],
    used: wp.array[int],
    error: wp.array[int],
):
    if symbolic_error[0] == 0:
        error[0] = 0
        recovered[0] = 1
        symbolic_recovered[0] = 1
        used[0] = 2
    else:
        error[0] = 2


@wp.kernel
def normalize(
    points: wp.array[wp.vec3],
    bounds: wp.array[wp.vec3],
    owners: wp.array[int],
    attempts: wp.array[int],
    normalized: wp.array[wp.vec3],
    weights: wp.array[float],
):
    i = wp.tid()
    lo, hi = wp.vec3d(bounds[0]), wp.vec3d(bounds[1])
    extent = wp.max(hi[0] - lo[0], wp.max(hi[1] - lo[1], hi[2] - lo[2]))
    scale = wp.max(extent, wp.float64(1.0e-300))
    normalized[i] = wp.vec3((wp.vec3d(points[i]) - (lo + hi) * wp.float64(0.5)) / scale)
    amplitude = float(1.0e-5)
    for _ in range(attempts[0]):
        amplitude *= 0.1
    weights[i] = amplitude * direction(owners[i])


@wp.func
def amplitude_level(attempt: int, preferred: int, maximum: int, cache: bool):
    first = int(0)
    if cache and preferred >= 0 and preferred < maximum:
        first = preferred
    level = attempt
    if attempt == 0:
        level = first
    elif attempt <= first:
        level = attempt - 1
    return level


@wp.kernel
def normalize_ordered(
    points: wp.array[wp.vec3],
    bounds: wp.array[wp.vec3],
    owners: wp.array[int],
    attempts: wp.array[int],
    preferred_level: wp.array[int],
    maximum: int,
    cache: bool,
    normalized: wp.array[wp.vec3],
    weights: wp.array[float],
    auxiliary_level: wp.array[int],
):
    i = wp.tid()
    level = amplitude_level(attempts[0], preferred_level[0], maximum, cache)
    if i == 0:
        auxiliary_level[0] = level
    lo, hi = wp.vec3d(bounds[0]), wp.vec3d(bounds[1])
    extent = wp.max(hi[0] - lo[0], wp.max(hi[1] - lo[1], hi[2] - lo[2]))
    scale = wp.max(extent, wp.float64(1.0e-300))
    normalized[i] = wp.vec3((wp.vec3d(points[i]) - (lo + hi) * wp.float64(0.5)) / scale)
    amplitude = float(1.0e-5)
    for _ in range(level):
        amplitude *= 0.1
    weights[i] = amplitude * direction(owners[i])


@wp.kernel
def candidate_ready(
    candidate_error: wp.array[int], certificate_error: wp.array[int], ready: wp.array[int]
):
    certificate_error[0] = 0
    ready[0] = int(candidate_error[0] == 0)


@wp.kernel
def finish_retry(
    candidate_error: wp.array[int],
    certificate_error: wp.array[int],
    attempts: wp.array[int],
    recovered: wp.array[int],
    used: wp.array[int],
    running: wp.array[int],
    error: wp.array[int],
    maximum: int,
):
    attempts[0] += 1
    if candidate_error[0] == 0 and certificate_error[0] == 0:
        recovered[0] = 1
        used[0] = 1
        running[0] = 0
        error[0] = 0
    elif attempts[0] >= maximum:
        error[0] = 2
        running[0] = 0


@wp.kernel
def finalize(
    error: wp.array[int],
    count: wp.array[int],
    used: wp.array[int],
    preferred: wp.array[int],
    auxiliary_level: wp.array[int],
    preferred_level: wp.array[int],
    cache: bool,
):
    preferred[0] = 0
    if cache and error[0] == 0:
        preferred[0] = used[0]
    preferred_level[0] = 0
    if cache and error[0] == 0 and used[0] == 1:
        preferred_level[0] = auxiliary_level[0]
    if error[0] != 0:
        count[0] = 0


@wp.kernel
def fast_preference(available: wp.array[int], cache: bool):
    if not cache:
        available[0] = 1


@wp.kernel
def finish_fast(
    attempted: wp.array[int],
    repairs: wp.array[int],
    base_error: wp.array[int],
    certificate_error: wp.array[int],
    error: wp.array[int],
    available: wp.array[int],
    reported_attempt: wp.array[int],
    accepted: wp.array[int],
    fast_error: wp.array[int],
    repaired_cells: wp.array[int],
    fallback: wp.array[int],
    cache: bool,
):
    reported_attempt[0] = attempted[0]
    repaired_cells[0] = repairs[0]
    accepted[0] = int(attempted[0] != 0 and error[0] == 0)
    fallback[0] = int(attempted[0] != 0 and error[0] == 2)
    if attempted[0] != 0:
        fast_error[0] = int(base_error[0] != 0) | (int(certificate_error[0] != 0) << 1)
        if cache and error[0] != 0:
            available[0] = 0


@wp.kernel
def fast_invalid_reset(error: wp.array[int], available: wp.array[int]):
    if error[0] == 1:
        available[0] = 1


class DelaunayWorkspace:
    """Exact-certified ordinary Delaunay output, with bounded recovery attempts.

    Setup allocates two construction workspaces and all certificate storage.
    Replay leaves input positions untouched and performs no host decisions.
    Exhaustion is explicit: error != 0 and tetrahedra_count == 0. This is not
    a universal exact constructor; finite candidate generation can still fail.
    """

    def __init__(
        self,
        points,
        *,
        tetrahedra_capacity=None,
        tetrahedra_cell_capacity=1024,
        chunk_size=None,
        max_attempts=6,
        cache_strategy=True,
        fast_candidates=True,
        symbolic_recovery=True,
        symbolic_max_expansions=16,
    ):
        self.max_attempts = operator.index(max_attempts)
        self.cache_strategy = bool(cache_strategy)
        self.symbolic_recovery = bool(symbolic_recovery)
        self.symbolic_max_expansions = operator.index(symbolic_max_expansions)
        if not 1 <= self.max_attempts <= 12:
            raise ValueError("max_attempts must be in [1, 12]")
        if not 1 <= self.symbolic_max_expansions <= 32:
            raise ValueError("symbolic_max_expansions must be in [1, 32]")
        settings = dict(
            adjacency=False,
            return_tetrahedra=True,
            tetrahedra_capacity=tetrahedra_capacity,
            tetrahedra_cell_capacity=tetrahedra_cell_capacity,
            chunk_size=chunk_size,
        )
        self.fast_candidates = bool(fast_candidates) and len(points) >= 32768
        builder = DiagramWorkspace
        if self.fast_candidates:
            from ._fast_candidates import CandidateWorkspace

            builder = CandidateWorkspace
        self._base = builder(points, **settings)
        self.points, self.device, self.n = self._base.points, self._base.device, self._base.n
        if self.n < 4:
            raise ValueError("DelaunayWorkspace requires at least four sites")
        with wp.ScopedDevice(self.device):
            self._normalized = wp.zeros_like(self.points)
            self._weights = wp.zeros(self.n, dtype=float)
            self._auxiliary = DiagramWorkspace(self._normalized, self._weights, **settings)
            self._original = SiteHierarchy(self.points)
            self._original.tree.owners = self._base._hierarchy.owners
            self._candidate = wp.empty_like(self._base.tetrahedra)
            self._candidate_count = wp.empty(1, dtype=int)
            self._certificate = DelaunayCertificate(
                self.points, self._candidate, self._candidate_count, self._base._hierarchy
            )
            self.tetrahedra = self._certificate.tetrahedra
            self.tetrahedra_count = self._certificate.tetrahedra_count
            self.certificate_error = self._certificate.error
            self.base_error = self._base.error
            self.candidate_error = self._auxiliary.error
            self.error = wp.zeros(1, dtype=int)
            self.attempts = wp.zeros(1, dtype=int)
            self.recovered = wp.zeros(1, dtype=int)
            self.base_attempted = wp.zeros(1, dtype=int)
            self.recovery_first = wp.zeros(1, dtype=int)
            self.symbolic_attempted = wp.zeros(1, dtype=int)
            self.symbolic_recovered = wp.zeros(1, dtype=int)
            self._symbolic = None
            if self.symbolic_recovery:
                from ._symbolic_delaunay import SymbolicConstructor

                self._symbolic = SymbolicConstructor(
                    self._base,
                    self._certificate,
                    self._candidate,
                    self._candidate_count,
                    max_expansions=self.symbolic_max_expansions,
                )
                self.symbolic_error = self._symbolic.error
                self.symbolic_expansions = self._symbolic.expansions
            else:
                self.symbolic_error = wp.zeros(1, dtype=int)
                self.symbolic_expansions = wp.zeros(1, dtype=int)
            self.fast_attempted = wp.zeros(1, dtype=int)
            self.fast_accepted = wp.zeros(1, dtype=int)
            self.fast_error = wp.zeros(1, dtype=int)
            self.fast_repaired_cells = wp.zeros(1, dtype=int)
            self._fast_available = wp.ones(1, dtype=int) if self.fast_candidates else None
            self._fast_fallback = wp.empty(1, dtype=int) if self.fast_candidates else None
            self._preferred = wp.zeros(1, dtype=int)
            self._preferred_level = wp.zeros(1, dtype=int)
            self.auxiliary_level = wp.zeros(1, dtype=int)
            self._used = wp.zeros(1, dtype=int)
            self._valid = wp.empty(1, dtype=int)
            self._ready = wp.empty(1, dtype=int)
            self._running = wp.empty(1, dtype=int)
            self._state = ReplayState()
            for name in (
                "error",
                "attempts",
                "recovered",
                "base_attempted",
                "symbolic_attempted",
                "symbolic_recovered",
                "symbolic_error",
                "symbolic_expansions",
                "base_error",
                "candidate_error",
                "certificate_error",
                "recovery_first",
                "auxiliary_level",
            ):
                setattr(self._state, name, getattr(self, name))
            self._state.count = self.tetrahedra_count
            self._state.used = self._used
            self._state.running = self._running
            self._state.preferred = self._preferred
            wp.load_module(module=__name__, device=self.device)
            with wp.ScopedCapture() as capture:
                self._record()
            self.graph = capture.graph
        wp.synchronize_device(self.device)

    def _launch(self, kernel, dim, args):
        wp.launch(kernel, dim, inputs=args, device=self.device)

    def _record(self):
        if self.fast_candidates:
            self.fast_attempted.zero_()
            self.fast_accepted.zero_()
            self.fast_error.zero_()
            self.fast_repaired_cells.zero_()
            self._launch(fast_preference, 1, [self._fast_available, self.cache_strategy])
        self._launch(reset, 1, [self._state, self.cache_strategy])
        wp.capture_if(self.recovery_first, self._preferred_path)
        self._launch(fallback_ready, 1, [self.error, self.base_attempted, self._ready])
        wp.capture_if(self._ready, self._fallback_path)
        self._launch(
            finalize,
            1,
            [
                self.error,
                self.tetrahedra_count,
                self._used,
                self._preferred,
                self.auxiliary_level,
                self._preferred_level,
                self.cache_strategy,
            ],
        )

        if self.fast_candidates:
            self._launch(fast_invalid_reset, 1, [self.error, self._fast_available])

    def _fallback_path(self):
        self._base_path()
        self._launch(fallback_ready, 1, [self.error, self.attempts, self._running])
        wp.capture_while(
            self._running, lambda: self._retry(self._base._hierarchy, self._base._bounds)
        )
        if self._symbolic is not None:
            self._launch(fallback_ready, 1, [self.error, self.symbolic_attempted, self._ready])
            wp.capture_if(
                self._ready,
                lambda: self._symbolic_path(self._base._hierarchy, self._base._bounds),
            )

    def _base_path(self):
        self.base_attempted.fill_(1)
        if self.fast_candidates:
            wp.copy(self._base.enabled, self._fast_available)
        self._base_attempt()
        if self.fast_candidates:
            self._launch(
                finish_fast,
                1,
                [
                    self._base.attempted,
                    self._base.repaired_cells,
                    self.base_error,
                    self.certificate_error,
                    self.error,
                    self._fast_available,
                    self.fast_attempted,
                    self.fast_accepted,
                    self.fast_error,
                    self.fast_repaired_cells,
                    self._fast_fallback,
                    self.cache_strategy,
                ],
            )
            wp.capture_if(self._fast_fallback, self._original_after_fast)

    def _original_after_fast(self):
        self._base.enabled.zero_()
        self._base_attempt()

    def _base_attempt(self):
        self._base.run()
        self._launch(
            prepare,
            1,
            [
                self.base_error,
                self.certificate_error,
                self._ready,
            ],
        )
        wp.copy(self._candidate, self._base.tetrahedra)
        wp.copy(self._candidate_count, self._base.tetrahedra_count)
        wp.capture_if(self._ready, self._certificate.run)
        self._launch(
            begin_retry, 1, [self.base_error, self.certificate_error, self.error, self._running]
        )

    def _preferred_path(self):
        self._base._validation.zero_()
        self._launch(
            _validation.validate,
            self.n + 1,
            [
                self.points,
                self._base.weights,
                self._base._empty,
                self._base._empty,
                False,
                self._base._validation,
            ],
        )
        self._launch(input_ready, 1, [self._base._validation, self.error, self._valid])
        wp.capture_if(self._valid, self._refresh_original)
        self._launch(preferred_ready, 1, [self._valid, self._preferred, 1, self._running])
        wp.capture_while(
            self._running, lambda: self._retry(self._original.tree, self._original._bounds)
        )
        if self._symbolic is not None:
            self._launch(preferred_ready, 1, [self._valid, self._preferred, 2, self._ready])
            wp.capture_if(
                self._ready,
                lambda: self._symbolic_path(self._original.tree, self._original._bounds),
            )

    def _symbolic_path(self, tree, bounds):
        self.symbolic_attempted.fill_(1)
        self._symbolic.run(tree, bounds)
        self._launch(
            finish_symbolic,
            1,
            [self.symbolic_error, self.recovered, self.symbolic_recovered, self._used, self.error],
        )

    def _refresh_original(self):
        self._original.rebuild()
        self._base._owner_table.fill_(-1)
        self._launch(
            _canonical.insert,
            self.n,
            [
                self.points,
                self._base.weights,
                self._base._bits,
                self._base._owner_table,
                self._base._buckets,
            ],
        )
        self._launch(
            _canonical.resolve,
            self.n,
            [self._base._owner_table, self._base._buckets, self._original.tree.owners],
        )

    def _retry(self, tree, bounds):
        self._launch(
            normalize_ordered,
            self.n,
            [
                self.points,
                bounds,
                tree.owners,
                self.attempts,
                self._preferred_level,
                self.max_attempts,
                self.cache_strategy,
                self._normalized,
                self._weights,
                self.auxiliary_level,
            ],
        )
        self._auxiliary.run()
        wp.copy(self._candidate, self._auxiliary.tetrahedra)
        wp.copy(self._candidate_count, self._auxiliary.tetrahedra_count)
        self._launch(
            candidate_ready, 1, [self.candidate_error, self.certificate_error, self._ready]
        )
        wp.capture_if(self._ready, lambda: self._certificate.run(tree))
        self._launch(
            finish_retry,
            1,
            [
                self.candidate_error,
                self.certificate_error,
                self.attempts,
                self.recovered,
                self._used,
                self._running,
                self.error,
                self.max_attempts,
            ],
        )

    def run(self, stream=None):
        stream = wp.get_stream(self.device) if stream is None else stream
        if stream.device != self.device:
            raise ValueError("Workspace stream must use the input device")
        with wp.ScopedStream(stream):
            if stream.is_capturing:
                if stream not in self.device.captures:
                    raise RuntimeError(
                        "Use Warp ScopedCapture on the shared stream for conditional graphs"
                    )
                self._record()
            else:
                wp.capture_launch(self.graph, stream=stream)
        return self
