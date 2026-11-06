"""Shared device transaction for optional VTM quality-improvement proposals."""

import math
import operator

import warp as wp

from . import _extraction, _selective
from ._extraction import ExtractionWorkspace
from ._surface_bvh import SurfaceTree

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.struct
class MoveState:
    active: wp.array[int]
    source_count: wp.array[int]
    error: wp.array[int]
    status: wp.array[int]
    ready: wp.array[int]
    running: wp.array[int]
    accepted: wp.array[int]
    attempts: wp.array[int]
    invalid_proposals: wp.array[int]
    proposal_valid: wp.array[int]
    rejection_reason: wp.array[int]
    site: wp.array[int]
    position: wp.array[wp.vec3]
    point_count: wp.array[int]
    full_count: wp.array[int]
    retained_count: wp.array[int]
    last_errors: wp.array[int]
    current: wp.array[wp.float64]
    previous: wp.array[wp.float64]
    topology: wp.array[int]
    previous_topology: wp.array[int]
    capacity: int
    tet_capacity: int
    max_attempts: int
    threshold: wp.float64
    boundary_tolerance: wp.float64
    volume_tolerance: wp.float64


@wp.kernel
def begin(
    state: MoveState,
    core_error: wp.array[int],
    mesh_error: wp.array[int],
    boundary_error: wp.array[int],
    domain_error: wp.array[int],
    sizing_error: wp.array[int],
):
    state.error[0] = 0
    state.status[0] = 1  # No eligible proposal unless preparation finds one.
    state.accepted[0] = 0
    state.attempts[0] = 0
    state.invalid_proposals[0] = 0
    state.running[0] = 0
    state.proposal_valid[0] = 0
    state.rejection_reason[0] = 0
    state.site[0] = -1
    state.point_count[0] = 0
    state.full_count[0] = 0
    state.retained_count[0] = 0
    for k in range(5):
        state.last_errors[k] = 0
    if (
        core_error[0] != 0
        or mesh_error[0] != 0
        or boundary_error[0] != 0
        or domain_error[0] != 0
        or sizing_error[0] != 0
        or state.active[0] < 4
        or state.active[0] > state.capacity
        or state.source_count[0] <= 0
        or state.source_count[0] > state.tet_capacity
    ):
        state.error[0] = 1
    state.ready[0] = int(state.error[0] == 0)


@wp.kernel
def initial_ready(
    state: MoveState, error: wp.array[int], valid: wp.array[int], count: wp.array[int]
):
    if error[0] != 0 or valid[0] == 0 or count[0] == 0:
        state.error[0] |= 2
    state.ready[0] = int(state.error[0] == 0)


@wp.kernel
def deficit_compact(
    quality: wp.array[wp.float64],
    count: wp.array[int],
    threshold: wp.float64,
    metrics: wp.array[wp.float64],
):
    i = wp.tid()
    if i < count[0]:
        deficit = wp.max(wp.float64(0.0), threshold - quality[i])
        wp.atomic_add(metrics, 1, deficit * deficit)


@wp.kernel
def deficit_full(
    quality: wp.array[wp.float64],
    keep: wp.array[int],
    error: wp.array[int],
    threshold: wp.float64,
    metrics: wp.array[wp.float64],
):
    i = wp.tid()
    if error[0] == 0 and keep[i] != 0:
        deficit = wp.max(wp.float64(0.0), threshold - quality[i])
        wp.atomic_add(metrics, 1, deficit * deficit)


@wp.kernel
def sum_euler(euler: wp.array[int], topology: wp.array[int]):
    wp.atomic_add(topology, 3, euler[wp.tid()])


@wp.kernel
def store_metrics(
    state: MoveState,
    minimum: wp.array[wp.float64],
    volume: wp.array[wp.float64],
    rms: wp.array[wp.float64],
    boundary_energy: wp.array[wp.float64],
    components: wp.array[int],
    shells: wp.array[int],
    cavities: wp.array[int],
):
    state.current[0] = minimum[0]
    state.current[2] = volume[0]
    state.current[3] = rms[0]
    state.current[4] = boundary_energy[0]
    state.current[5] = boundary_energy[1]
    state.topology[0] = components[0]
    state.topology[1] = shells[0]
    state.topology[2] = cavities[0]


@wp.kernel
def start_trial(state: MoveState):
    state.attempts[0] += 1
    state.proposal_valid[0] = 0
    state.accepted[0] = 0
    state.rejection_reason[0] = 0


@wp.kernel
def reject_coincident(state: MoveState, original: wp.array[wp.vec3]):
    i = wp.tid()
    if i < state.active[0]:
        a, b = original[i], state.position[0]
        if a[0] == b[0] and a[1] == b[1] and a[2] == b[2]:
            wp.atomic_and(state.proposal_valid, 0, 0)


@wp.kernel
def apply_proposal(state: MoveState, points: wp.array[wp.vec3]):
    points[state.site[0]] = state.position[0]


@wp.kernel
def geometry_ready(
    state: MoveState,
    mesh_error: wp.array[int],
    invalid: wp.array[int],
    boundary_error: wp.array[int],
):
    state.last_errors[0] = mesh_error[0]
    state.last_errors[1] = invalid[0]
    state.last_errors[2] = boundary_error[0]
    state.last_errors[3] = 0
    state.last_errors[4] = 0
    state.ready[0] = int(mesh_error[0] == 0 and invalid[0] == 0 and boundary_error[0] == 0)
    if state.ready[0] == 0:
        state.rejection_reason[0] = 1


@wp.func
def quality_reason(state: MoveState):
    for k in range(6):
        if k != 2 and k != 3 and not wp.isfinite(state.current[k]):
            return 1
    numerical = wp.float64(1.0e-10)
    fit_limit = wp.float64(1.0) + state.boundary_tolerance + numerical
    reason = int(0)
    if state.current[0] < state.previous[0] * (wp.float64(1.0) - numerical):
        reason |= 2
    if state.current[1] >= state.previous[1] * (wp.float64(1.0) - numerical):
        reason |= 4
    if (
        state.current[4] > state.previous[4] * fit_limit
        or state.current[5] > state.previous[5] * fit_limit
    ):
        reason |= 8
    return reason


@wp.kernel
def quality_ready(
    state: MoveState,
    minimum: wp.array[wp.float64],
    count: wp.array[int],
    error: wp.array[int],
    boundary_energy: wp.array[wp.float64],
):
    state.current[0] = minimum[0]
    state.current[4] = boundary_energy[0]
    state.current[5] = boundary_energy[1]
    state.last_errors[3] = error[0]
    reason = quality_reason(state)
    if error[0] != 0 or count[0] == 0:
        reason |= 1
    state.rejection_reason[0] = reason
    state.ready[0] = int(reason == 0)


@wp.kernel
def accept_trial(
    state: MoveState,
    valid: wp.array[int],
    error: wp.array[int],
    diagnostics: wp.array[int],
    tree: SurfaceTree,
):
    state.last_errors[3] = error[0]
    state.last_errors[4] = diagnostics[0]
    reason = quality_reason(state)
    if valid[0] == 0 or error[0] != 0 or diagnostics[0] != 0:
        reason |= 1
    for k in range(6):
        if not wp.isfinite(state.current[k]):
            reason |= 1
    for k in range(4):
        if state.topology[k] != state.previous_topology[k]:
            reason |= 16
    if wp.abs(state.current[2] - state.previous[2]) > state.previous[2] * (
        state.volume_tolerance + wp.float64(1.0e-10)
    ):
        reason |= 32
    diagonal = wp.length(wp.vec3d(tree.upper[1]) - wp.vec3d(tree.lower[1]))
    if state.current[3] > state.previous[3] * (
        wp.float64(1.0) + state.boundary_tolerance
    ) + diagonal * wp.float64(1.0e-7):
        reason |= 64
    state.rejection_reason[0] = reason
    state.accepted[0] = int(reason == 0)


@wp.kernel
def finish_trial(state: MoveState):
    if state.proposal_valid[0] == 0:
        state.invalid_proposals[0] += 1
        state.rejection_reason[0] |= 128
    state.running[0] = int(state.accepted[0] == 0 and state.attempts[0] < state.max_attempts)
    state.status[0] = wp.where(state.accepted[0] != 0, 0, 2)


@wp.kernel
def need_restore(state: MoveState):
    state.ready[0] = int(state.accepted[0] == 0 and state.attempts[0] > 0)


@wp.kernel
def history_start(
    energy: wp.array[wp.float64], boundary: wp.array[wp.float64], history: wp.array2d[wp.float64]
):
    history[0, 0] = energy[0]
    history[0, 1] = boundary[0]
    history[0, 2] = boundary[1]


@wp.kernel
def restored_core(
    state: MoveState, invalid: wp.array[int], boundary: wp.array[int], core_error: wp.array[int]
):
    core_error[0] = 0
    if invalid[0] != 0:
        core_error[0] |= 4
    if boundary[0] != 0:
        core_error[0] |= 2
    if core_error[0] != 0:
        state.error[0] |= 8
        state.source_count[0] = 0


@wp.kernel
def restored_extraction(state: MoveState, valid: wp.array[int], error: wp.array[int]):
    if valid[0] == 0 or error[0] != 0:
        state.error[0] |= 8


@wp.kernel
def publish(state: MoveState, retained: wp.array[int]):
    if state.error[0] == 0:
        state.point_count[0] = state.active[0]
        state.full_count[0] = state.source_count[0]
        state.retained_count[0] = retained[0]
    else:
        state.point_count[0] = 0
        state.full_count[0] = 0
        state.retained_count[0] = 0
        state.accepted[0] = 0
        state.status[0] = 3


class QualityMove:
    """Private base; subclasses allocate proposal state, then call _finish_setup."""

    def __init__(
        self,
        core,
        *,
        threshold=0.3,
        boundary_tolerance=0.01,
        volume_tolerance=0.01,
        max_attempts=8,
        extraction_tolerance=0.4,
    ):
        if not math.isfinite(threshold) or not 0 < threshold <= 1:
            raise ValueError("threshold must be in (0,1]")
        for name, value in [
            ("boundary_tolerance", boundary_tolerance),
            ("volume_tolerance", volume_tolerance),
        ]:
            if not math.isfinite(value) or value < 0:
                raise ValueError(name + " must be finite and nonnegative")
        attempts = operator.index(max_attempts)
        if not 1 <= attempts <= 1024:
            raise ValueError("max_attempts must be in [1,1024]")
        self.core, self.device = core, core.device
        self.points, self.full_tetrahedra = core.points, core.full_tetrahedra
        self._m = len(self.full_tetrahedra)
        self.extraction = ExtractionWorkspace(
            core.domain,
            self.points,
            self.full_tetrahedra,
            core.full_tetrahedra_count,
            source_error=core.error,
            tolerance=extraction_tolerance,
        )
        self.tetrahedra = self.extraction.tetrahedra
        self._tolerance = extraction_tolerance
        with wp.ScopedDevice(self.device):
            state = self._state = MoveState()
            state.capacity, state.tet_capacity = core.n, self._m
            state.max_attempts, state.threshold = attempts, wp.float64(threshold)
            state.boundary_tolerance, state.volume_tolerance = (
                wp.float64(boundary_tolerance),
                wp.float64(volume_tolerance),
            )
            state.active = (
                core.active_count
                if core.active_count is not None
                else wp.full(1, core.n, dtype=int)
            )
            state.source_count = core.full_tetrahedra_count
            for name in (
                "error",
                "status",
                "ready",
                "running",
                "accepted",
                "attempts",
                "invalid_proposals",
                "proposal_valid",
                "rejection_reason",
                "site",
                "point_count",
                "full_count",
                "retained_count",
            ):
                value = wp.zeros(1, dtype=int)
                setattr(state, name, value)
                setattr(self, name, value)
            state.position = self.proposed_position = wp.zeros(1, dtype=wp.vec3)
            state.last_errors = self.last_errors = wp.zeros(5, dtype=int)
            state.current = self.metrics = wp.zeros(6, dtype=wp.float64)
            state.previous = self.previous_metrics = wp.zeros(6, dtype=wp.float64)
            state.topology = self.topology = wp.zeros(4, dtype=int)
            state.previous_topology = self.previous_topology = wp.zeros(4, dtype=int)
            self.full_tetrahedra_count = self.full_count
            self.tetrahedra_count = self.retained_count
            self._sizing_error = core.error if core.sizing is None else core.sizing.error
            self._old_points = wp.empty_like(core.points)
            self._old_tets = wp.empty_like(core.full_tetrahedra)
            self._old_count = wp.empty(1, dtype=int)
            self._old_active = wp.empty(1, dtype=int)
            self._query_error = wp.zeros(1, dtype=int)
            self._query_count = wp.zeros(1, dtype=int)
            self._query_minimum = wp.ones(1, dtype=wp.float64)
            self._query_keep = wp.zeros(self._m, dtype=int)
            self._query_quality = wp.zeros(self._m, dtype=wp.float64)

    def _finish_setup(self):
        with wp.ScopedDevice(self.device):
            wp.load_module(module=__name__, device=self.device)
            wp.load_module(module="warpower._selective", device=self.device)
            with wp.ScopedCapture() as capture:
                self._record()
            self.graph = capture.graph
        wp.synchronize_device(self.device)

    def _launch(self, kernel, dim, args):
        wp.launch(kernel, dim, inputs=args, device=self.device)

    def _record(self):
        c = self.core
        self._launch(
            begin,
            1,
            [
                self._state,
                c.error,
                c.mesh.error,
                c.boundary.error,
                c.domain.error,
                self._sizing_error,
            ],
        )
        wp.capture_if(self.ready, self._initial)
        self._launch(publish, 1, [self._state, self.extraction.tetrahedra_count])

    def _initial(self):
        e = self.extraction
        e.run()
        self._launch(initial_ready, 1, [self._state, e.error, e.valid, e.tetrahedra_count])
        wp.capture_if(self.ready, self._operate)

    def _metrics(self):
        e = self.extraction
        self.metrics.zero_()
        self.topology.zero_()
        self._launch(
            deficit_compact,
            self._m,
            [e.radius_ratio, e.tetrahedra_count, self._state.threshold, self.metrics],
        )
        self._launch(sum_euler, len(e.shell_euler), [e.shell_euler, self.topology])
        self._launch(
            store_metrics,
            1,
            [
                self._state,
                e.minimum_radius_ratio,
                e.volume,
                e.surface_rms_distance,
                self.core.boundary.energy,
                e.component_count,
                e.shell_count,
                e.cavity_count,
            ],
        )

    def _operate(self):
        self._metrics()
        wp.copy(self.previous_metrics, self.metrics)
        wp.copy(self.previous_topology, self.topology)
        wp.copy(self._old_points, self.points)
        wp.copy(self._old_tets, self.full_tetrahedra)
        wp.copy(self._old_count, self.core.full_tetrahedra_count)
        wp.copy(self._old_active, self._state.active)
        self._prepare_proposal()
        wp.capture_while(self.running, self._try)
        wp.capture_if(self.accepted, self._commit)
        self._launch(need_restore, 1, [self._state])
        wp.capture_if(self.ready, self._restore)

    def _try(self):
        wp.copy(self.points, self._old_points)
        self._launch(start_trial, 1, [self._state])
        self._propose()
        self._launch(reject_coincident, self.core.n, [self._state, self._old_points])
        wp.capture_if(self.proposal_valid, self._build)
        self._launch(finish_trial, 1, [self._state])

    def _build(self):
        c = self.core
        self._launch(apply_proposal, 1, [self._state, self.points])
        c._reconstruct()
        c._evaluate(True)
        c.boundary.run()
        self._launch(geometry_ready, 1, [self._state, c.mesh.error, c._invalid, c.boundary.error])
        wp.capture_if(self.ready, self._quality)

    def _quality(self):
        c = self.core
        self._query_count.zero_()
        self._query_minimum.fill_(1)
        # Clear just the deficit; retained-volume and RMS entries remain finite
        # previous metrics until the promising candidate is fully extracted.
        wp.launch(clear_deficit, 1, inputs=[self.metrics], device=self.device)
        self._launch(
            _extraction.input_ready,
            1,
            [c.domain.error, c.mesh.error, c.full_tetrahedra_count, self._m, self._query_error],
        )
        self._launch(
            _selective.classify_retained,
            self._m,
            [
                c.domain.tree,
                self.points,
                self.full_tetrahedra,
                c.full_tetrahedra_count,
                self._tolerance,
                self._query_error,
                self._query_keep,
                self._query_quality,
            ],
        )
        self._launch(
            _selective.summarize,
            self._m,
            [
                self._query_keep,
                self._query_quality,
                self._query_error,
                self._query_count,
                self._query_minimum,
            ],
        )
        self._launch(
            deficit_full,
            self._m,
            [
                self._query_quality,
                self._query_keep,
                self._query_error,
                self._state.threshold,
                self.metrics,
            ],
        )
        self._launch(
            quality_ready,
            1,
            [
                self._state,
                self._query_minimum,
                self._query_count,
                self._query_error,
                c.boundary.energy,
            ],
        )
        wp.capture_if(self.ready, self._complete_quality)

    def _complete_quality(self):
        e = self.extraction
        e.run()
        self._metrics()
        self._launch(
            accept_trial, 1, [self._state, e.valid, e.error, e.diagnostics, self.core.domain.tree]
        )

    def _commit(self):
        c = self.core
        for a in (
            c.iterations,
            c.termination,
            c.rejected_steps,
            c.last_geometry_error,
            c.last_boundary_error,
            c._max_move,
        ):
            a.zero_()
        c.energy_history.fill_(float("nan"))
        self._launch(history_start, 1, [c.energy, c.boundary.energy, c.energy_history])

    def _restore(self):
        c = self.core
        wp.copy(self.points, self._old_points)
        wp.copy(self._state.active, self._old_active)
        wp.copy(self.full_tetrahedra, self._old_tets)
        wp.copy(c.full_tetrahedra_count, self._old_count)
        c._prepare_mesh()
        c.mesh.error.zero_()
        c._evaluate(True)
        c.boundary.run()
        self._launch(restored_core, 1, [self._state, c._invalid, c.boundary.error, c.error])
        self.extraction.run()
        self._metrics()
        self._launch(
            restored_extraction, 1, [self._state, self.extraction.valid, self.extraction.error]
        )

    def run(self, stream=None):
        stream = wp.get_stream(self.device) if stream is None else stream
        if stream.device != self.device:
            raise ValueError("Quality move stream must use the core device")
        with wp.ScopedStream(stream):
            if stream.is_capturing:
                if stream not in self.device.captures:
                    raise RuntimeError("Use Warp ScopedCapture for conditional graphs")
                self._record()
            else:
                wp.capture_launch(self.graph, stream=stream)
        return self


@wp.kernel
def clear_deficit(metrics: wp.array[wp.float64]):
    metrics[1] = wp.float64(0.0)
