"""Captured initialization, growth, fitting, quality improvement, and extraction."""

import math
import operator

import warp as wp

from ._budgeted_sampling import BudgetedSamplingWorkspace
from ._domain_topology import DomainTopology
from ._extraction import ExtractionWorkspace
from ._jitter import BoundaryJitterWorkspace
from ._quality_moves import history_start, sum_euler
from ._refinement import RefinementWorkspace
from ._teleport import InteriorImprovementWorkspace
from ._vtm import VTMWorkspace
from ._vtm_controls import scalar

wp.set_module_options({"enable_backward": False, "fast_math": False})


@wp.struct
class State:
    target: wp.array[int]
    initial_count: wp.array[int]
    active: wp.array[int]
    error: wp.array[int]
    status: wp.array[int]
    ready: wp.array[int]
    grow: wp.array[int]
    recover: wp.array[int]
    quality: wp.array[int]
    rollback: wp.array[int]
    retry: wp.array[int]
    progress: wp.array[int]
    rounds: wp.array[int]
    steps: wp.array[int]
    moves: wp.array[int]
    refinement_rejections: wp.array[int]
    quality_rejections: wp.array[int]
    extraction_reason: wp.array[int]
    last_bad_extraction: wp.array[int]
    last_errors: wp.array[int]
    counts: wp.array[int]
    valid: wp.array[int]
    capacity: int
    initial: int
    automatic_initial: int
    growth_limit: int
    recovery_limit: int
    quality_limit: int


@wp.kernel
def begin(state: State, domain_error: wp.array[int], corners: wp.array[int]):
    state.error[0] = 0
    state.status[0] = 0
    state.active[0] = 0
    state.valid[0] = 0
    state.grow[0] = 0
    state.recover[0] = 0
    state.quality[0] = 0
    state.rollback[0] = 0
    state.retry[0] = 0
    state.progress[0] = 0
    state.refinement_rejections[0] = 0
    state.quality_rejections[0] = 0
    state.extraction_reason[0] = 0
    state.last_bad_extraction[0] = 0
    for k in range(3):
        state.rounds[k] = 0
    for k in range(2):
        state.steps[k] = 0
        state.moves[k] = 0
    for k in range(6):
        state.last_errors[k] = 0
    for k in range(4):
        state.counts[k] = 0
    minimum = state.initial
    if state.automatic_initial != 0:
        minimum = wp.min(minimum, state.target[0])
    # Leave volumetric samples after anchoring where the target permits it;
    # an entire corner set can lie on a plane even for a volumetric domain.
    initial = wp.min(state.target[0], wp.max(minimum, corners[0] + 4))
    state.initial_count[0] = initial
    if (
        domain_error[0] != 0
        or state.target[0] < 4
        or corners[0] > state.target[0]
        or state.target[0] < minimum
        or state.target[0] > state.capacity
    ):
        state.error[0] = 1
    state.ready[0] = int(state.error[0] == 0)


@wp.kernel
def check(state: State, error: wp.array[int], bit: int, slot: int):
    if slot >= 0 and (error[0] != 0 or state.error[0] == 0):
        state.last_errors[slot] = error[0]
    if error[0] != 0:
        state.error[0] |= bit
    state.ready[0] = int(state.error[0] == 0)


@wp.kernel
def sampled(state: State, count: wp.array[int]):
    if count[0] != state.initial_count[0]:
        state.error[0] |= 4
    if state.error[0] == 0:
        state.active[0] = count[0]
    state.ready[0] = int(state.error[0] == 0)


@wp.kernel
def start_growth(state: State):
    state.grow[0] = int(state.active[0] < state.target[0] and state.growth_limit > 0)
    if state.active[0] < state.target[0] and state.growth_limit == 0:
        state.error[0] |= 32
        state.ready[0] = 0


@wp.kernel
def finish_core(state: State, error: wp.array[int], iterations: wp.array[int], mode: int):
    state.steps[mode] += iterations[0]
    state.last_errors[2] = error[0]
    if mode == 0 and error[0] != 0:
        state.error[0] |= 8
    state.ready[0] = int(state.error[0] == 0)


@wp.kernel
def check_refinement(
    state: State,
    error: wp.array[int],
    core_error: wp.array[int],
    target: wp.array[int],
    allow_retry: bool,
):
    state.last_errors[3] = error[0]
    state.retry[0] = 0
    if error[0] != 0:
        state.refinement_rejections[0] += 1
        if error[0] == 4 and core_error[0] == 0:
            if allow_retry:
                target[0] = wp.min(state.active[0] + 1, state.target[0])
                state.retry[0] = 1
        else:
            state.error[0] |= 16
    state.ready[0] = int(state.error[0] == 0)


@wp.kernel
def finish_growth(state: State):
    state.rounds[0] += 1
    state.grow[0] = int(
        state.error[0] == 0
        and state.active[0] < state.target[0]
        and state.rounds[0] < state.growth_limit
    )
    if state.error[0] == 0 and state.active[0] < state.target[0] and state.grow[0] == 0:
        state.error[0] |= 32
    state.ready[0] = int(state.error[0] == 0)


@wp.kernel
def extraction_status(
    state: State,
    error: wp.array[int],
    valid: wp.array[int],
    count: wp.array[int],
    components: wp.array[int],
    shells: wp.array[int],
    cavities: wp.array[int],
    actual: wp.array[int],
    expected: wp.array[int],
):
    actual[0] = components[0]
    actual[1] = shells[0]
    actual[2] = cavities[0]
    reason = int(error[0] != 0 or valid[0] == 0)
    if count[0] == 0:
        reason |= 2
    for k in range(4):
        if actual[k] != expected[k]:
            reason |= 4
    state.extraction_reason[0] = reason


@wp.kernel
def start_recovery(state: State):
    state.recover[0] = int(state.extraction_reason[0] != 0 and state.recovery_limit > 0)


@wp.kernel
def finish_recovery(state: State):
    state.rounds[1] += 1
    state.recover[0] = int(
        state.error[0] == 0
        and state.extraction_reason[0] != 0
        and state.rounds[1] < state.recovery_limit
    )


@wp.kernel
def require_extraction(state: State):
    if state.error[0] == 0 and state.extraction_reason[0] != 0:
        state.error[0] |= 64
    state.ready[0] = int(state.error[0] == 0)


@wp.kernel
def enable_exterior_recovery(state: State, enabled: wp.array[int]):
    enabled[0] = int(state.error[0] == 0 and state.extraction_reason[0] != 0)


@wp.kernel
def start_quality(state: State):
    state.quality[0] = int(state.error[0] == 0 and state.quality_limit > 0)


@wp.kernel
def quality_seed(state: State, seed: wp.array[int], output: wp.array[int]):
    state.progress[0] = 0
    output[0] = wp.int32(
        wp.uint32(seed[0]) + wp.uint32(state.rounds[2] + 1) * wp.uint32(0x9E3779B9)
    )


@wp.kernel
def remember_material_recovery(valid: wp.array[int], changes: wp.array[int], active: wp.array[int]):
    if valid[0] != 0 and changes[0] > 0:
        active[0] = 1


@wp.kernel
def quality_result(
    state: State,
    stage: int,
    source_error: wp.array[int],
    accepted: wp.array[int],
    core_error: wp.array[int],
    mesh_error: wp.array[int],
    boundary_error: wp.array[int],
    invalid: wp.array[int],
    termination: wp.array[int],
    material_recovery: wp.array[int],
    previous_minimum: wp.array[wp.float64],
    current_minimum: wp.array[wp.float64],
):
    slot = 2
    if stage > 0:
        slot = stage + 3
    state.last_errors[slot] = source_error[0]
    failed = (
        source_error[0] != 0
        or core_error[0] != 0
        or mesh_error[0] != 0
        or boundary_error[0] != 0
        or invalid[0] != 0
        or state.extraction_reason[0] != 0
    )
    if stage == 0 and material_recovery[0] != 0:
        failed = failed or current_minimum[0] < previous_minimum[0] * (
            wp.float64(1.0) - wp.float64(1.0e-10)
        )
    state.rollback[0] = int(failed)
    if failed:
        state.quality_rejections[0] += 1
        state.last_bad_extraction[0] = state.extraction_reason[0]
    else:
        if accepted[0] > 0 and (stage > 0 or termination[0] != 1):
            state.progress[0] = 1
        if stage > 0:
            state.moves[stage - 1] += accepted[0]


@wp.kernel
def restored(
    state: State,
    core_error: wp.array[int],
    boundary_error: wp.array[int],
    invalid: wp.array[int],
):
    core_error[0] = 0
    if invalid[0] != 0:
        core_error[0] |= 4
    if boundary_error[0] != 0:
        core_error[0] |= 2
    if core_error[0] != 0:
        state.error[0] |= 128
    state.ready[0] = int(state.error[0] == 0)


@wp.kernel
def restored_extraction(state: State):
    if state.extraction_reason[0] != 0:
        state.error[0] |= 128
    state.ready[0] = int(state.error[0] == 0)


@wp.kernel
def finish_quality(state: State):
    state.rounds[2] += 1
    state.quality[0] = int(
        state.error[0] == 0 and state.progress[0] != 0 and state.rounds[2] < state.quality_limit
    )
    state.status[0] = wp.where(state.progress[0] == 0, 2, 1)


@wp.kernel
def publish(
    state: State,
    full_count: wp.array[int],
    tet_count: wp.array[int],
    boundary_count: wp.array[int],
    topology: wp.array[int],
    volume: wp.array[wp.float64],
    domain_volume: wp.array[wp.float64],
    minimum: wp.array[wp.float64],
    rms: wp.array[wp.float64],
    metrics: wp.array[wp.float64],
):
    if state.error[0] == 0 and state.extraction_reason[0] != 0:
        state.error[0] |= 64
    state.valid[0] = int(state.error[0] == 0)
    if state.error[0] == 0:
        state.counts[0] = state.active[0]
        state.counts[1] = full_count[0]
        state.counts[2] = tet_count[0]
        state.counts[3] = boundary_count[0]
        metrics[0] = volume[0]
        metrics[1] = domain_volume[0]
        metrics[2] = volume[0] - domain_volume[0]
        metrics[3] = minimum[0]
        metrics[4] = rms[0]
    else:
        state.status[0] = 3
        for k in range(4):
            state.counts[k] = 0
            topology[k] = 0
        for k in range(5):
            metrics[k] = wp.float64(0.0)


class VariationalMeshingWorkspace:
    """Fixed-capacity, device-resident Alliez-style variational tetrahedral mesher."""

    def __init__(
        self,
        domain,
        capacity,
        *,
        target_count=None,
        initial_count=None,
        sizing=None,
        automatic_sizing=False,
        automatic_control_subdivisions=0,
        seed=2026,
        growth_iterations=5,
        polish_iterations=20,
        selective_iterations=10,
        max_growth_rounds=None,
        max_recovery_rounds=3,
        quality_rounds=4,
        refinement_fraction=0.5,
        quality_threshold=0.3,
        local_quality_retries=0,
        local_topology_retries=0,
        jitter_attempts=8,
        extraction_tolerance=0.4,
        containment_recovery=True,
        exterior_recovery=True,
        proven_exterior_recovery=None,
        tangential_recovery=None,
        sampling_resolution=32,
        sampling_max_attempts=4096,
        tetrahedra_capacity=None,
        tetrahedra_cell_capacity=1024,
        chunk_size=None,
        boundary_query_precision="float64",
    ):
        self.domain, self.device = domain, domain.device
        self.containment_recovery = bool(containment_recovery)
        self.exterior_recovery = bool(exterior_recovery)
        # Inherit the existing exterior-policy opt-out unless explicitly set.
        self.proven_exterior_recovery = (
            self.exterior_recovery
            if proven_exterior_recovery is None
            else bool(proven_exterior_recovery)
        )
        self.tangential_recovery = (
            self.exterior_recovery if tangential_recovery is None else bool(tangential_recovery)
        )
        if self.tangential_recovery and not self.exterior_recovery:
            raise ValueError("tangential_recovery requires exterior_recovery")
        self.topology_recovery = None
        self._uses_recovery = (
            self.containment_recovery or self.exterior_recovery or self.proven_exterior_recovery
        )
        self.capacity = operator.index(capacity)
        if not 4 <= self.capacity < 2**29:
            raise ValueError("capacity must be in [4,2**29)")
        target = self.capacity if target_count is None else target_count
        if not isinstance(target, wp.array):
            target = operator.index(target)
            if not 4 <= target <= self.capacity:
                raise ValueError("target_count must be in [4,capacity]")
        self.automatic_initial_count = initial_count is None
        if initial_count is None:
            initial_count = min(256, self.capacity if isinstance(target, wp.array) else target)
        self.initial_count = operator.index(initial_count)
        if not 4 <= self.initial_count <= self.capacity:
            raise ValueError("initial_count must be in [4,capacity]")
        if not isinstance(target, wp.array) and target < self.initial_count:
            raise ValueError("target_count must be at least initial_count")
        if not math.isfinite(refinement_fraction) or not 0 < refinement_fraction <= 1:
            raise ValueError("refinement_fraction must be in (0,1]")
        if not math.isfinite(quality_threshold) or not 0 < quality_threshold <= 1:
            raise ValueError("quality_threshold must be in (0,1]")
        jitter_attempts = operator.index(jitter_attempts)
        if not 1 <= jitter_attempts <= 1024:
            raise ValueError("jitter_attempts must be in [1,1024]")
        self.growth_iterations = operator.index(growth_iterations)
        self.polish_iterations = operator.index(polish_iterations)
        self.selective_iterations = operator.index(selective_iterations)
        self.max_recovery_rounds = operator.index(max_recovery_rounds)
        self.quality_rounds = operator.index(quality_rounds)
        self.local_topology_retries = operator.index(local_topology_retries)
        if not 0 <= self.local_topology_retries < 2**31:
            raise ValueError("local_topology_retries must be in [0,2**31)")
        if self.local_topology_retries and (
            self.quality_rounds == 0 or not self._uses_recovery or local_quality_retries != 0
        ):
            raise ValueError(
                "local_topology_retries requires quality rounds, material recovery, "
                "and local_quality_retries=0"
            )
        self.local_quality_retries = operator.index(local_quality_retries)
        if not 0 <= self.local_quality_retries < 2**31:
            raise ValueError("local_quality_retries must be in [0,2**31)")
        if self.local_quality_retries and self.quality_rounds == 0:
            raise ValueError("local_quality_retries requires quality_rounds > 0")
        if max_growth_rounds is None:
            max_growth_rounds = (
                math.ceil(
                    math.log(self.capacity / self.initial_count) / math.log1p(refinement_fraction)
                )
                + 4
            )
        self.max_growth_rounds = operator.index(max_growth_rounds)
        if (
            min(
                self.growth_iterations,
                self.polish_iterations,
                self.selective_iterations,
                self.max_growth_rounds,
                self.max_recovery_rounds,
                self.quality_rounds,
            )
            < 0
            or max(
                self.growth_iterations,
                self.polish_iterations,
                self.selective_iterations,
                self.max_growth_rounds,
                self.max_recovery_rounds,
                self.quality_rounds,
            )
            >= 2**31
        ):
            raise ValueError("Iteration and round budgets must be in [0,2**31)")
        if automatic_sizing and sizing is not None:
            raise ValueError("Choose either automatic_sizing or an explicit sizing field")
        if not isinstance(seed, wp.array):
            seed = operator.index(seed)
            if not -(2**31) <= seed < 2**31:
                raise ValueError("seed must fit int32")
        if sizing is not None:
            from ._sizing import SizingField

            if not isinstance(sizing, SizingField) or sizing.device != self.device:
                raise ValueError("sizing must be a SizingField on the domain device")
        self.automatic = None
        if automatic_sizing:
            from ._automatic_sizing import AutomaticSizingWorkspace

            self.automatic = AutomaticSizingWorkspace(
                domain, control_subdivisions=automatic_control_subdivisions
            )
            sizing = self.automatic.field
        self.sizing = sizing
        with wp.ScopedDevice(self.device):
            self.target_count = (
                scalar(target, self.device, "target_count")
                if isinstance(target, wp.array)
                else wp.full(1, target, dtype=int)
            )
            self.active_count = wp.full(1, self.initial_count, dtype=int)
            self.effective_initial_count = wp.full(1, self.initial_count, dtype=int)
            self.points = wp.zeros(self.capacity, dtype=wp.vec3)
            self._budget = wp.zeros(1, dtype=int)
            self._mode = wp.zeros(1, dtype=int)
            self._material_recovery = wp.zeros(1, dtype=int)
            self._exterior_enabled = wp.zeros(1, dtype=int)
            self._proven_exterior_enabled = (
                wp.zeros(1, dtype=int) if self.proven_exterior_recovery else None
            )
            self._jitter_seed = wp.zeros(1, dtype=int)
            self._refinement_target = wp.full(1, self.capacity, dtype=int)
            self.input_topology = DomainTopology(domain)
            self.sampler = BudgetedSamplingWorkspace(
                domain,
                min(self.capacity, max(self.initial_count, len(domain.points) + 4)),
                self.effective_initial_count,
                sizing=sizing,
                resolution=sampling_resolution,
                seed=seed,
                max_attempts=sampling_max_attempts,
            )
            self.seed = self.sampler.seed
            self.core = VTMWorkspace(
                domain,
                self.points,
                max_iterations=max(
                    self.growth_iterations,
                    self.polish_iterations,
                    self.selective_iterations if self.quality_rounds else 0,
                    int(self.proven_exterior_recovery),
                ),
                active_count=self.active_count,
                sizing=sizing,
                iteration_limit=self._budget,
                selective_threshold=quality_threshold if self.quality_rounds else None,
                selective_tolerance=extraction_tolerance,
                local_quality_retries=self.local_quality_retries,
                selective_enabled=self._mode if self.quality_rounds else None,
                material_recovery=self._material_recovery if self._uses_recovery else None,
                containment_recovery=self.containment_recovery,
                exterior_recovery=self.exterior_recovery,
                exterior_enabled=self._exterior_enabled,
                proven_exterior_recovery=self.proven_exterior_recovery,
                proven_exterior_enabled=self._proven_exterior_enabled,
                tetrahedra_capacity=tetrahedra_capacity,
                tetrahedra_cell_capacity=tetrahedra_cell_capacity,
                chunk_size=chunk_size,
                _cache_surface_queries=True,
                _selective_step_history=True,
                _local_topology_retries=self.local_topology_retries,
                boundary_query_precision=boundary_query_precision,
            )
            self._boundary_rounding = None
            if self.proven_exterior_recovery:
                from ._boundary_rounding import BoundaryRounding

                self._boundary_rounding = BoundaryRounding(self.core)
            self.refinement = None
            if self.initial_count < self.capacity and self.max_growth_rounds:
                self.refinement = RefinementWorkspace(
                    self.core, self._refinement_target, fraction=refinement_fraction
                )
            self.jitter = self.interior = None
            if self.quality_rounds:
                self.jitter = BoundaryJitterWorkspace(
                    self.core,
                    seed=self._jitter_seed,
                    threshold=quality_threshold,
                    max_attempts=jitter_attempts,
                    extraction_tolerance=extraction_tolerance,
                )
                self.interior = InteriorImprovementWorkspace(
                    self.core,
                    target_count=self.target_count,
                    threshold=quality_threshold,
                    extraction_tolerance=extraction_tolerance,
                )
            self._pinched_site = (
                wp.full(1, 2**31 - 1, dtype=int) if self.tangential_recovery else None
            )
            self.extraction = ExtractionWorkspace(
                domain,
                self.points,
                self.core.full_tetrahedra,
                self.core.full_tetrahedra_count,
                source_error=self.core.error,
                tolerance=extraction_tolerance,
                containment_recovery=self.containment_recovery,
                exterior_recovery=self.exterior_recovery,
                feature_roles=self.core.boundary.roles,
                exterior_enabled=self._exterior_enabled,
                proven_exterior_recovery=self.proven_exterior_recovery,
                proven_exterior_enabled=self._proven_exterior_enabled,
                _cache_surface_queries=True,
                _rejected_link_witness=self._pinched_site,
            )
            self.full_tetrahedra = self.core.full_tetrahedra
            self.tetrahedra = self.extraction.tetrahedra
            self.boundary_faces = self.extraction.boundary_faces
            self.radius_ratio = self.extraction.radius_ratio
            self.topology = wp.zeros(4, dtype=int)
            self.metrics = wp.zeros(5, dtype=wp.float64)
            self.component_count, self.shell_count, self.cavity_count, self.euler = (
                self.topology[k : k + 1] for k in range(4)
            )
            (
                self.volume,
                self.domain_volume,
                self.volume_difference,
                self.minimum_radius_ratio,
                self.surface_rms_distance,
            ) = (self.metrics[k : k + 1] for k in range(5))
            state = self._state = State()
            state.target, state.active = self.target_count, self.active_count
            state.capacity, state.initial = self.capacity, self.initial_count
            state.initial_count = self.effective_initial_count
            state.automatic_initial = int(self.automatic_initial_count)
            state.growth_limit, state.recovery_limit, state.quality_limit = (
                self.max_growth_rounds,
                self.max_recovery_rounds,
                self.quality_rounds,
            )
            for name in (
                "error",
                "status",
                "ready",
                "grow",
                "recover",
                "quality",
                "rollback",
                "retry",
                "progress",
                "refinement_rejections",
                "quality_rejections",
                "extraction_reason",
                "last_bad_extraction",
                "valid",
            ):
                value = wp.zeros(1, dtype=int)
                setattr(state, name, value)
                setattr(self, name, value)
            for name, size in (
                ("rounds", 3),
                ("steps", 2),
                ("moves", 2),
                ("last_errors", 6),
                ("counts", 4),
            ):
                value = wp.zeros(size, dtype=int)
                setattr(state, name, value)
                setattr(self, name, value)
            (
                self.point_count,
                self.full_tetrahedra_count,
                self.tetrahedra_count,
                self.boundary_count,
            ) = (self.counts[k : k + 1] for k in range(4))
            if self.quality_rounds:
                self._old_points = wp.empty_like(self.points)
                self._old_tets = wp.empty_like(self.full_tetrahedra)
                self._old_count = wp.empty(1, dtype=int)
                self._old_active = wp.empty(1, dtype=int)
                self._old_minimum = wp.empty(1, dtype=wp.float64)
            self._field_error = wp.zeros(1, dtype=int) if sizing is None else sizing.error
            if self.tangential_recovery:
                from ._topology_recovery import TopologyRecovery

                self.topology_recovery = TopologyRecovery(self, threshold=quality_threshold)
            wp.load_module(module=__name__, device=self.device)
            with wp.ScopedCapture() as cap:
                self._record()
            self.graph = cap.graph
        wp.synchronize_device(self.device)

    def _launch(self, kernel, dim, args):
        wp.launch(kernel, dim, inputs=args, device=self.device)

    def _record(self):
        if self.core.step_history is not None:
            self.core.step_history.statistics.zero_()
        if self.local_topology_retries:
            self.core.local_backtracking.statistics.zero_()
        if self.topology_recovery is not None:
            self.topology_recovery.reset()
        self.topology.zero_()
        self.metrics.zero_()
        self._material_recovery.zero_()
        self._exterior_enabled.zero_()
        if self.proven_exterior_recovery:
            self._proven_exterior_enabled.zero_()
            self._boundary_rounding.statistics.zero_()
        self._launch(begin, 1, [self._state, self.domain.error, self.domain.corner_count])
        wp.capture_if(self.ready, self._initialize)
        e = self.extraction
        self._launch(
            publish,
            1,
            [
                self._state,
                self.core.full_tetrahedra_count,
                e.tetrahedra_count,
                e.boundary_count,
                self.topology,
                e.volume,
                self.input_topology.volume,
                e.minimum_radius_ratio,
                e.surface_rms_distance,
                self.metrics,
            ],
        )

    def _initialize(self):
        self.input_topology.run()
        self._launch(check, 1, [self._state, self.input_topology.error, 1, -1])
        wp.capture_if(self.ready, self._sizing)

    def _sizing(self):
        if self.automatic is not None:
            self.automatic.refit()
            self._launch(check, 1, [self._state, self.automatic.error, 2, 0])
        self._launch(check, 1, [self._state, self._field_error, 2, 0])
        wp.capture_if(self.ready, self._sample)

    def _sample(self):
        self.sampler.run()
        self._launch(check, 1, [self._state, self.sampler.error, 4, 1])
        self._launch(sampled, 1, [self._state, self.sampler.count])
        wp.capture_if(self.ready, self._solve)

    def _solve(self):
        # Fresh samples start a new trajectory. Keep recovery reuse within it,
        # but do not inherit the preceding solve's most expensive strategy.
        self.core.mesh._preferred.zero_()
        self.core.mesh._preferred_level.zero_()
        if self.core.reuses_delaunay_cells:
            self.core.mesh.clear_reuse()
        if self.core.mesh.fast_candidates:
            self.core.mesh._fast_available.fill_(1)
        wp.copy(self.points, self.sampler.points, count=self.sampler.n)
        self._launch(start_growth, 1, [self._state])
        if self.refinement is not None:
            wp.capture_while(self.grow, self._growth)
        wp.capture_if(self.ready, self._polish)

    def _global(self, budget):
        self._budget.fill_(budget)
        self._mode.zero_()
        self.core.run()
        self._launch(finish_core, 1, [self._state, self.core.error, self.core.iterations, 0])

    def _growth(self):
        self._global(self.growth_iterations)
        wp.capture_if(self.ready, self._refine)
        self._launch(finish_growth, 1, [self._state])

    def _refine(self):
        wp.copy(self._refinement_target, self.target_count)
        self.refinement.run()
        self._launch(
            check_refinement,
            1,
            [self._state, self.refinement.error, self.core.error, self._refinement_target, True],
        )
        wp.capture_if(self.retry, self._refine_one)

    def _refine_one(self):
        self.refinement.run()
        self._launch(
            check_refinement,
            1,
            [self._state, self.refinement.error, self.core.error, self._refinement_target, False],
        )

    def _polish(self):
        self._global(self.polish_iterations)
        wp.capture_if(self.ready, self._initial_extract)

    def _extract(self):
        e = self.extraction
        e.run()
        if self.containment_recovery:
            self._launch(
                remember_material_recovery, 1, [e.valid, e.repaired_count, self._material_recovery]
            )
        if self.exterior_recovery:
            self._launch(
                remember_material_recovery, 1, [e.valid, e.removed_count, self._material_recovery]
            )
        if self.proven_exterior_recovery:
            self._launch(
                remember_material_recovery,
                1,
                [e.valid, e.proven_removed_count, self._material_recovery],
            )
        self.topology.zero_()
        self._launch(sum_euler, len(e.shell_euler), [e.shell_euler, self.topology])
        self._launch(
            extraction_status,
            1,
            [
                self._state,
                e.error,
                e.valid,
                e.tetrahedra_count,
                e.component_count,
                e.shell_count,
                e.cavity_count,
                self.topology,
                self.input_topology.counts,
            ],
        )

    def _initial_extract(self):
        self._extract()
        self._launch(start_recovery, 1, [self._state])
        wp.capture_while(self.recover, self._recovery)
        if self.exterior_recovery:
            # Preserve fitting and quality on previously successful trajectories.
            self._launch(enable_exterior_recovery, 1, [self._state, self._exterior_enabled])
            wp.capture_if(self._exterior_enabled, self._extract)
        if self.proven_exterior_recovery:
            self._launch(enable_exterior_recovery, 1, [self._state, self._proven_exterior_enabled])
            wp.capture_if(self._proven_exterior_enabled, self._proven_recovery)
        if self.topology_recovery is not None:
            self.topology_recovery.run()
            wp.capture_if(self.topology_recovery.invoked, self._extract)
        self._launch(require_extraction, 1, [self._state])
        wp.capture_if(self.ready, self._improve)

    def _proven_recovery(self):
        # One extra certified/energy-guarded global step only after ordinary
        # fitting and exterior-band recovery failed. No host replay decisions.
        self._budget.fill_(1)
        self._mode.zero_()
        self.core._record_boundary_recovery(self._boundary_rounding)
        self._launch(finish_core, 1, [self._state, self.core.error, self.core.iterations, 0])
        wp.capture_if(self.ready, self._extract)

    def _recovery(self):
        self._global(self.polish_iterations)
        wp.capture_if(self.ready, self._extract)
        self._launch(finish_recovery, 1, [self._state])

    def _improve(self):
        if self.quality_rounds:
            self._launch(start_quality, 1, [self._state])
            wp.capture_while(self.quality, self._quality_round)

    def _snapshot(self):
        wp.copy(self._old_points, self.points)
        wp.copy(self._old_tets, self.full_tetrahedra)
        wp.copy(self._old_count, self.core.full_tetrahedra_count)
        wp.copy(self._old_active, self.active_count)
        wp.copy(self._old_minimum, self.extraction.minimum_radius_ratio)

    def _assess(self, stage, error, accepted):
        c = self.core
        self._extract()
        self._launch(
            quality_result,
            1,
            [
                self._state,
                stage,
                error,
                accepted,
                c.error,
                c.mesh.error,
                c.boundary.error,
                c._invalid,
                c.termination,
                self._material_recovery,
                self._old_minimum,
                self.extraction.minimum_radius_ratio,
            ],
        )
        wp.capture_if(self.rollback, self._restore)

    def _quality_round(self):
        self._launch(quality_seed, 1, [self._state, self.seed, self._jitter_seed])
        self._snapshot()
        self._budget.fill_(self.selective_iterations)
        self._mode.fill_(1)
        self.core.run()
        self._launch(finish_core, 1, [self._state, self.core.error, self.core.iterations, 1])
        self._assess(0, self.core.error, self.core.iterations)
        wp.capture_if(self.ready, self._jitter)
        wp.capture_if(self.ready, self._interior)
        self._launch(finish_quality, 1, [self._state])

    def _jitter(self):
        self._snapshot()
        self.jitter.run()
        self._assess(1, self.jitter.error, self.jitter.accepted)

    def _interior(self):
        self._snapshot()
        self.interior.run()
        self._assess(2, self.interior.error, self.interior.accepted)

    def _restore(self):
        c = self.core
        wp.copy(self.points, self._old_points)
        wp.copy(self.full_tetrahedra, self._old_tets)
        wp.copy(c.full_tetrahedra_count, self._old_count)
        wp.copy(self.active_count, self._old_active)
        c._prepare_mesh()
        c.mesh.error.zero_()
        c._evaluate(True)
        c.boundary.run()
        self._launch(restored, 1, [self._state, c.error, c.boundary.error, c._invalid])
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
        self._extract()
        self._launch(restored_extraction, 1, [self._state])

    def run(self, stream=None):
        stream = wp.get_stream(self.device) if stream is None else stream
        if stream.device != self.device:
            raise ValueError("Mesher stream must use the domain device")
        with wp.ScopedStream(stream):
            if stream.is_capturing:
                if stream not in self.device.captures:
                    raise RuntimeError("Use Warp ScopedCapture for conditional graphs")
                self._record()
            else:
                wp.capture_launch(self.graph, stream=stream)
        return self
