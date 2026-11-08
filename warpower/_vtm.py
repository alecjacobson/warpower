"""Developing fixed-budget VTM core; public full-mesher API is tracked in #38.

This layer combines evolving boundary constraints and interior ODT updates. It
returns full-hull connectivity; domain extraction and initialization are separate
services. Its acceptance rule is a boundary-first heuristic,
not a global energy minimization theorem.
"""

import math
import operator

import warp as wp

from ._boundary import BoundaryWorkspace
from ._delaunay_recovery import DelaunayWorkspace
from ._graph import DiagramWorkspace
from ._odt import accumulate_tets
from ._vtm_domain import constrain

wp.set_module_options({"enable_backward": False, "fast_math": False})


@wp.kernel
def anchor_corners(
    points: wp.array[wp.vec3],
    targets: wp.array[wp.vec3],
    roles: wp.array[int],
    error: wp.array[int],
):
    i = wp.tid()
    if error[0] == 0 and roles[i] == 3:
        points[i] = targets[i]


@wp.kernel
def initialize(
    mesh_error: wp.array[int],
    boundary_error: wp.array[int],
    invalid: wp.array[int],
    error: wp.array[int],
    energy: wp.array[wp.float64],
    boundary_energy: wp.array[wp.float64],
    history: wp.array2d[wp.float64],
    running: wp.array[int],
    max_iterations: int,
):
    error[0] = 0
    if mesh_error[0] != 0:
        error[0] = error[0] | 1
    if boundary_error[0] != 0:
        error[0] = error[0] | 2
    if invalid[0] != 0:
        error[0] = error[0] | 4
    history[0, 0] = energy[0]
    history[0, 1] = boundary_energy[0]
    history[0, 2] = boundary_energy[1]
    running[0] = int(error[0] == 0 and max_iterations > 0)


@wp.kernel
def directions(
    points: wp.array[wp.vec3],
    targets: wp.array[wp.vec3],
    roles: wp.array[int],
    mass: wp.array[wp.float64],
    moment: wp.array[wp.vec3d],
    edge: wp.array[wp.float64],
    limit: wp.float64,
    output: wp.array[wp.vec3d],
):
    i = wp.tid()
    delta = wp.vec3d(0.0)
    if roles[i] == 1 or roles[i] == 2:
        delta = wp.vec3d(targets[i]) - wp.vec3d(points[i])
    elif roles[i] == 0 and mass[i] > wp.float64(0.0):
        delta = moment[i] / mass[i]
    length = wp.length(delta)
    if length > limit * edge[i]:
        delta *= limit * edge[i] / length
    output[i] = delta


@wp.func
def _start_attempt(
    alpha: wp.array[wp.float64],
    damping: wp.float64,
    attempt: wp.array[int],
    accepted: wp.array[int],
    backtracks: wp.array[int],
    error: wp.array[int],
    running: wp.array[int],
):
    alpha[0] = damping
    attempt[0] = int(error[0] == 0)
    if error[0] != 0:
        running[0] = 0
    accepted[0] = 0
    backtracks[0] = 0


@wp.kernel
def start_attempt(
    alpha: wp.array[wp.float64],
    damping: wp.float64,
    attempt: wp.array[int],
    accepted: wp.array[int],
    backtracks: wp.array[int],
    error: wp.array[int],
    running: wp.array[int],
):
    _start_attempt(alpha, damping, attempt, accepted, backtracks, error, running)


@wp.kernel
def propose(
    original: wp.array[wp.vec3],
    points: wp.array[wp.vec3],
    delta: wp.array[wp.vec3d],
    edge: wp.array[wp.float64],
    alpha: wp.array[wp.float64],
    maximum: wp.array[wp.float64],
):
    i = wp.tid()
    p = wp.vec3(wp.vec3d(original[i]) + alpha[0] * delta[i])
    points[i] = p
    movement = wp.length(wp.vec3d(p) - wp.vec3d(original[i])) / edge[i]
    if not wp.isfinite(movement):
        movement = wp.float64(1.0e300)
    wp.atomic_max(maximum, 0, movement)


@wp.kernel
def candidate_ready(
    mesh_error: wp.array[int],
    boundary_error: wp.array[int],
    invalid: wp.array[int],
    energy: wp.array[wp.float64],
    old_energy: wp.array[wp.float64],
    boundary_energy: wp.array[wp.float64],
    old_boundary: wp.array[wp.float64],
    accepted: wp.array[int],
):
    accepted[0] = 0
    if mesh_error[0] != 0 or boundary_error[0] != 0 or invalid[0] != 0:
        return
    e, surface, crease = energy[0], boundary_energy[0], boundary_energy[1]
    if not wp.isfinite(e) or not wp.isfinite(surface) or not wp.isfinite(crease):
        return
    # A hierarchy of objectives respects feature priority while the boundary
    # changes the current hull. Only when boundary objectives are stationary do
    # we require the volume interpolation error to decrease as well.
    tolerance = wp.float64(1.0e-10)
    crease_limit = old_boundary[1] * (wp.float64(1.0) + tolerance) + wp.float64(1.0e-30)
    surface_limit = old_boundary[0] * (wp.float64(1.0) + tolerance) + wp.float64(1.0e-30)
    crease_improved = crease < old_boundary[1] * (wp.float64(1.0) - tolerance)
    surface_improved = surface < old_boundary[0] * (wp.float64(1.0) - tolerance)
    accepted[0] = int(
        crease <= crease_limit
        and (
            crease_improved
            or (
                surface <= surface_limit
                and (surface_improved or e <= old_energy[0] * (wp.float64(1.0) + tolerance))
            )
        )
    )


@wp.func
def _finish_attempt(
    mesh_error: wp.array[int],
    boundary_error: wp.array[int],
    accepted: wp.array[int],
    attempt: wp.array[int],
    alpha: wp.array[wp.float64],
    backtracks: wp.array[int],
    rejected: wp.array[int],
    last_geometry: wp.array[int],
    last_boundary: wp.array[int],
    iterations: wp.array[int],
    running: wp.array[int],
    termination: wp.array[int],
    maximum: wp.array[wp.float64],
    tolerance: wp.float64,
    max_iterations: int,
    max_backtracks: int,
    energy: wp.array[wp.float64],
    boundary_energy: wp.array[wp.float64],
    history: wp.array2d[wp.float64],
):
    if accepted[0] != 0:
        attempt[0] = 0
        iterations[0] += 1
        history[iterations[0], 0] = energy[0]
        history[iterations[0], 1] = boundary_energy[0]
        history[iterations[0], 2] = boundary_energy[1]
        converged = maximum[0] <= tolerance
        termination[0] = int(converged)
        running[0] = int(not converged and iterations[0] < max_iterations)
    else:
        if mesh_error[0] != 0:
            last_geometry[0] = mesh_error[0]
        if boundary_error[0] != 0:
            last_boundary[0] = boundary_error[0]
        rejected[0] += 1
        backtracks[0] += 1
        alpha[0] *= wp.float64(0.5)
        attempt[0] = int(backtracks[0] <= max_backtracks)
        if attempt[0] == 0:
            running[0] = 0
            termination[0] = 2


@wp.kernel
def finish_attempt(
    mesh_error: wp.array[int],
    boundary_error: wp.array[int],
    accepted: wp.array[int],
    attempt: wp.array[int],
    alpha: wp.array[wp.float64],
    backtracks: wp.array[int],
    rejected: wp.array[int],
    last_geometry: wp.array[int],
    last_boundary: wp.array[int],
    iterations: wp.array[int],
    running: wp.array[int],
    termination: wp.array[int],
    maximum: wp.array[wp.float64],
    tolerance: wp.float64,
    max_iterations: int,
    max_backtracks: int,
    energy: wp.array[wp.float64],
    boundary_energy: wp.array[wp.float64],
    history: wp.array2d[wp.float64],
):
    _finish_attempt(
        mesh_error,
        boundary_error,
        accepted,
        attempt,
        alpha,
        backtracks,
        rejected,
        last_geometry,
        last_boundary,
        iterations,
        running,
        termination,
        maximum,
        tolerance,
        max_iterations,
        max_backtracks,
        energy,
        boundary_energy,
        history,
    )


@wp.kernel
def final_count(error: wp.array[int], count: wp.array[int]):
    if error[0] != 0:
        count[0] = 0


class VTMWorkspace:
    """Internal VTM core with optional sampled sizing and a device active prefix."""

    def __init__(
        self,
        domain,
        points,
        *,
        max_iterations=20,
        damping=0.5,
        step_limit=0.5,
        tolerance=1e-4,
        max_backtracks=12,
        tetrahedra_capacity=None,
        tetrahedra_cell_capacity=1024,
        chunk_size=None,
        certified_delaunay=True,
        sizing=None,
        active_count=None,
        selective_threshold=None,
        selective_tolerance=0.4,
        local_quality_retries=0,
        iteration_limit=None,
        selective_enabled=None,
        material_recovery=None,
        containment_recovery=True,
        exterior_recovery=False,
        exterior_enabled=None,
        proven_exterior_recovery=False,
        proven_exterior_enabled=None,
        _cache_surface_queries=False,
        _selective_step_history=False,
    ):
        self._cache_surface_queries = bool(_cache_surface_queries)
        self.max_iterations = operator.index(max_iterations)
        self.max_backtracks = operator.index(max_backtracks)
        if self.max_iterations < 0 or self.max_backtracks < 0:
            raise ValueError("Iteration and backtracking limits must be nonnegative")
        if not math.isfinite(damping) or not 0 < damping <= 1:
            raise ValueError("damping must be in (0, 1]")
        if not math.isfinite(step_limit) or step_limit <= 0:
            raise ValueError("step_limit must be positive and finite")
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("tolerance must be nonnegative and finite")
        self.damping, self.step_limit, self.tolerance = damping, step_limit, tolerance
        self.selection = None
        self.local_backtracking = None
        self.step_history = None
        self._start_attempt_kernel, self._start_attempt_args = start_attempt, []
        self._finish_attempt_kernel, self._finish_attempt_args = finish_attempt, []
        self._boundary_rounding = None
        self.controls = None
        local_quality_retries = operator.index(local_quality_retries)
        if not 0 <= local_quality_retries < 2**31:
            raise ValueError("local_quality_retries must be in [0,2**31)")
        if local_quality_retries and selective_threshold is None:
            raise ValueError("local_quality_retries requires selective_threshold")
        if selective_enabled is not None and selective_threshold is None:
            raise ValueError("selective_enabled requires selective_threshold")
        self._directions, self._direction_args = directions, []
        self.domain = domain
        self.active_count = active_count
        self._propose, self._propose_args = propose, []
        mesh_points = points
        if active_count is not None:
            from . import _active_sites
            from ._domain import _contiguous_points

            self._active = _active_sites
            self._input_points = _contiguous_points(points)
            if (
                not isinstance(active_count, wp.array)
                or active_count.dtype != wp.int32
                or active_count.shape != (1,)
                or active_count.device != self._input_points.device
                or not active_count.is_contiguous
            ):
                raise ValueError("active_count must be one contiguous device int32")
            mesh_points = wp.empty_like(self._input_points)
            wp.launch(
                self._active.prepare_mesh,
                len(mesh_points),
                inputs=[self._input_points, active_count, mesh_points],
                device=mesh_points.device,
            )
            self._propose, self._propose_args = self._active.propose_active, [active_count]
        mesh_type = DelaunayWorkspace if certified_delaunay else DiagramWorkspace
        # Bound the exhaustive seed filter to the small-cloud completion range.
        # Keep standalone constructors and large-cloud reconstruction unchanged.
        self.reuses_delaunay_cells = certified_delaunay and 256 <= len(mesh_points) <= 2048
        if self.reuses_delaunay_cells:
            from ._wall_reuse import WallReuseDelaunayWorkspace as ReuseDelaunayWorkspace

            mesh_type = ReuseDelaunayWorkspace
        mesh_options = {} if certified_delaunay else dict(adjacency=False, return_tetrahedra=True)
        if certified_delaunay and active_count is not None and len(mesh_points) > 2048:
            from ._active_delaunay import ActiveDelaunayWorkspace

            mesh_type = ActiveDelaunayWorkspace
            # Device-controlled solves can grow or shrink without reallocating.
            # Reserve fitting children during setup; each retains its complete
            # current-input certificate and the full-constructor fallback.
            mesh_options = dict(
                active_count=active_count,
                prefix_capacities=tuple(
                    n for n in (1024, 2048, 4096, 8192) if n < len(mesh_points)
                ),
            )
            self.reuses_delaunay_cells = True
        self.mesh = mesh_type(
            mesh_points,
            tetrahedra_capacity=tetrahedra_capacity,
            tetrahedra_cell_capacity=tetrahedra_cell_capacity,
            chunk_size=chunk_size,
            **mesh_options,
        )
        self.points = self.mesh.points if active_count is None else self._input_points
        self.device, self.n = self.mesh.device, self.mesh.n
        self.material_recovery = material_recovery
        self.containment_recovery = bool(containment_recovery)
        self.exterior_recovery = bool(exterior_recovery)
        self.exterior_enabled = exterior_enabled
        self.proven_exterior_recovery = bool(proven_exterior_recovery)
        self.proven_exterior_enabled = proven_exterior_enabled
        if material_recovery is not None:
            from ._vtm_controls import scalar

            scalar(material_recovery, self.device, "material_recovery")
        if self.n < 4:
            raise ValueError("Tetrahedral meshing needs at least four sites")
        self.sizing = sizing
        self._accumulate = accumulate_tets
        self._sizing_args = []
        boundary_options = {}
        if sizing is not None:
            from ._graded_odt import accumulate_graded_tets
            from ._sizing import SizingField, SizingQueryWorkspace

            if not isinstance(sizing, SizingField) or sizing.device != self.device:
                raise ValueError("sizing must be a SizingField on the input device")
            self._surface_sizing = SizingQueryWorkspace(sizing, domain.surface_points)
            self._crease_sizing = SizingQueryWorkspace(sizing, domain.crease_points)
            self._accumulate = accumulate_graded_tets
            self._sizing_args = [sizing.data]
            boundary_options = dict(
                surface_sizing=self._surface_sizing.values,
                crease_sizing=self._crease_sizing.values,
            )
            wp.load_module(module="warpower._graded_odt", device=self.device)
        self.boundary = BoundaryWorkspace(
            domain, self.points, active_count=active_count, **boundary_options
        )
        self.full_tetrahedra = self.mesh.tetrahedra
        self.full_tetrahedra_count = self.mesh.tetrahedra_count
        with wp.ScopedDevice(self.device):
            self.error = wp.zeros(1, dtype=int)
            self._active_ready = wp.zeros(1, dtype=int) if active_count is not None else None
            self.iterations = wp.zeros(1, dtype=int)
            self.termination = wp.zeros(1, dtype=int)
            self.rejected_steps = wp.zeros(1, dtype=int)
            self.last_geometry_error = wp.zeros(1, dtype=int)
            self.last_boundary_error = wp.zeros(1, dtype=int)
            self.energy = wp.zeros(1, dtype=wp.float64)
            self.volume = wp.zeros_like(self.energy)
            self.minimum_radius_ratio = wp.ones_like(self.energy)
            self.energy_history = wp.empty((self.max_iterations + 1, 3), dtype=wp.float64)
            self._invalid = wp.zeros(1, dtype=int)
            self._running = wp.zeros(1, dtype=int)
            self._attempt = wp.zeros(1, dtype=int)
            self._accepted = wp.zeros(1, dtype=int)
            self._backtracks = wp.zeros(1, dtype=int)
            self._alpha = wp.zeros(1, dtype=wp.float64)
            self._max_move = wp.zeros_like(self._alpha)
            self._mass = wp.zeros(self.n, dtype=wp.float64)
            self._moment = wp.zeros(self.n, dtype=wp.vec3d)
            self._edge = wp.empty(self.n, dtype=wp.float64)
            self._direction = wp.empty(self.n, dtype=wp.vec3d)
            self._interior_known = wp.zeros(self.n, dtype=int)
            self._original = wp.empty_like(self.points)
            self._old_tets = wp.empty_like(self.full_tetrahedra)
            self._old_count = wp.empty_like(self.full_tetrahedra_count)
            self._old_energy = wp.empty_like(self.energy)
            self._old_boundary = wp.empty_like(self.boundary.energy)
            if selective_threshold is not None:
                from ._selective import SelectiveUpdate, masked_directions

                self.selection = SelectiveUpdate(
                    self, selective_threshold, selective_tolerance, enabled=selective_enabled
                )
                self._directions, self._direction_args = masked_directions, [self.selection.mask]
                if local_quality_retries:
                    from ._selective_backtracking import LocalQualityBacktracking

                    self.local_backtracking = LocalQualityBacktracking(self, local_quality_retries)
            if iteration_limit is not None or selective_enabled is not None:
                from ._vtm_controls import DeviceControls

                self.controls = DeviceControls(self, iteration_limit, selective_enabled)
            if (
                _selective_step_history
                and selective_enabled is not None
                and local_quality_retries == 0
            ):
                from ._selective_step_history import SelectiveStepHistory

                self.step_history = SelectiveStepHistory(self)
            wp.load_module(module="warpower._vtm_domain", device=self.device, block_dim=32)
            wp.load_module(module=__name__, device=self.device)
            with wp.ScopedCapture() as capture:
                self._record()
            self.graph = capture.graph
        wp.synchronize_device(self.device)

    def _launch(self, kernel, dim, args, *, block_dim=256):
        wp.launch(kernel, dim, inputs=args, device=self.device, block_dim=block_dim)

    def _evaluate(self, compute_directions):
        self._invalid.zero_()
        self.energy.zero_()
        self.volume.zero_()
        if compute_directions:
            self._mass.zero_()
            self._moment.zero_()
            self._edge.fill_(1e300)
            self.minimum_radius_ratio.fill_(1)
        self._launch(
            self._accumulate,
            len(self.full_tetrahedra),
            [
                self.points,
                self.full_tetrahedra,
                self.full_tetrahedra_count,
                self.mesh.error,
                self._invalid,
                self._mass,
                self._moment,
                self._edge,
                self.energy,
                self.volume,
                self.minimum_radius_ratio,
                compute_directions,
            ]
            + self._sizing_args,
        )

    def _record(self):
        if self.step_history is not None:
            self.step_history.reset()
        self._interior_known.zero_()
        if self.local_backtracking is not None:
            self.local_backtracking.reset()
        if self.selection is not None:
            self.selection.reset()
        self.iterations.zero_()
        self.termination.zero_()
        self.rejected_steps.zero_()
        self.last_geometry_error.zero_()
        self.last_boundary_error.zero_()
        self.energy_history.fill_(float("nan"))
        if self.active_count is None:
            self._record_valid()
        else:
            self._launch(
                self._active.validate,
                1,
                [self.active_count, self.n, self.error, self._active_ready],
            )
            self.full_tetrahedra_count.zero_()
            self.energy.zero_()
            self.volume.zero_()
            self.minimum_radius_ratio.zero_()
            wp.capture_if(self._active_ready, self._record_valid)

    def _prepare_mesh(self):
        if self.active_count is not None:
            self._launch(
                self._active.prepare_mesh,
                self.n,
                [self.points, self.active_count, self.mesh.points],
            )

    def _reconstruct(self):
        self._prepare_mesh()
        self.mesh.run()

    def _record_valid(self):
        if self.sizing is not None:
            self._surface_sizing.run()
            self._crease_sizing.run()
        self.boundary.run()
        self._launch(
            anchor_corners,
            self.n,
            [self.points, self.boundary.targets, self.boundary.roles, self.boundary.error],
        )
        self._reconstruct()
        self._evaluate(True)
        self.boundary.run()
        self._launch(
            initialize,
            1,
            [
                self.mesh.error,
                self.boundary.error,
                self._invalid,
                self.error,
                self.energy,
                self.boundary.energy,
                self.energy_history,
                self._running,
                self.max_iterations,
            ],
        )
        if self.controls is not None:
            self.controls.gate()
        wp.capture_while(self._running, self._step)
        self._launch(final_count, 1, [self.error, self.full_tetrahedra_count])

    def _step(self):
        if self.selection is not None:
            self.selection._select_current()
            wp.capture_if(self.selection.ready, self._step_valid)
        else:
            self._step_valid()
        if self.controls is not None:
            self.controls.gate()

    def _step_valid(self):
        if self.local_backtracking is not None:
            self.local_backtracking.start_step()
        wp.copy(self._original, self.points)
        wp.copy(self._old_tets, self.full_tetrahedra)
        wp.copy(self._old_count, self.full_tetrahedra_count)
        wp.copy(self._old_energy, self.energy)
        wp.copy(self._old_boundary, self.boundary.energy)
        self._launch(
            self._directions,
            self.n,
            [
                self.points,
                self.boundary.targets,
                self.boundary.roles,
                self._mass,
                self._moment,
                self._edge,
                self.step_limit,
                self._direction,
            ]
            + self._direction_args,
        )
        if self._boundary_rounding is not None:
            self._boundary_rounding.freeze(self)
        self._launch(
            constrain,
            self.n,
            [
                self.domain.tree,
                self.points,
                self.boundary.roles,
                self._direction,
                self._edge,
                self.step_limit,
                self.damping,
                self.error,
                self._interior_known,
            ],
            block_dim=32,
        )
        self._launch(
            self._start_attempt_kernel,
            1,
            [
                self._alpha,
                self.damping,
                self._attempt,
                self._accepted,
                self._backtracks,
                self.error,
                self._running,
            ]
            + self._start_attempt_args,
        )
        wp.capture_while(self._attempt, self._try_step)
        wp.capture_if(self._accepted, self._accept, self._restore)

    def _try_step(self):
        if self.selection is not None:
            self.selection._invalidate_current()
        if self.local_backtracking is not None:
            self.local_backtracking.begin_attempt()
        self._max_move.zero_()
        proposal, arguments = self._propose, self._propose_args
        if self._boundary_rounding is not None:
            # This variant is recorded only for a global recovery step. Local
            # selective blocking is reset and ineligible in that stage.
            proposal, arguments = self._boundary_rounding.propose, self._boundary_rounding.args
        self._launch(
            proposal,
            self.n,
            [self._original, self.points, self._direction, self._edge, self._alpha, self._max_move]
            + arguments,
        )
        self._reconstruct()
        self._evaluate(False)
        self.boundary.run()
        self._launch(
            candidate_ready,
            1,
            [
                self.mesh.error,
                self.boundary.error,
                self._invalid,
                self.energy,
                self._old_energy,
                self.boundary.energy,
                self._old_boundary,
                self._accepted,
            ],
        )
        if self.selection is not None:
            wp.capture_if(self._accepted, self.selection.check)
        if self.local_backtracking is None:
            self._finish_attempt()
        else:
            self.local_backtracking.finish(self._finish_attempt)

    def _finish_attempt(self):
        self._launch(
            self._finish_attempt_kernel,
            1,
            [
                self.mesh.error,
                self.boundary.error,
                self._accepted,
                self._attempt,
                self._alpha,
                self._backtracks,
                self.rejected_steps,
                self.last_geometry_error,
                self.last_boundary_error,
                self.iterations,
                self._running,
                self.termination,
                self._max_move,
                self.tolerance,
                self.max_iterations,
                self.max_backtracks,
                self.energy,
                self.boundary.energy,
                self.energy_history,
            ]
            + self._finish_attempt_args,
        )

    def _accept(self):
        self._evaluate(True)
        if self._boundary_rounding is not None:
            self._boundary_rounding.accepted(self)

    def _record_boundary_recovery(self, rounding):
        # Python chooses the recorded kernels, never a replay-time outcome.
        # Restore the ordinary recording path even if capture raises.
        previous = self._boundary_rounding
        self._boundary_rounding = rounding
        try:
            self._record()
        finally:
            self._boundary_rounding = previous

    def _restore(self):
        if self.selection is not None:
            self.selection._invalidate_current()
        wp.copy(self.points, self._original)
        self._prepare_mesh()
        wp.copy(self.full_tetrahedra, self._old_tets)
        wp.copy(self.full_tetrahedra_count, self._old_count)
        self.mesh.error.zero_()
        self._evaluate(True)
        self.boundary.run()

    def run(self, stream=None):
        stream = wp.get_stream(self.device) if stream is None else stream
        if stream.device != self.device:
            raise ValueError("Workspace stream must use the input device")
        with wp.ScopedStream(stream):
            if stream.is_capturing:
                if stream not in self.device.captures:
                    raise RuntimeError("Use Warp ScopedCapture for conditional graphs")
                self._record()
            else:
                wp.capture_launch(self.graph, stream=stream)
        return self
