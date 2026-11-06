"""Developing fixed-budget VTM core; public full-mesher API is tracked in #38.

This layer combines evolving boundary constraints and interior ODT updates. It
currently returns full-hull connectivity; domain extraction and initialization
are separate milestones. Its acceptance rule is a boundary-first heuristic,
not a global energy minimization theorem.
"""

import math
import operator

import warp as wp

from ._boundary import BoundaryWorkspace
from ._delaunay_recovery import DelaunayWorkspace
from ._graph import DiagramWorkspace
from ._odt import accumulate_tets

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


@wp.kernel
def start_attempt(
    alpha: wp.array[wp.float64],
    damping: wp.float64,
    attempt: wp.array[int],
    accepted: wp.array[int],
    backtracks: wp.array[int],
):
    alpha[0] = damping
    attempt[0] = 1
    accepted[0] = 0
    backtracks[0] = 0


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
def final_count(error: wp.array[int], count: wp.array[int]):
    if error[0] != 0:
        count[0] = 0


class VTMWorkspace:
    """Internal fixed-budget core with optional sampled sizing."""

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
    ):
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
        self.domain = domain
        mesh_type = DelaunayWorkspace if certified_delaunay else DiagramWorkspace
        mesh_options = {} if certified_delaunay else dict(adjacency=False, return_tetrahedra=True)
        self.mesh = mesh_type(
            points,
            tetrahedra_capacity=tetrahedra_capacity,
            tetrahedra_cell_capacity=tetrahedra_cell_capacity,
            chunk_size=chunk_size,
            **mesh_options,
        )
        self.points, self.device, self.n = self.mesh.points, self.mesh.device, self.mesh.n
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
        self.boundary = BoundaryWorkspace(domain, self.points, **boundary_options)
        self.full_tetrahedra = self.mesh.tetrahedra
        self.full_tetrahedra_count = self.mesh.tetrahedra_count
        with wp.ScopedDevice(self.device):
            self.error = wp.zeros(1, dtype=int)
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
            self._original = wp.empty_like(self.points)
            self._old_tets = wp.empty_like(self.full_tetrahedra)
            self._old_count = wp.empty_like(self.full_tetrahedra_count)
            self._old_energy = wp.empty_like(self.energy)
            self._old_boundary = wp.empty_like(self.boundary.energy)
            wp.load_module(module=__name__, device=self.device)
            with wp.ScopedCapture() as capture:
                self._record()
            self.graph = capture.graph
        wp.synchronize_device(self.device)

    def _launch(self, kernel, dim, args):
        wp.launch(kernel, dim, inputs=args, device=self.device)

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
        self.iterations.zero_()
        self.termination.zero_()
        self.rejected_steps.zero_()
        self.last_geometry_error.zero_()
        self.last_boundary_error.zero_()
        self.energy_history.fill_(float("nan"))
        if self.sizing is not None:
            self._surface_sizing.run()
            self._crease_sizing.run()
        self.boundary.run()
        self._launch(
            anchor_corners,
            self.n,
            [self.points, self.boundary.targets, self.boundary.roles, self.boundary.error],
        )
        self.mesh.run()
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
        wp.capture_while(self._running, self._step)
        self._launch(final_count, 1, [self.error, self.full_tetrahedra_count])

    def _step(self):
        wp.copy(self._original, self.points)
        wp.copy(self._old_tets, self.full_tetrahedra)
        wp.copy(self._old_count, self.full_tetrahedra_count)
        wp.copy(self._old_energy, self.energy)
        wp.copy(self._old_boundary, self.boundary.energy)
        self._launch(
            directions,
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
            ],
        )
        self._launch(
            start_attempt,
            1,
            [self._alpha, self.damping, self._attempt, self._accepted, self._backtracks],
        )
        wp.capture_while(self._attempt, self._try_step)
        wp.capture_if(self._accepted, self._accept, self._restore)

    def _try_step(self):
        self._max_move.zero_()
        self._launch(
            propose,
            self.n,
            [self._original, self.points, self._direction, self._edge, self._alpha, self._max_move],
        )
        self.mesh.run()
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
        self._launch(
            finish_attempt,
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
            ],
        )

    def _accept(self):
        self._evaluate(True)

    def _restore(self):
        wp.copy(self.points, self._original)
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
