"""Uniform ODT optimization with fixed convex-hull boundary, entirely on device.

This is the interior optimizer foundation, not the complete domain mesher.
The final input hull is fixed; nonconvex domains need separate boundary fitting
and extraction. Float32 sites use float64 geometric arithmetic and reductions.
"""

import math
import operator

import warp as wp

from ._graph import DiagramWorkspace

wp.set_module_options({"enable_backward": False, "fast_math": False})


@wp.kernel
def pin_hull(
    points: wp.array[wp.vec3],
    faces: wp.array2d[int],
    table: wp.array[int],
    counts: wp.array[int],
    sides: wp.array[int],
    mesh_error: wp.array[int],
    fixed: wp.array[int],
    plane_points: wp.array[wp.vec3d],
    plane_normals: wp.array[wp.vec3d],
    plane_count: wp.array[int],
    invalid: wp.array[int],
):
    slot = wp.tid()
    if mesh_error[0] == 0 and counts[slot] == 1:
        f = table[slot]
        for k in range(3):
            wp.atomic_or(fixed, faces[f, k], 1)
        at = wp.atomic_add(plane_count, 0, 1)
        if at >= plane_points.shape[0]:
            wp.atomic_or(invalid, 0, 1024)
        else:
            p = wp.vec3d(points[faces[f, 0]])
            a = wp.vec3d(points[faces[f, 1]]) - p
            b = wp.vec3d(points[faces[f, 2]]) - p
            plane_points[at] = p
            plane_normals[at] = wp.normalize(wp.cross(a, b)) * wp.float64(-sides[slot])


@wp.kernel
def check_fixed_hull(
    faces: wp.array2d[int],
    table: wp.array[int],
    counts: wp.array[int],
    mesh_error: wp.array[int],
    fixed: wp.array[int],
    invalid: wp.array[int],
):
    slot = wp.tid()
    if mesh_error[0] == 0 and counts[slot] == 1:
        f = table[slot]
        for k in range(3):
            if fixed[faces[f, k]] == 0:
                wp.atomic_or(invalid, 0, 512)


@wp.kernel
def accumulate_tets(
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    count: wp.array[int],
    mesh_error: wp.array[int],
    invalid: wp.array[int],
    mass: wp.array[wp.float64],
    moment: wp.array[wp.vec3d],
    edge: wp.array[wp.float64],
    energy: wp.array[wp.float64],
    volume: wp.array[wp.float64],
    minimum_quality: wp.array[wp.float64],
    directions: bool,
):
    t = wp.tid()
    if t >= count[0] or mesh_error[0] != 0:
        return
    i0, i1, i2, i3 = tets[t, 0], tets[t, 1], tets[t, 2], tets[t, 3]
    p = wp.vec3d(points[i0])
    a, b, c = wp.vec3d(points[i1]) - p, wp.vec3d(points[i2]) - p, wp.vec3d(points[i3]) - p
    cross_bc, cross_ca, cross_ab = wp.cross(b, c), wp.cross(c, a), wp.cross(a, b)
    det = wp.dot(a, cross_bc)
    v = det / wp.float64(6.0)
    if v <= wp.float64(0.0) or not wp.isfinite(v):
        wp.atomic_or(invalid, 0, 256)
        return
    squared_edges = (
        wp.dot(a, a)
        + wp.dot(b, b)
        + wp.dot(c, c)
        + wp.dot(a - b, a - b)
        + wp.dot(a - c, a - c)
        + wp.dot(b - c, b - c)
    )
    e = v * squared_edges / wp.float64(20.0)
    if not wp.isfinite(e):
        wp.atomic_or(invalid, 0, 256)
        return
    wp.atomic_add(energy, 0, e)
    wp.atomic_add(volume, 0, v)
    if directions:
        # V * (circumcenter - p), without division by det. Near-flat tets can
        # have huge circumcenters while this product remains well conditioned.
        vc = (
            wp.dot(a, a) * cross_bc + wp.dot(b, b) * cross_ca + wp.dot(c, c) * cross_ab
        ) / wp.float64(12.0)
        indices = wp.vec4i(i0, i1, i2, i3)
        for k in range(4):
            i = indices[k]
            pi = wp.vec3d(points[i])
            wp.atomic_add(mass, i, v)
            wp.atomic_add(moment, i, vc + v * (p - pi))
            shortest = wp.float64(1.0e300)
            for j in range(4):
                if j != k:
                    shortest = wp.min(shortest, wp.length(pi - wp.vec3d(points[indices[j]])))
            wp.atomic_min(edge, i, shortest)
        # 3 * inradius / circumradius. Evaluating the reciprocal radius avoids
        # constructing the potentially very large circumcenter itself.
        twice_area = (
            wp.length(cross_bc)
            + wp.length(cross_ca)
            + wp.length(cross_ab)
            + wp.length(wp.cross(b - a, c - a))
        )
        q = wp.float64(18.0) * v * v / (twice_area * wp.length(vc))
        wp.atomic_min(minimum_quality, 0, wp.min(wp.float64(1.0), q))


@wp.kernel
def initialize(
    mesh_error: wp.array[int],
    invalid: wp.array[int],
    hull_error: wp.array[int],
    error: wp.array[int],
    energy: wp.array[wp.float64],
    history: wp.array[wp.float64],
    running: wp.array[int],
    max_iterations: int,
):
    error[0] = mesh_error[0] | invalid[0] | hull_error[0]
    history[0] = energy[0]
    running[0] = int(error[0] == 0 and max_iterations > 0)


@wp.kernel
def begin_step(
    energy: wp.array[wp.float64],
    old_energy: wp.array[wp.float64],
    alpha: wp.array[wp.float64],
    damping: wp.float64,
    attempt: wp.array[int],
    accepted: wp.array[int],
    backtracks: wp.array[int],
):
    old_energy[0] = energy[0]
    alpha[0] = damping
    attempt[0] = 1
    accepted[0] = 0
    backtracks[0] = 0


@wp.kernel
def propose(
    original: wp.array[wp.vec3],
    points: wp.array[wp.vec3],
    fixed: wp.array[int],
    mass: wp.array[wp.float64],
    moment: wp.array[wp.vec3d],
    edge: wp.array[wp.float64],
    alpha: wp.array[wp.float64],
    max_move: wp.array[wp.float64],
    step_limit: wp.float64,
    plane_points: wp.array[wp.vec3d],
    plane_normals: wp.array[wp.vec3d],
    plane_count: wp.array[int],
):
    i = wp.tid()
    p = original[i]
    if fixed[i] == 0 and mass[i] > wp.float64(0.0):
        delta = moment[i] / mass[i]
        length = wp.length(delta)
        limit = step_limit * edge[i]
        if length > limit:
            delta *= limit / length
        delta *= alpha[0]
        fraction = wp.float64(1.0)
        # Independently limit each site's motion to the fixed convex domain.
        # A nearly-boundary site must not freeze the entire parallel update.
        for face in range(plane_count[0]):
            n = plane_normals[face]
            outward = wp.dot(n, delta)
            if outward > wp.float64(0.0):
                distance = wp.max(wp.float64(0.0), -wp.dot(n, wp.vec3d(p) - plane_points[face]))
                # Leave room for float32 site storage to round in any direction.
                # Without this margin, a point can round onto/outside a face and
                # force every subsequent global line-search step to be rejected.
                roundoff = wp.float64(8.0e-7) * (
                    wp.length(wp.vec3d(p)) + wp.length(plane_points[face]) + edge[i]
                )
                distance = wp.max(wp.float64(0.0), distance - roundoff)
                fraction = wp.min(fraction, wp.float64(0.9) * distance / outward)
        p = wp.vec3(wp.vec3d(p) + fraction * delta)
        movement = wp.length(wp.vec3d(p) - wp.vec3d(original[i])) / edge[i]
        if not wp.isfinite(movement):
            movement = wp.float64(1.0e300)
        wp.atomic_max(max_move, 0, movement)
    points[i] = p


@wp.kernel
def candidate_ready(
    energy: wp.array[wp.float64],
    old_energy: wp.array[wp.float64],
    invalid: wp.array[int],
    mesh_error: wp.array[int],
    ready: wp.array[int],
):
    # Tolerance is for reduction roundoff, not an optimization stopping rule.
    ready[0] = int(
        invalid[0] == 0
        and mesh_error[0] == 0
        and wp.isfinite(energy[0])
        and energy[0] <= old_energy[0] * wp.float64(1.000000000001)
    )


@wp.kernel
def finish_attempt(
    accepted: wp.array[int],
    attempt: wp.array[int],
    backtracks: wp.array[int],
    alpha: wp.array[wp.float64],
    rejected: wp.array[int],
    iterations: wp.array[int],
    running: wp.array[int],
    termination: wp.array[int],
    energy: wp.array[wp.float64],
    history: wp.array[wp.float64],
    max_move: wp.array[wp.float64],
    tolerance: wp.float64,
    mesh_error: wp.array[int],
    last_geometry_error: wp.array[int],
    max_iterations: int,
    max_backtracks: int,
):
    if accepted[0] != 0:
        attempt[0] = 0
        iterations[0] += 1
        history[iterations[0]] = energy[0]
        converged = max_move[0] <= tolerance
        termination[0] = int(converged)
        running[0] = int(not converged and iterations[0] < max_iterations)
    else:
        if mesh_error[0] != 0:
            last_geometry_error[0] = mesh_error[0]
        rejected[0] += 1
        backtracks[0] += 1
        alpha[0] *= wp.float64(0.5)
        attempt[0] = int(backtracks[0] <= max_backtracks)
        if attempt[0] == 0:
            running[0] = 0
            termination[0] = 2


class ODTWorkspace:
    """Captured uniform ODT optimization with the initial convex hull pinned.

    ``points`` is updated in place. ``fixed`` may pin additional vertices using
    a contiguous int32 Warp mask on the same device. Every run discovers and
    pins the hull of its current input. This is not yet a nonconvex-domain mesher.

    ``termination`` is 0 for the iteration limit, 1 for displacement convergence,
    and 2 for a rejected step after exhausting backtracking. The latter returns
    the last accepted mesh. ``error`` reports unrecoverable initial construction
    errors; ``last_geometry_error`` records a rejected reconstruction failure.
    """

    def __init__(
        self,
        points,
        *,
        fixed=None,
        max_iterations=20,
        damping=0.5,
        tolerance=1.0e-4,
        step_limit=0.5,
        max_backtracks=12,
        tetrahedra_capacity=None,
        tetrahedra_cell_capacity=1024,
        chunk_size=None,
    ):
        self.max_iterations = operator.index(max_iterations)
        self.max_backtracks = operator.index(max_backtracks)
        self.damping, self.tolerance = float(damping), float(tolerance)
        self.step_limit = float(step_limit)
        if self.max_iterations < 0 or self.max_backtracks < 0:
            raise ValueError("Iteration and backtracking limits must be nonnegative")
        if not math.isfinite(self.damping) or self.damping <= 0:
            raise ValueError("damping must be finite and positive")
        if not math.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("tolerance must be finite and nonnegative")
        if not math.isfinite(self.step_limit) or self.step_limit <= 0:
            raise ValueError("step_limit must be finite and positive")
        self.mesh = DiagramWorkspace(
            points,
            adjacency=False,
            return_tetrahedra=True,
            tetrahedra_capacity=tetrahedra_capacity,
            tetrahedra_cell_capacity=tetrahedra_cell_capacity,
            chunk_size=chunk_size,
        )
        self.points, self.device = self.mesh.points, self.mesh.device
        self.n = len(self.points)
        if self.n < 4:
            raise ValueError("ODT requires at least four sites")
        if fixed is not None and (
            not isinstance(fixed, wp.array)
            or fixed.dtype != wp.int32
            or fixed.shape != (self.n,)
            or fixed.device != self.device
            or not fixed.is_contiguous
        ):
            raise ValueError("fixed must be a contiguous int32 Warp mask on the points device")
        self._user_fixed = fixed
        self.tetrahedra = self.mesh.tetrahedra
        self.tetrahedra_count = self.mesh.tetrahedra_count
        with wp.ScopedDevice(self.device):
            self.fixed = wp.zeros(self.n, dtype=int)
            self.error = wp.zeros(1, dtype=int)
            self.iterations = wp.zeros(1, dtype=int)
            self.termination = wp.zeros(1, dtype=int)
            self.rejected_steps = wp.zeros(1, dtype=int)
            self.last_geometry_error = wp.zeros(1, dtype=int)
            self.energy = wp.zeros(1, dtype=wp.float64)
            self.volume = wp.zeros(1, dtype=wp.float64)
            self.minimum_radius_ratio = wp.ones(1, dtype=wp.float64)
            self.energy_history = wp.empty(self.max_iterations + 1, dtype=wp.float64)
            self._mass = wp.zeros(self.n, dtype=wp.float64)
            self._moment = wp.zeros(self.n, dtype=wp.vec3d)
            self._edge = wp.empty(self.n, dtype=wp.float64)
            self._plane_points = wp.empty(2 * self.n, dtype=wp.vec3d)
            self._plane_normals = wp.empty(2 * self.n, dtype=wp.vec3d)
            self._plane_count = wp.zeros(1, dtype=int)
            self._hull_error = wp.zeros(1, dtype=int)
            self._original = wp.empty_like(self.points)
            self._old_tets = wp.empty_like(self.tetrahedra)
            self._old_count = wp.zeros(1, dtype=int)
            self._old_energy = wp.zeros(1, dtype=wp.float64)
            self._alpha = wp.zeros(1, dtype=wp.float64)
            self._max_move = wp.zeros(1, dtype=wp.float64)
            self._running = wp.zeros(1, dtype=int)
            self._attempt = wp.zeros(1, dtype=int)
            self._accepted = wp.zeros(1, dtype=int)
            self._backtracks = wp.zeros(1, dtype=int)
            self._invalid = wp.zeros(1, dtype=int)
            with wp.ScopedCapture(device=self.device) as capture:
                self._record()
            self.graph = capture.graph
            wp.synchronize_stream(wp.get_stream(self.device))

    def _launch(self, kernel, dim, args):
        wp.launch(kernel, dim, inputs=args, device=self.device)

    def _evaluate(self, tets, count, mesh_error, directions=False):
        self._invalid.zero_()
        self.energy.zero_()
        self.volume.zero_()
        if directions:
            self._mass.zero_()
            self._moment.zero_()
            self._edge.fill_(1.0e300)
            self.minimum_radius_ratio.fill_(1.0)
        self._launch(
            accumulate_tets,
            len(tets),
            [
                self.points,
                tets,
                count,
                mesh_error,
                self._invalid,
                self._mass,
                self._moment,
                self._edge,
                self.energy,
                self.volume,
                self.minimum_radius_ratio,
                directions,
            ],
        )

    def _record(self):
        self.error.zero_()
        self.iterations.zero_()
        self.termination.zero_()
        self.rejected_steps.zero_()
        self.last_geometry_error.zero_()
        self.energy_history.fill_(float("nan"))
        self.mesh.run()
        if self._user_fixed is None:
            self.fixed.zero_()
        else:
            wp.copy(self.fixed, self._user_fixed)
        # Reuse the already-validated full-hull face index; no new adjacency
        # construction is needed. These internals are owned by this package.
        self._plane_count.zero_()
        self._hull_error.zero_()
        self._launch(
            pin_hull,
            len(self.mesh._face_table),
            [
                self.points,
                self.mesh._faces,
                self.mesh._face_table,
                self.mesh._face_counts,
                self.mesh._face_sides,
                self.mesh.error,
                self.fixed,
                self._plane_points,
                self._plane_normals,
                self._plane_count,
                self._hull_error,
            ],
        )
        self._evaluate(self.tetrahedra, self.tetrahedra_count, self.mesh.error, True)
        self._launch(
            initialize,
            1,
            [
                self.mesh.error,
                self._invalid,
                self._hull_error,
                self.error,
                self.energy,
                self.energy_history,
                self._running,
                self.max_iterations,
            ],
        )
        wp.capture_while(self._running, self._step)

    def _step(self):
        wp.copy(self._original, self.points)
        wp.copy(self._old_tets, self.tetrahedra)
        wp.copy(self._old_count, self.tetrahedra_count)
        self._launch(
            begin_step,
            1,
            [
                self.energy,
                self._old_energy,
                self._alpha,
                self.damping,
                self._attempt,
                self._accepted,
                self._backtracks,
            ],
        )
        wp.capture_while(self._attempt, self._try_step)
        wp.capture_if(self._accepted, self._accept, self._restore)

    def _try_step(self):
        self._max_move.zero_()
        self._launch(
            propose,
            self.n,
            [
                self._original,
                self.points,
                self.fixed,
                self._mass,
                self._moment,
                self._edge,
                self._alpha,
                self._max_move,
                self.step_limit,
                self._plane_points,
                self._plane_normals,
                self._plane_count,
            ],
        )
        # The old connectivity may invert during a legitimate topology change.
        # Evaluate the rebuilt mesh, then check the fixed domain and energy.
        self._reconstruct()
        self._launch(
            finish_attempt,
            1,
            [
                self._accepted,
                self._attempt,
                self._backtracks,
                self._alpha,
                self.rejected_steps,
                self.iterations,
                self._running,
                self.termination,
                self.energy,
                self.energy_history,
                self._max_move,
                self.tolerance,
                self.mesh.error,
                self.last_geometry_error,
                self.max_iterations,
                self.max_backtracks,
            ],
        )

    def _reconstruct(self):
        self.mesh.run()
        self._evaluate(self.tetrahedra, self.tetrahedra_count, self.mesh.error)
        self._launch(
            check_fixed_hull,
            len(self.mesh._face_table),
            [
                self.mesh._faces,
                self.mesh._face_table,
                self.mesh._face_counts,
                self.mesh.error,
                self.fixed,
                self._invalid,
            ],
        )
        self._launch(
            candidate_ready,
            1,
            [
                self.energy,
                self._old_energy,
                self._invalid,
                self.mesh.error,
                self._accepted,
            ],
        )

    def _accept(self):
        self._evaluate(self.tetrahedra, self.tetrahedra_count, self.mesh.error, True)

    def _restore(self):
        wp.copy(self.points, self._original)
        wp.copy(self.tetrahedra, self._old_tets)
        wp.copy(self.tetrahedra_count, self._old_count)
        self.mesh.error.zero_()
        self._evaluate(self.tetrahedra, self.tetrahedra_count, self.mesh.error, True)

    def run(self, stream=None):
        """Optimize current sites in place, without host reads or allocation."""
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
