"""Optional retained-tet selection and quality guard for the private VTM core."""

import math
import weakref

import warp as wp

from . import _extraction
from ._surface_bvh import SurfaceTree, closest_point, material_side
from ._wide_center import exact_geometry, uncertain

wp.set_module_options({"enable_backward": False, "fast_math": False})


@wp.kernel
def classify_retained(
    tree: SurfaceTree,
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    count: wp.array[int],
    tolerance: wp.float64,
    error: wp.array[int],
    keep: wp.array[int],
    quality: wp.array[wp.float64],
):
    i = wp.tid()
    keep[i] = 0
    if error[0] != 0 or i >= count[0]:
        return
    ids = wp.vec4i(tets[i, 0], tets[i, 1], tets[i, 2], tets[i, 3])
    for k in range(4):
        if ids[k] < 0 or ids[k] >= points.shape[0]:
            wp.atomic_or(error, 0, 4)
            return
        p = points[ids[k]]
        if not wp.isfinite(p[0]) or not wp.isfinite(p[1]) or not wp.isfinite(p[2]):
            wp.atomic_or(error, 0, 4)
            return
    p0 = wp.vec3d(points[ids[0]])
    a = wp.vec3d(points[ids[1]]) - p0
    b = wp.vec3d(points[ids[2]]) - p0
    c = wp.vec3d(points[ids[3]]) - p0
    bc, ca, ab = wp.cross(b, c), wp.cross(c, a), wp.cross(a, b)
    determinant = wp.dot(a, bc)
    area2 = wp.length(bc) + wp.length(ca) + wp.length(ab) + wp.length(wp.cross(b - a, c - a))
    center, radius = wp.vec3d(), wp.float64(0.0)
    if uncertain(a, b, c, determinant):
        geometry = exact_geometry(points[ids[0]], points[ids[1]], points[ids[2]], points[ids[3]])
        if geometry.valid == 0:
            wp.atomic_or(error, 0, 8)
            return
        determinant = wp.float64(geometry.determinant)
        center = wp.vec3d(geometry.center)
        radius = wp.float64(geometry.radius)
        area2 = wp.float64(geometry.area2)
    else:
        if determinant <= wp.float64(0.0):
            wp.atomic_or(error, 0, 8)
            return
        offset = (wp.dot(a, a) * bc + wp.dot(b, b) * ca + wp.dot(c, c) * ab) / (
            wp.float64(2.0) * determinant
        )
        center, radius = p0 + offset, wp.length(offset)
    if not wp.isfinite(radius) or radius <= wp.float64(0.0):
        wp.atomic_or(error, 0, 8)
        return
    side = material_side(tree, center)
    retained = side == 1
    if not retained or radius >= wp.float64(1.0e140):
        limit = wp.float64(1.0e150)
        # Preserve the full classifier's finite search-limit failure behavior
        # for extreme circumradii as well as ambiguous ray results.
        bounded = side == 0 and tolerance > wp.float64(0.0) and radius < wp.float64(1.0e140)
        if bounded:
            # Expand the *search* cutoff, then apply the original strict d/R
            # predicate below. This avoids changing threshold ties through a
            # rounded product/square compared to the original division.
            limit = wp.min(
                limit,
                wp.max(
                    wp.float64(1.0e-150),
                    tolerance * radius * (wp.float64(1.0) + wp.float64(1.0e-12)),
                ),
            )
        hit = closest_point(tree, center, limit)
        if hit.face < 0:
            if not bounded or limit == wp.float64(1.0e150):
                wp.atomic_or(error, 0, 16)
                return
        else:
            if not wp.isfinite(hit.distance2):
                wp.atomic_or(error, 0, 16)
                return
            # Extraction treats an exact boundary center as inside, including
            # tolerance zero and ambiguous outgoing ray directions.
            if hit.distance2 == wp.float64(0.0):
                retained = True
            elif side != 0 and side != 1:
                wp.atomic_or(error, 0, 16)
                return
            else:
                retained = side == 1 or wp.sqrt(hit.distance2) / radius < tolerance
    keep[i] = int(retained)
    quality[i] = wp.min(wp.float64(1.0), wp.float64(3.0) * determinant / (area2 * radius))


@wp.kernel
def summarize(
    keep: wp.array[int],
    quality: wp.array[wp.float64],
    error: wp.array[int],
    count: wp.array[int],
    minimum: wp.array[wp.float64],
):
    i = wp.tid()
    if error[0] == 0 and keep[i] != 0:
        wp.atomic_add(count, 0, 1)
        wp.atomic_min(minimum, 0, quality[i])


@wp.kernel
def mark_bad(
    tets: wp.array2d[int],
    keep: wp.array[int],
    quality: wp.array[wp.float64],
    threshold: wp.float64,
    error: wp.array[int],
    bad: wp.array[int],
):
    i = wp.tid()
    if error[0] == 0 and keep[i] != 0 and quality[i] < threshold:
        for k in range(4):
            wp.atomic_or(bad, tets[i, k], 1)


@wp.kernel
def expand_one_ring(
    tets: wp.array2d[int],
    keep: wp.array[int],
    error: wp.array[int],
    bad: wp.array[int],
    mask: wp.array[int],
):
    i = wp.tid()
    if error[0] == 0 and keep[i] != 0:
        adjacent = int(0)
        for k in range(4):
            adjacent |= bad[tets[i, k]]
        if adjacent != 0:
            for k in range(4):
                wp.atomic_or(mask, tets[i, k], 1)


@wp.kernel
def count_selected(mask: wp.array[int], roles: wp.array[int], count: wp.array[int]):
    i = wp.tid()
    if mask[i] != 0 and roles[i] != 3:
        wp.atomic_add(count, 0, 1)


@wp.kernel
def ready_selection(
    error: wp.array[int],
    retained: wp.array[int],
    selected: wp.array[int],
    ready: wp.array[int],
    core_error: wp.array[int],
    running: wp.array[int],
    termination: wp.array[int],
):
    ready[0] = int(error[0] == 0 and retained[0] > 0 and selected[0] > 0)
    if error[0] != 0 or retained[0] == 0:
        core_error[0] |= 16
        running[0] = 0
    elif selected[0] == 0:
        termination[0] = 3
        running[0] = 0


@wp.kernel
def masked_directions(
    points: wp.array[wp.vec3],
    targets: wp.array[wp.vec3],
    roles: wp.array[int],
    mass: wp.array[wp.float64],
    moment: wp.array[wp.vec3d],
    edge: wp.array[wp.float64],
    limit: wp.float64,
    output: wp.array[wp.vec3d],
    mask: wp.array[int],
):
    i = wp.tid()
    delta = wp.vec3d(0.0)
    if mask[i] != 0:
        if roles[i] == 1 or roles[i] == 2:
            delta = wp.vec3d(targets[i]) - wp.vec3d(points[i])
        elif roles[i] == 0 and mass[i] > wp.float64(0.0):
            delta = moment[i] / mass[i]
        length = wp.length(delta)
        if length > limit * edge[i]:
            delta *= limit * edge[i] / length
    output[i] = delta


@wp.kernel
def guard_quality(
    error: wp.array[int],
    count: wp.array[int],
    minimum: wp.array[wp.float64],
    previous: wp.array[wp.float64],
    accepted: wp.array[int],
    rejections: wp.array[int],
    last_error: wp.array[int],
):
    failure = int(0)
    if error[0] != 0:
        failure |= 1
    if count[0] == 0:
        failure |= 2
    if minimum[0] < previous[0] * (wp.float64(1.0) - wp.float64(1.0e-10)):
        failure |= 4
    if failure != 0:
        accepted[0] = 0
        rejections[0] += 1
        last_error[0] = failure


@wp.kernel
def recovered_status(source_error: wp.array[int], valid: wp.array[int], error: wp.array[int]):
    error[0] = source_error[0]
    if valid[0] == 0:
        error[0] |= 16


@wp.kernel
def accepted_evaluation(error: wp.array[int], accepted: wp.array[int], needed: wp.array[int]):
    needed[0] = int(error[0] != 0 or accepted[0] == 0)


class SelectiveUpdate:
    def __init__(self, core, threshold, tolerance=0.4, *, enabled=None):
        if not math.isfinite(threshold) or not 0 < threshold <= 1:
            raise ValueError("selective_threshold must be in (0,1]")
        if not math.isfinite(tolerance) or not 0 <= tolerance < 1:
            raise ValueError("selective_tolerance must be finite and in [0,1)")
        self.core, self.device = weakref.proxy(core), core.device
        self.enabled = enabled
        if enabled is not None:
            from ._vtm_controls import scalar

            scalar(enabled, self.device, "selective_enabled")
        self.threshold, self.tolerance = threshold, tolerance
        self._m = len(core.full_tetrahedra)
        self._query_cache = None
        if getattr(core, "_cache_surface_queries", False):
            from ._ordered_surface_queries import OrderedSurfaceQueryCache as SurfaceQueryCache

            self._query_cache = SurfaceQueryCache(core.domain, self._m)
        with wp.ScopedDevice(self.device):
            self.error = wp.zeros(1, dtype=int)
            self.ready = wp.zeros(1, dtype=int)
            self.retained_count = wp.zeros(1, dtype=int)
            self.selected_count = wp.zeros(1, dtype=int)
            self.rejections = wp.zeros(1, dtype=int)
            self.last_error = wp.zeros(1, dtype=int)
            self._current_needed = wp.ones(1, dtype=int)
            self.minimum = wp.ones(1, dtype=wp.float64)
            self.previous_minimum = wp.ones(1, dtype=wp.float64)
            self.keep = wp.empty(self._m, dtype=int)
            self.quality = wp.empty(self._m, dtype=wp.float64)
            self.bad_vertices = wp.zeros(core.n, dtype=int)
            self.mask = wp.zeros(core.n, dtype=int)
            self.recovery_extraction = None
            if core.material_recovery is not None:
                self.recovery_extraction = _extraction._SelectionExtractionWorkspace(
                    core.domain,
                    core.points,
                    core.full_tetrahedra,
                    core.full_tetrahedra_count,
                    source_error=core.mesh.error,
                    tolerance=tolerance,
                    containment_recovery=getattr(core, "containment_recovery", True),
                    exterior_recovery=getattr(core, "exterior_recovery", False),
                    feature_roles=(
                        core.boundary.roles if getattr(core, "exterior_recovery", False) else None
                    ),
                    exterior_enabled=getattr(core, "exterior_enabled", None),
                    proven_exterior_recovery=getattr(core, "proven_exterior_recovery", False),
                    proven_exterior_enabled=getattr(core, "proven_exterior_enabled", None),
                    _cache_surface_queries=getattr(core, "_cache_surface_queries", False),
                )
            wp.load_module(module=__name__, device=self.device)
            wp.load_module(module="warpower._extraction", device=self.device)

    def _launch(self, kernel, dim, args, *, block_dim=256):
        wp.launch(kernel, dim, inputs=args, device=self.device, block_dim=block_dim)

    def reset(self):
        self._invalidate_current()
        self.rejections.zero_()
        self.last_error.zero_()
        self.selected_count.zero_()
        self.mask.zero_()
        self.bad_vertices.zero_()

    def evaluate(self):
        # Explicit evaluations and every trial quality check always recompute.
        self._invalidate_current()
        self._evaluate()

    def _invalidate_current(self):
        self._current_needed.fill_(1)

    def _evaluate(self):
        if self.recovery_extraction is None:
            self._evaluate_ordinary()
        else:
            wp.capture_if(
                self.core.material_recovery, self._evaluate_recovered, self._evaluate_ordinary
            )

    def _evaluate_recovered(self):
        e = self.recovery_extraction
        e.run()
        wp.copy(self.keep, e.selected)
        wp.copy(self.quality, e._input_quality)
        self._launch(recovered_status, 1, [e.error, e.valid, self.error])
        self.retained_count.zero_()
        self.minimum.fill_(1)
        self._launch(
            summarize,
            self._m,
            [self.keep, self.quality, self.error, self.retained_count, self.minimum],
        )

    def _evaluate_ordinary(self):
        core = self.core
        self.retained_count.zero_()
        self.minimum.fill_(1)
        self._launch(
            _extraction.input_ready,
            1,
            [
                core.domain.error,
                core.mesh.error,
                core.full_tetrahedra_count,
                self._m,
                self.error,
            ],
        )
        args = [
            core.domain.tree,
            core.points,
            core.full_tetrahedra,
            core.full_tetrahedra_count,
            self.tolerance,
            self.error,
            self.keep,
            self.quality,
        ]
        if self._query_cache is not None:
            self._query_cache.retained(args)
        else:
            self._launch(classify_retained, self._m, args, block_dim=32)
        self._launch(
            summarize,
            self._m,
            [
                self.keep,
                self.quality,
                self.error,
                self.retained_count,
                self.minimum,
            ],
        )

    def select(self):
        if self.enabled is None:
            self._select()
        else:
            wp.capture_if(self.enabled, self._select, self._select_all)

    def _select_current(self):
        # Only the uninterrupted VTM step schedule may reuse the preceding
        # accepted check. _accept updates ODT statistics, not classification
        # inputs or outputs. Reset, proposal and rollback invalidate before
        # changing geometry; direct select/evaluate calls remain fresh.
        if self.enabled is None:
            self._select(current=True)
        else:
            wp.capture_if(self.enabled, lambda: self._select(current=True), self._select_all)

    def _select_all(self):
        self._invalidate_current()
        self.error.zero_()
        self.ready.fill_(1)
        self.mask.fill_(1)
        self.retained_count.zero_()
        self.minimum.fill_(1)
        self.previous_minimum.fill_(1)

    def _select(self, *, current=False):
        if current:
            wp.capture_if(self._current_needed, self._evaluate)
        else:
            self.evaluate()
        wp.copy(self.previous_minimum, self.minimum)
        self.bad_vertices.zero_()
        self.mask.zero_()
        self.selected_count.zero_()
        self._launch(
            mark_bad,
            self._m,
            [
                self.core.full_tetrahedra,
                self.keep,
                self.quality,
                self.threshold,
                self.error,
                self.bad_vertices,
            ],
        )
        self._launch(
            expand_one_ring,
            self._m,
            [
                self.core.full_tetrahedra,
                self.keep,
                self.error,
                self.bad_vertices,
                self.mask,
            ],
        )
        self._launch(
            count_selected,
            self.core.n,
            [
                self.mask,
                self.core.boundary.roles,
                self.selected_count,
            ],
        )
        self._launch(
            ready_selection,
            1,
            [
                self.error,
                self.retained_count,
                self.selected_count,
                self.ready,
                self.core.error,
                self.core._running,
                self.core.termination,
            ],
        )

    def check(self):
        if self.enabled is None:
            self._check()
        else:
            wp.capture_if(self.enabled, self._check)

    def _check(self):
        self.evaluate()
        self._launch(
            guard_quality,
            1,
            [
                self.error,
                self.retained_count,
                self.minimum,
                self.previous_minimum,
                self.core._accepted,
                self.rejections,
                self.last_error,
            ],
        )
        self._launch(
            accepted_evaluation, 1, [self.error, self.core._accepted, self._current_needed]
        )
