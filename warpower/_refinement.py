"""Transactional, device-resident Voronoi-center insertion for the VTM core."""

import math
import operator

import warp as wp
from warp.utils import radix_sort_pairs

from . import _bvh, _canonical
from ._domain import _Scan
from ._site_bvh import nearest_site
from ._sizing import SizingData, sizing_value
from ._stable_center import circumcenter
from ._surface_bvh import SurfaceTree, material_side
from ._tetrahedra import sorted4

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.struct
class State:
    active: wp.array[int]
    target: wp.array[int]
    source_count: wp.array[int]
    error: wp.array[int]
    ready: wp.array[int]
    status: wp.array[int]
    requested: wp.array[int]
    eligible: wp.array[int]
    attempted: wp.array[int]
    inserted: wp.array[int]
    invalid_candidates: wp.array[int]
    point_count: wp.array[int]
    tetrahedra_count: wp.array[int]
    last_errors: wp.array[int]
    capacity: int
    tetrahedra_capacity: int
    fraction: wp.float64


@wp.kernel
def begin(
    state: State,
    core_error: wp.array[int],
    mesh_error: wp.array[int],
    domain_error: wp.array[int],
    field: SizingData,
    graded: bool,
):
    state.error[0] = 0
    state.status[0] = 0
    state.requested[0] = 0
    state.eligible[0] = 0
    state.attempted[0] = 0
    state.inserted[0] = 0
    state.invalid_candidates[0] = 0
    state.point_count[0] = 0
    state.tetrahedra_count[0] = 0
    for k in range(3):
        state.last_errors[k] = 0
    n, target = state.active[0], state.target[0]
    if (
        n < 4
        or n > state.capacity
        or target < n
        or target > state.capacity
        or state.source_count[0] <= 0
        or state.source_count[0] > state.tetrahedra_capacity
        or core_error[0] != 0
        or mesh_error[0] != 0
        or domain_error[0] != 0
    ):
        state.error[0] = 1
    if graded and field.error[0] != 0:
        state.error[0] = 1
    if state.error[0] == 0:
        state.requested[0] = wp.min(
            target - n, wp.max(1, int(wp.ceil(wp.float64(n) * state.fraction)))
        )
        if target == n:
            state.status[0] = 1
    state.ready[0] = int(state.error[0] == 0 and state.requested[0] > 0)


@wp.kernel
def candidates(
    state: State,
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    domain: SurfaceTree,
    sites: _bvh.Tree,
    site_points: wp.array[wp.vec3],
    occupied: wp.array[int],
    field: SizingData,
    graded: bool,
    centers: wp.array[wp.vec3],
    scores: wp.array[wp.float64],
    valid: wp.array[int],
    keys: wp.array[wp.uint64],
    ids: wp.array[int],
):
    i = wp.tid()
    centers[i] = wp.vec3(0.0)
    scores[i] = wp.float64(1.0e300)
    valid[i] = 0
    ids[i] = i
    keys[i] = wp.uint64(0xFFFFFFFFFFFFFFFF)
    if i >= state.source_count[0]:
        return
    vertices = sorted4(wp.vec4i(tets[i, 0], tets[i, 1], tets[i, 2], tets[i, 3]))
    for k in range(4):
        if vertices[k] < 0 or vertices[k] >= state.active[0]:
            wp.atomic_or(state.error, 0, 2)
            return
        p = points[vertices[k]]
        if not wp.isfinite(p[0]) or not wp.isfinite(p[1]) or not wp.isfinite(p[2]):
            wp.atomic_or(state.error, 0, 2)
            return
    keys[i] = (wp.uint64(vertices[2]) << wp.uint64(32)) | wp.uint64(vertices[3])
    # A canonical ordering makes geometric scores independent of tet row order
    # and positive-orientation permutations. Its determinant may have either sign.
    sphere = circumcenter(
        points[vertices[0]], points[vertices[1]], points[vertices[2]], points[vertices[3]]
    )
    if sphere.valid == 0:
        wp.atomic_add(state.invalid_candidates, 0, 1)
        return
    center, radius = sphere.point, sphere.radius
    if material_side(domain, wp.vec3d(center)) != 1:
        return
    nearest = nearest_site(sites, site_points, wp.vec3d(center), state.active, occupied, False)
    if nearest.index < 0 or nearest.distance2 == wp.float64(0.0):
        return
    mu = wp.float64(1.0)
    if graded:
        mu = sizing_value(field, wp.vec3d(center))
    priority = radius / mu
    if mu <= wp.float64(0.0) or not wp.isfinite(priority):
        wp.atomic_add(state.invalid_candidates, 0, 1)
        return
    centers[i] = center
    scores[i] = -priority
    valid[i] = 1


@wp.kernel
def ready_candidates(state: State):
    state.ready[0] = int(state.error[0] == 0)


@wp.kernel
def high_keys(
    tets: wp.array2d[int], count: wp.array[int], ids: wp.array[int], keys: wp.array[wp.uint64]
):
    i = wp.tid()
    source = ids[i]
    key = wp.uint64(0xFFFFFFFFFFFFFFFF)
    if source < count[0]:
        vertices = sorted4(
            wp.vec4i(tets[source, 0], tets[source, 1], tets[source, 2], tets[source, 3])
        )
        key = (wp.uint64(vertices[0]) << wp.uint64(32)) | wp.uint64(vertices[1])
    keys[i] = key


@wp.kernel
def priority_keys(ids: wp.array[int], scores: wp.array[wp.float64], keys: wp.array[wp.float64]):
    i = wp.tid()
    keys[i] = scores[ids[i]]


@wp.kernel
def sorted_candidates(
    ids: wp.array[int],
    centers: wp.array[wp.vec3],
    source_valid: wp.array[int],
    positions: wp.array[wp.vec3],
    valid: wp.array[int],
):
    i = wp.tid()
    source = ids[i]
    positions[i] = centers[source]
    valid[i] = source_valid[source]


@wp.kernel
def unique_candidates(owners: wp.array[int], valid: wp.array[int]):
    i = wp.tid()
    valid[i] = int(valid[i] != 0 and owners[i] == i)


@wp.kernel
def ready_trial(state: State, prefix: wp.array[int]):
    state.eligible[0] = prefix[prefix.shape[0] - 1]
    state.attempted[0] = wp.min(state.requested[0], state.eligible[0])
    state.ready[0] = int(state.attempted[0] > 0)
    if state.attempted[0] == 0:
        state.status[0] = 2


@wp.kernel
def append_sites(
    state: State,
    positions: wp.array[wp.vec3],
    ids: wp.array[int],
    valid: wp.array[int],
    prefix: wp.array[int],
    points: wp.array[wp.vec3],
    selected: wp.array[int],
):
    i = wp.tid()
    if valid[i] != 0 and prefix[i] <= state.attempted[0]:
        rank = prefix[i] - 1
        points[state.active[0] + rank] = positions[i]
        selected[rank] = ids[i]


@wp.kernel
def grow(state: State):
    state.active[0] += state.attempted[0]


@wp.kernel
def accept_trial(
    state: State, mesh_error: wp.array[int], invalid: wp.array[int], boundary_error: wp.array[int]
):
    state.last_errors[0] = mesh_error[0]
    state.last_errors[1] = invalid[0]
    state.last_errors[2] = boundary_error[0]
    state.ready[0] = int(mesh_error[0] == 0 and invalid[0] == 0 and boundary_error[0] == 0)
    if state.ready[0] == 0:
        state.error[0] |= 4
    else:
        state.inserted[0] = state.attempted[0]
        state.status[0] = int(state.active[0] == state.target[0])


@wp.kernel
def history_start(
    energy: wp.array[wp.float64], boundary: wp.array[wp.float64], history: wp.array2d[wp.float64]
):
    history[0, 0] = energy[0]
    history[0, 1] = boundary[0]
    history[0, 2] = boundary[1]


@wp.kernel
def restored(
    state: State, invalid: wp.array[int], boundary_error: wp.array[int], core_error: wp.array[int]
):
    core_error[0] = 0
    if invalid[0] != 0:
        core_error[0] |= 4
    if boundary_error[0] != 0:
        core_error[0] |= 2
    if core_error[0] != 0:
        state.error[0] |= 8
        state.source_count[0] = 0


@wp.kernel
def publish(state: State):
    if state.error[0] == 0:
        state.point_count[0] = state.active[0]
        state.tetrahedra_count[0] = state.source_count[0]
    else:
        state.point_count[0] = 0
        state.tetrahedra_count[0] = 0
        state.inserted[0] = 0
        state.status[0] = 3


class RefinementWorkspace:
    """One captured, transactional insertion batch from a current valid VTM state.

    Run the supplied active-prefix core first, with current geometry and sizing.
    This operation inserts sites and rebuilds connectivity; it does not optimize
    their positions. A stage scheduler can alternate it with core.run().
    """

    def __init__(self, core, target_count=None, *, fraction=0.5):
        if core.active_count is None:
            raise ValueError("Refinement requires an active-prefix VTM core")
        if not math.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError("fraction must be in (0,1]")
        self.core, self.device = core, core.device
        self.points, self.tetrahedra = core.points, core.full_tetrahedra
        self._capacity, self._m = core.n, len(self.tetrahedra)
        with wp.ScopedDevice(self.device):
            if isinstance(target_count, wp.array):
                if (
                    target_count.dtype != wp.int32
                    or target_count.shape != (1,)
                    or target_count.device != self.device
                    or not target_count.is_contiguous
                ):
                    raise ValueError("Target count must be one contiguous device int32")
                self.target_count = target_count
            else:
                target = self._capacity if target_count is None else operator.index(target_count)
                if not 4 <= target <= self._capacity:
                    raise ValueError("Target count must be in [4, point capacity]")
                self.target_count = wp.full(1, target, dtype=int)
            self._state = State()
            self._state.active, self._state.target = core.active_count, self.target_count
            self._state.source_count = core.full_tetrahedra_count
            self._state.capacity, self._state.tetrahedra_capacity = self._capacity, self._m
            self._state.fraction = wp.float64(fraction)
            for name in (
                "error",
                "ready",
                "status",
                "requested",
                "eligible",
                "attempted",
                "inserted",
                "invalid_candidates",
                "point_count",
                "tetrahedra_count",
            ):
                value = wp.zeros(1, dtype=int)
                setattr(self, name, value)
                setattr(self._state, name, value)
            self.last_errors = wp.zeros(3, dtype=int)
            self._state.last_errors = self.last_errors
            self._field = SizingData()
            self._field.error = core.error
            if core.sizing is not None:
                self._field = core.sizing.data
            self._centers = wp.empty(self._m, dtype=wp.vec3)
            self._scores = wp.empty(self._m, dtype=wp.float64)
            self._source_valid = wp.empty(self._m, dtype=int)
            self._keys = wp.empty(2 * self._m, dtype=wp.uint64)
            self._double_keys = wp.array(
                ptr=self._keys.ptr, shape=self._keys.shape, dtype=wp.float64, device=self.device
            )
            self._ids = wp.empty(2 * self._m, dtype=int)
            self._positions = wp.empty_like(self._centers)
            self._valid = wp.empty(self._m, dtype=int)
            self._prefix = wp.empty(self._m, dtype=int)
            self._scan = _Scan(self._m, self.device)
            self._weights = wp.zeros(self._m, dtype=float)
            self._bits = wp.array(
                ptr=self._positions.ptr, shape=(self._m, 3), dtype=wp.uint32, device=self.device
            )
            self._table = wp.empty(1 << (2 * self._m - 1).bit_length(), dtype=int)
            self._buckets, self._owners = wp.empty(self._m, dtype=int), wp.empty(self._m, dtype=int)
            self.selected_tetrahedra = wp.full(self._capacity, -1, dtype=int)
            self._old_points = wp.empty_like(core.points)
            self._old_tets = wp.empty_like(core.full_tetrahedra)
            self._old_count = wp.empty(1, dtype=int)
            self._old_active = wp.empty(1, dtype=int)
            wp.load_module(module=__name__, device=self.device)
            with wp.ScopedCapture() as capture:
                self._record()
            self.graph = capture.graph
        wp.synchronize_device(self.device)

    def _launch(self, kernel, n, args):
        wp.launch(kernel, n, inputs=args, device=self.device)

    def _record(self):
        self.selected_tetrahedra.fill_(-1)
        self._launch(
            begin,
            1,
            [
                self._state,
                self.core.error,
                self.core.mesh.error,
                self.core.domain.error,
                self._field,
                self.core.sizing is not None,
            ],
        )
        wp.capture_if(self.ready, self._candidates)
        self._launch(publish, 1, [self._state])

    def _candidates(self):
        boundary = self.core.boundary
        self._launch(
            candidates,
            self._m,
            [
                self._state,
                self.points,
                self.tetrahedra,
                self.core.domain.tree,
                boundary._hierarchy.tree,
                boundary._safe,
                boundary.corner_owners,
                self._field,
                self.core.sizing is not None,
                self._centers,
                self._scores,
                self._source_valid,
                self._keys,
                self._ids,
            ],
        )
        self._launch(ready_candidates, 1, [self._state])
        wp.capture_if(self.ready, self._sort_select)

    def _sort_select(self):
        # Stable low-pair, high-pair, then float64-priority sorts implement a
        # deterministic (priority, canonical tetrahedron) ordering.
        radix_sort_pairs(self._keys, self._ids, self._m)
        self._launch(
            high_keys,
            self._m,
            [self.tetrahedra, self.core.full_tetrahedra_count, self._ids, self._keys],
        )
        radix_sort_pairs(self._keys, self._ids, self._m)
        self._launch(priority_keys, self._m, [self._ids, self._scores, self._double_keys])
        radix_sort_pairs(self._double_keys, self._ids, self._m)
        self._launch(
            sorted_candidates,
            self._m,
            [self._ids, self._centers, self._source_valid, self._positions, self._valid],
        )
        self._table.fill_(-1)
        self._launch(
            _canonical.insert,
            self._m,
            [self._positions, self._weights, self._bits, self._table, self._buckets],
        )
        self._launch(_canonical.resolve, self._m, [self._table, self._buckets, self._owners])
        self._launch(unique_candidates, self._m, [self._owners, self._valid])
        self._scan.run(self._valid, self._prefix)
        self._launch(ready_trial, 1, [self._state, self._prefix])
        wp.capture_if(self.ready, self._try)

    def _try(self):
        core = self.core
        wp.copy(self._old_points, self.points)
        wp.copy(self._old_tets, self.tetrahedra)
        wp.copy(self._old_count, core.full_tetrahedra_count)
        wp.copy(self._old_active, core.active_count)
        self._launch(
            append_sites,
            self._m,
            [
                self._state,
                self._positions,
                self._ids,
                self._valid,
                self._prefix,
                self.points,
                self.selected_tetrahedra,
            ],
        )
        self._launch(grow, 1, [self._state])
        core._reconstruct()
        core._evaluate(True)
        core.boundary.run()
        self._launch(
            accept_trial, 1, [self._state, core.mesh.error, core._invalid, core.boundary.error]
        )
        wp.capture_if(self.ready, self._accept, self._restore)

    def _accept(self):
        core = self.core
        core.iterations.zero_()
        core.termination.zero_()
        core.rejected_steps.zero_()
        core.last_geometry_error.zero_()
        core.last_boundary_error.zero_()
        core._max_move.zero_()
        core.energy_history.fill_(float("nan"))
        self._launch(history_start, 1, [core.energy, core.boundary.energy, core.energy_history])

    def _restore(self):
        core = self.core
        wp.copy(self.points, self._old_points)
        wp.copy(core.active_count, self._old_active)
        wp.copy(self.tetrahedra, self._old_tets)
        wp.copy(core.full_tetrahedra_count, self._old_count)
        core._prepare_mesh()
        core.mesh.error.zero_()
        core._evaluate(True)
        core.boundary.run()
        self._launch(restored, 1, [self._state, core._invalid, core.boundary.error, core.error])

    def run(self, stream=None):
        stream = wp.get_stream(self.device) if stream is None else stream
        if stream.device != self.device:
            raise ValueError("Refinement stream must use the core device")
        with wp.ScopedStream(stream):
            if stream.is_capturing:
                if stream not in self.device.captures:
                    raise RuntimeError("Use Warp ScopedCapture for conditional graphs")
                self._record()
            else:
                wp.capture_launch(self.graph, stream=stream)
        return self
