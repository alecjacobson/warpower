"""One transactional interior-site teleportation or centroid insertion."""

import operator

import warp as wp

from ._jitter import high_key, low_key, select_tet
from ._quality_moves import MoveState, QualityMove
from ._surface_bvh import SurfaceTree, material_side
from ._tetrahedra import sorted4
from ._voronoi_volume import BoundedVoronoiVolumes

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.kernel
def configure(state: MoveState, target: wp.array[int], mode: wp.array[int]):
    if target[0] < state.active[0] or target[0] > state.capacity:
        state.error[0] |= 32
    mode[0] = int(target[0] > state.active[0])
    state.ready[0] = int(state.error[0] == 0)


@wp.kernel
def candidates(
    state: MoveState,
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    count: wp.array[int],
    quality: wp.array[wp.float64],
    roles: wp.array[int],
    tree: SurfaceTree,
    eligible: wp.array[int],
    centers: wp.array[wp.vec3],
    minimum: wp.array[wp.float64],
):
    t = wp.tid()
    eligible[t] = 0
    if t >= count[0] or quality[t] >= state.threshold:
        return
    ids = sorted4(wp.vec4i(tets[t, 0], tets[t, 1], tets[t, 2], tets[t, 3]))
    for k in range(4):
        if roles[ids[k]] != 0:
            return
    # Canonical IDs fix the summation order under equivalent mesh permutations.
    p = wp.vec3d(0.0)
    for k in range(4):
        p += wp.vec3d(points[ids[k]])
    center = wp.vec3(p * wp.float64(0.25))
    if material_side(tree, wp.vec3d(center)) != 1:
        return
    centers[t] = center
    eligible[t] = 1
    wp.atomic_min(minimum, 0, quality[t])


@wp.kernel
def route(
    state: MoveState,
    row: wp.array[int],
    mode: wp.array[int],
    centers: wp.array[wp.vec3],
    donor_ready: wp.array[int],
):
    state.running[0] = 0
    donor_ready[0] = 0
    if row[0] >= 0:
        state.position[0] = centers[row[0]]
        if mode[0] == 1:
            state.site[0] = state.active[0]
            state.running[0] = 1
        else:
            state.site[0] = state.capacity
            donor_ready[0] = 1


@wp.kernel
def donor_minimum(
    state: MoveState,
    normalized: wp.array[wp.float64],
    valid: wp.array[int],
    roles: wp.array[int],
    selected: wp.array[int],
    minimum: wp.array[wp.float64],
    eligible: wp.array[int],
):
    i = wp.tid()
    eligible[i] = 0
    if i >= state.active[0] or roles[i] != 0 or valid[i] == 0:
        return
    for k in range(4):
        if i == selected[k]:
            return
    eligible[i] = 1
    wp.atomic_min(minimum, 0, normalized[i])


@wp.kernel
def donor_index(
    state: MoveState,
    normalized: wp.array[wp.float64],
    eligible: wp.array[int],
    minimum: wp.array[wp.float64],
):
    i = wp.tid()
    if eligible[i] != 0 and normalized[i] == minimum[0]:
        wp.atomic_min(state.site, 0, i)


@wp.kernel
def finish_donor(state: MoveState, volume_error: wp.array[int]):
    if volume_error[0] != 0:
        state.error[0] |= 16
    state.running[0] = int(state.site[0] < state.active[0] and state.error[0] == 0)
    if state.running[0] == 0:
        state.site[0] = -1


@wp.kernel
def proposal(state: MoveState):
    p = state.position[0]
    state.proposal_valid[0] = int(
        state.site[0] >= 0 and wp.isfinite(p[0]) and wp.isfinite(p[1]) and wp.isfinite(p[2])
    )


@wp.kernel
def apply(state: MoveState, points: wp.array[wp.vec3], mode: wp.array[int]):
    points[state.site[0]] = state.position[0]
    if mode[0] == 1:
        state.active[0] += 1


class InteriorImprovementWorkspace(QualityMove):
    """Move the densest eligible interior site, or append a site if budget permits."""

    def __init__(self, core, *, target_count=None, **quality_options):
        if quality_options.pop("max_attempts", 1) != 1:
            raise ValueError("Interior improvement makes exactly one candidate trial")
        super().__init__(core, max_attempts=1, **quality_options)
        self.cells = BoundedVoronoiVolumes(core)
        with wp.ScopedDevice(self.device):
            if target_count is None:
                # Omitting the target preserves the current active count.
                self.target_count = self._state.active
            elif isinstance(target_count, wp.array):
                if (
                    target_count.dtype != wp.int32
                    or target_count.shape != (1,)
                    or target_count.device != self.device
                    or not target_count.is_contiguous
                ):
                    raise ValueError("Target must be one contiguous device int32")
                if core.active_count is None:
                    raise ValueError("A target requires core.active_count")
                self.target_count = target_count
            else:
                target_count = operator.index(target_count)
                if core.active_count is None:
                    raise ValueError("A target requires core.active_count")
                if not 4 <= target_count <= core.n:
                    raise ValueError("Target must be within [4,capacity]")
                self.target_count = wp.full(1, target_count, dtype=int)
            self.mode = wp.zeros(1, dtype=int)
            self.inserted = wp.zeros(1, dtype=int)
            self.source_tetrahedron = wp.full(1, -1, dtype=int)
            self.canonical_tetrahedron = wp.full(4, -1, dtype=int)
            self._eligible = wp.zeros(self._m, dtype=int)
            self._centers = wp.zeros(self._m, dtype=wp.vec3)
            self._minimum = wp.ones(1, dtype=wp.float64)
            self._high = wp.zeros(1, dtype=wp.uint64)
            self._low = wp.zeros(1, dtype=wp.uint64)
            self._donor_ready = wp.zeros(1, dtype=int)
            self._donor_minimum = wp.zeros(1, dtype=wp.float64)
            self._donor_eligible = wp.zeros(core.n, dtype=int)
            wp.load_module(module=__name__, device=self.device)
            wp.load_module(module="warpower._jitter", device=self.device)
        self._finish_setup()

    def _record(self):
        self.inserted.zero_()
        super()._record()

    def _prepare_proposal(self):
        self.source_tetrahedron.fill_(-1)
        self.canonical_tetrahedron.fill_(-1)
        self._launch(configure, 1, [self._state, self.target_count, self.mode])
        wp.capture_if(self.ready, self._select)

    def _select(self):
        e = self.extraction
        self._minimum.fill_(1)
        self._high.fill_(0xFFFFFFFFFFFFFFFF)
        self._low.fill_(0xFFFFFFFFFFFFFFFF)
        self._launch(
            candidates,
            self._m,
            [
                self._state,
                self.points,
                e.tetrahedra,
                e.tetrahedra_count,
                e.radius_ratio,
                self.core.boundary.roles,
                self.core.domain.tree,
                self._eligible,
                self._centers,
                self._minimum,
            ],
        )
        self._launch(
            high_key,
            self._m,
            [e.tetrahedra, e.radius_ratio, self._eligible, self._minimum, self._high],
        )
        self._launch(
            low_key,
            self._m,
            [e.tetrahedra, e.radius_ratio, self._eligible, self._minimum, self._high, self._low],
        )
        self._launch(
            select_tet,
            self._m,
            [
                e.tetrahedra,
                e.radius_ratio,
                self._eligible,
                self._minimum,
                self._high,
                self._low,
                self.source_tetrahedron,
                self.canonical_tetrahedron,
            ],
        )
        self._launch(
            route,
            1,
            [self._state, self.source_tetrahedron, self.mode, self._centers, self._donor_ready],
        )
        wp.capture_if(self._donor_ready, self._donor)

    def _donor(self):
        self.cells.run()
        self._donor_minimum.fill_(float("inf"))
        self._launch(
            donor_minimum,
            self.core.n,
            [
                self._state,
                self.cells.normalized,
                self.cells.valid,
                self.core.boundary.roles,
                self.canonical_tetrahedron,
                self._donor_minimum,
                self._donor_eligible,
            ],
        )
        self._launch(
            donor_index,
            self.core.n,
            [self._state, self.cells.normalized, self._donor_eligible, self._donor_minimum],
        )
        self._launch(finish_donor, 1, [self._state, self.cells.error])

    def _propose(self):
        self._launch(proposal, 1, [self._state])

    def _apply(self):
        self._launch(apply, 1, [self._state, self.points, self.mode])

    def _commit(self):
        super()._commit()
        wp.copy(self.inserted, self.mode)
