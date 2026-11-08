"""Bounded quality recovery after a rejected boundary-link transaction.

Every accepted proposal has a fresh complete Delaunay certificate and passes the
existing quality-move bounds. Two deterministic rings stay within 10/20% of a
local edge. The strict global VTM fitting rule is unchanged.
"""

import weakref

import warp as wp

from . import _mesher as mesher
from . import _quality_moves as qm
from ._surface_bvh import SurfaceTree, closest_point

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.kernel
def reset_state(state: qm.MoveState, invoked: wp.array[int], chosen: wp.array[int]):
    invoked[0] = 0
    chosen[0] = -1
    state.error[0] = 0
    state.status[0] = 1
    state.ready[0] = 0
    state.running[0] = 0
    state.accepted[0] = 0
    state.attempts[0] = 0
    state.invalid_proposals[0] = 0
    state.proposal_valid[0] = 0
    state.rejection_reason[0] = 0
    state.site[0] = -1
    state.position[0] = wp.vec3(0.0)
    state.point_count[0] = 0
    state.full_count[0] = 0
    state.retained_count[0] = 0
    for k in range(5):
        state.last_errors[k] = 0
    for k in range(6):
        state.current[k] = wp.float64(0.0)
        state.previous[k] = wp.float64(0.0)
    for k in range(4):
        state.topology[k] = 0
        state.previous_topology[k] = 0


@wp.kernel
def eligible(
    state: mesher.State,
    core_error: wp.array[int],
    extraction_error: wp.array[int],
    witness: wp.array[int],
    band_rejected: wp.array[int],
    roles: wp.array[int],
    chosen: wp.array[int],
    ready: wp.array[int],
):
    ready[0] = 0
    chosen[0] = -1
    site = witness[0]
    if (
        state.error[0] == 0
        and state.extraction_reason[0] != 0
        and core_error[0] == 0
        and extraction_error[0] == 0
        and band_rejected[0] != 0
    ):
        if site >= 0 and site < state.active[0] and roles[site] == 1:
            chosen[0] = site
            ready[0] = 1


@wp.kernel
def initial_invalid_ready(
    state: qm.MoveState,
    error: wp.array[int],
    diagnostics: wp.array[int],
    count: wp.array[int],
    invalid: wp.array[int],
    boundary_error: wp.array[int],
):
    state.ready[0] = int(
        error[0] == 0
        and count[0] > 0
        and (diagnostics[0] == 2 or diagnostics[0] == 4)
        and invalid[0] == 0
        and boundary_error[0] == 0
    )


@wp.kernel
def mark_references(tets: wp.array2d[int], count: wp.array[int], flags: wp.array[int]):
    i = wp.tid()
    if i < count[0]:
        for k in range(4):
            wp.atomic_or(flags, tets[i, k], 1)


@wp.kernel
def references_and_roles(
    before: wp.array[int],
    after: wp.array[int],
    old_roles: wp.array[int],
    new_roles: wp.array[int],
    active: wp.array[int],
    rejected: wp.array[int],
):
    i = wp.tid()
    if i < active[0]:
        if (before[i] != 0 and after[i] == 0) or old_roles[i] != new_roles[i]:
            wp.atomic_or(rejected, 0, 1)


@wp.kernel
def reject_changed_references(state: qm.MoveState, rejected: wp.array[int]):
    if rejected[0] != 0:
        state.accepted[0] = 0
        state.rejection_reason[0] |= 256


@wp.kernel
def prepare(
    state: qm.MoveState,
    tree: SurfaceTree,
    points: wp.array[wp.vec3],
    chosen: wp.array[int],
    roles: wp.array[int],
    edge: wp.array[wp.float64],
    frame: wp.array[wp.vec3d],
    limit: wp.array[wp.float64],
):
    state.running[0] = 0
    site = chosen[0]
    if site < 0 or site >= state.active[0] or roles[site] != 1:
        return
    length = edge[site]
    if not wp.isfinite(length) or length <= wp.float64(0.0) or length >= wp.float64(1.0e299):
        return
    hit = closest_point(tree, wp.vec3d(points[site]), wp.float64(1.0e150))
    if hit.face < 0:
        return
    a = wp.vec3d(tree.points[tree.indices[3 * hit.face]])
    b = wp.vec3d(tree.points[tree.indices[3 * hit.face + 1]])
    c = wp.vec3d(tree.points[tree.indices[3 * hit.face + 2]])
    normal = wp.cross(b - a, c - a)
    size = wp.length(normal)
    if not wp.isfinite(size) or size <= wp.float64(0.0):
        return
    normal /= size
    axis = wp.vec3d(1.0, 0.0, 0.0)
    if wp.abs(normal[0]) > wp.float64(0.8):
        axis = wp.vec3d(0.0, 1.0, 0.0)
    frame[0] = normal
    frame[1] = wp.normalize(wp.cross(normal, axis))
    frame[2] = wp.cross(normal, frame[1])
    limit[0] = wp.float64(0.2) * length
    state.running[0] = 1


@wp.kernel
def propose(
    state: qm.MoveState,
    original: wp.array[wp.vec3],
    chosen: wp.array[int],
    frame: wp.array[wp.vec3d],
    limit: wp.array[wp.float64],
):
    site = chosen[0]
    step = state.attempts[0] - 1
    angle = wp.float64(step % 16) * wp.float64(0.39269908169872415481)
    scale = wp.float64(0.9999)
    if step < 16:
        scale *= wp.float64(0.5)
    offset = (scale * limit[0]) * (wp.cos(angle) * frame[1] + wp.sin(angle) * frame[2])
    position = wp.vec3(wp.vec3d(original[site]) + offset)
    moved = wp.length(wp.vec3d(position) - wp.vec3d(original[site]))
    state.site[0] = site
    state.position[0] = position
    # Float32 rounding can exceed the intended radius: reject, never enlarge it.
    state.proposal_valid[0] = int(
        wp.isfinite(position[0])
        and wp.isfinite(position[1])
        and wp.isfinite(position[2])
        and moved > wp.float64(0.0)
        and moved <= wp.where(step < 16, wp.float64(0.5) * limit[0], limit[0])
    )


@wp.kernel
def restored(
    state: qm.MoveState,
    core_error: wp.array[int],
    mesh_error: wp.array[int],
    invalid: wp.array[int],
    boundary_error: wp.array[int],
):
    core_error[0] = 0
    if mesh_error[0] != 0:
        core_error[0] |= 1
    if boundary_error[0] != 0:
        core_error[0] |= 2
    if invalid[0] != 0:
        core_error[0] |= 4
    if core_error[0] != 0:
        state.error[0] |= 8
        state.source_count[0] = 0


class TopologyRecovery(qm.QualityMove):
    def __init__(self, owner, *, threshold=0.3):
        self.owner = weakref.proxy(owner)
        super().__init__(
            owner.core,
            threshold=threshold,
            max_attempts=32,
            extraction_tolerance=owner.extraction.tolerance,
            _proven_exterior_recovery=False,
            _approximate_queries=False,
        )
        with wp.ScopedDevice(self.device):
            self.chosen = wp.full(1, -1, dtype=int)
            self.invoked = wp.zeros(1, dtype=int)
            self.frame = wp.zeros(3, dtype=wp.vec3d)
            self.limit = wp.zeros(1, dtype=wp.float64)
            self.old_roles = wp.zeros(self.core.n, dtype=int)
            self.before_refs = wp.zeros(self.core.n, dtype=int)
            self.after_refs = wp.zeros(self.core.n, dtype=int)
            self.references_rejected = wp.zeros(1, dtype=int)
            wp.load_module(module=__name__, device=self.device)
        self._finish_setup()

    def reset(self):
        self._launch(reset_state, 1, [self._state, self.invoked, self.chosen])

    def _record(self):
        m = self.owner
        e = m.extraction
        self._launch(
            eligible,
            1,
            [
                m._state,
                m.core.error,
                e.error,
                e._exterior.pinched_site,
                e._exterior.rejected,
                m.core.boundary.roles,
                self.chosen,
                self.invoked,
            ],
        )
        wp.capture_if(self.invoked, super()._record)

    def _initial(self):
        c = self.core
        e = self.extraction
        c._evaluate(True)
        c.boundary.run()
        e.run()
        self._launch(
            initial_invalid_ready,
            1,
            [self._state, e.error, e.diagnostics, e.tetrahedra_count, c._invalid, c.boundary.error],
        )
        wp.capture_if(self.ready, self._operate)

    def _operate(self):
        self._metrics()
        wp.copy(self.previous_metrics, self.metrics)
        # The invalid baseline's Euler characteristic is not an acceptance target.
        wp.copy(self.previous_topology, self.owner.input_topology.counts)
        wp.copy(self._old_points, self.points)
        wp.copy(self._old_tets, self.full_tetrahedra)
        wp.copy(self._old_count, self.core.full_tetrahedra_count)
        wp.copy(self._old_active, self._state.active)
        wp.copy(self.old_roles, self.core.boundary.roles)
        self.before_refs.zero_()
        self._launch(
            mark_references,
            self._m,
            [self.extraction.tetrahedra, self.extraction.tetrahedra_count, self.before_refs],
        )
        self._prepare_proposal()
        wp.capture_while(self.running, self._try)
        wp.capture_if(self.accepted, self._commit)
        self._launch(qm.need_restore, 1, [self._state])
        wp.capture_if(self.ready, self._restore)

    def _prepare_proposal(self):
        self._launch(
            prepare,
            1,
            [
                self._state,
                self.core.domain.tree,
                self._old_points,
                self.chosen,
                self.old_roles,
                self.core._edge,
                self.frame,
                self.limit,
            ],
        )

    def _propose(self):
        self._launch(
            propose, 1, [self._state, self._old_points, self.chosen, self.frame, self.limit]
        )

    def _quality(self):
        self._complete_quality()
        self.after_refs.zero_()
        self.references_rejected.zero_()
        self._launch(
            mark_references,
            self._m,
            [self.extraction.tetrahedra, self.extraction.tetrahedra_count, self.after_refs],
        )
        self._launch(
            references_and_roles,
            self.core.n,
            [
                self.before_refs,
                self.after_refs,
                self.old_roles,
                self.core.boundary.roles,
                self._state.active,
                self.references_rejected,
            ],
        )
        self._launch(reject_changed_references, 1, [self._state, self.references_rejected])

    def _restore(self):
        c = self.core
        if c.selection is not None:
            c.selection._invalidate_current()
        wp.copy(self.points, self._old_points)
        wp.copy(self._state.active, self._old_active)
        # Re-establish private certificate/index state, not only public cell rows.
        c._reconstruct()
        c._evaluate(True)
        c.boundary.run()
        self._launch(
            restored, 1, [self._state, c.error, c.mesh.error, c._invalid, c.boundary.error]
        )
        self.extraction.run()
        self._metrics()
