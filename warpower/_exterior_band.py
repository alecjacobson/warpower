"""Bounded removal of exterior-center tolerance cells at nonmanifold edges.

This is a boundary approximation, not a proof that removed cells are exterior.
The transaction preserves interior-center cells, referenced vertices and cells
incident to prescribed feature sites. Failure restores the original selection.
"""

import weakref

import warp as wp

from ._containment import bad_edge
from ._mesh_topology import previous_edge

wp.set_module_options({"enable_backward": False, "fast_math": False})


@wp.kernel
def begin(
    error: wp.array[int],
    diagnostics: wp.array[int],
    enabled: wp.array[int],
    ready: wp.array[int],
    rounds: wp.array[int],
    removed: wp.array[int],
    rejected: wp.array[int],
):
    ready[0] = int(enabled[0] != 0 and error[0] == 0 and diagnostics[0] == 2)
    rounds[0] = 0
    removed[0] = 0
    rejected[0] = 0


@wp.kernel
def prune(
    tets: wp.array2d[int],
    count: wp.array[int],
    keep: wp.array[int],
    distance: wp.array[wp.float64],
    roles: wp.array[int],
    indices: wp.array[int],
    table: wp.array[int],
    occurrences: wp.array[int],
    balance: wp.array[int],
    changes: wp.array[int],
    total: wp.array[int],
):
    t = wp.tid()
    if t >= count[0] or keep[t] == 0 or distance[t] <= wp.float64(0.0):
        return
    ids = wp.vec4i(tets[t, 0], tets[t, 1], tets[t, 2], tets[t, 3])
    for k in range(4):
        if roles[ids[k]] >= 2:
            return
    touches = bool(False)
    for i in range(4):
        for j in range(i + 1, 4):
            if bad_edge(ids[i], ids[j], indices, table, occurrences, balance):
                touches = True
    if touches:
        keep[t] = 0
        wp.atomic_add(changes, 0, 1)
        wp.atomic_add(total, 0, 1)


@wp.kernel
def references(
    tets: wp.array2d[int],
    count: wp.array[int],
    before: wp.array[int],
    after: wp.array[int],
    flags: wp.array[int],
):
    t = wp.tid()
    if t >= count[0] or before[t] == 0:
        return
    bits = 1 + 2 * after[t]
    for k in range(4):
        wp.atomic_or(flags, tets[t, k], bits)


@wp.kernel
def reject_orphans(flags: wp.array[int], rejected: wp.array[int]):
    if flags[wp.tid()] == 1:
        wp.atomic_or(rejected, 0, 1)


@wp.kernel
def next_round(
    error: wp.array[int],
    diagnostics: wp.array[int],
    changes: wp.array[int],
    rejected: wp.array[int],
    rounds: wp.array[int],
    ready: wp.array[int],
):
    rounds[0] += 1
    ready[0] = int(
        error[0] == 0
        and diagnostics[0] == 2
        and changes[0] > 0
        and rejected[0] == 0
        and rounds[0] < 4
    )


@wp.kernel
def accept(
    error: wp.array[int],
    diagnostics: wp.array[int],
    before: wp.array[wp.float64],
    after: wp.array[wp.float64],
    rejected: wp.array[int],
):
    # Bound the change even when a connected exterior band is very large.
    if (
        error[0] != 0
        or diagnostics[0] != 0
        or before[0] <= wp.float64(0.0)
        or after[0] < wp.float64(0.99) * before[0]
    ):
        rejected[0] = 1


@wp.kernel
def remember_pinched(
    first: wp.array[int],
    degree: wp.array[int],
    twins: wp.array[int],
    error: wp.array[int],
    diagnostics: wp.array[int],
    roles: wp.array[int],
    site: wp.array[int],
):
    i = wp.tid()
    if error[0] != 0 or diagnostics[0] != 4 or roles[i] != 1 or degree[i] <= 0:
        return
    e = first[i]
    visited = int(0)
    if e < 0 or e >= twins.shape[0]:
        return
    for _ in range(degree[i]):
        visited += 1
        e = twins[previous_edge(e)]
        if e < 0 or e >= twins.shape[0]:
            return
        if e == first[i]:
            break
    if visited != degree[i] or e != first[i]:
        wp.atomic_min(site, 0, i)


class ExteriorBandRecovery:
    def __init__(self, extraction, roles, enabled, *, pinched_site=None):
        self.extraction = weakref.proxy(extraction)
        self.device = extraction.device
        self.pinched_site = pinched_site
        self.roles = roles if roles is not None else wp.zeros(len(extraction.points), dtype=int)
        self.enabled = enabled if enabled is not None else wp.ones(1, dtype=int)
        self.before = wp.empty_like(extraction.selected)
        self.before_volume = wp.empty_like(extraction.volume)
        self.references = wp.empty(len(extraction.points), dtype=int)
        self.ready = wp.zeros(1, dtype=int)
        self.rounds = wp.zeros(1, dtype=int)
        self.removed = wp.zeros(1, dtype=int)
        self.rejected = wp.zeros(1, dtype=int)
        self.changes = wp.zeros(1, dtype=int)
        wp.load_module(module=__name__, device=self.device)

    def _launch(self, kernel, dim, args):
        wp.launch(kernel, dim, inputs=args, device=self.device)

    def run(self):
        e = self.extraction
        self._launch(
            begin,
            1,
            [
                e.error,
                e.diagnostics,
                self.enabled,
                self.ready,
                self.rounds,
                self.removed,
                self.rejected,
            ],
        )
        wp.capture_if(self.ready, self._attempt)

    def _attempt(self):
        if self.pinched_site is not None:
            self.pinched_site.fill_(2**31 - 1)
        e = self.extraction
        wp.copy(self.before, e.selected)
        wp.copy(self.before_volume, e.volume)
        wp.capture_while(self.ready, self._round)
        self._launch(
            accept, 1, [e.error, e.diagnostics, self.before_volume, e.volume, self.rejected]
        )
        wp.capture_if(self.rejected, self._restore)

    def _round(self):
        e, t = self.extraction, self.extraction._topology
        self.changes.zero_()
        self.references.zero_()
        self._launch(
            prune,
            e.capacity,
            [
                e.input_tetrahedra,
                e.input_count,
                e.selected,
                e.signed_distance,
                self.roles,
                t.boundary_indices,
                t._edges,
                t._edge_counts,
                t._edge_balance,
                self.changes,
                self.removed,
            ],
        )
        self._launch(
            references,
            e.capacity,
            [e.input_tetrahedra, e.input_count, self.before, e.selected, self.references],
        )
        self._launch(reject_orphans, len(e.points), [self.references, self.rejected])
        wp.capture_if(self.changes, e._recompact)
        self._launch(
            next_round,
            1,
            [e.error, e.diagnostics, self.changes, self.rejected, self.rounds, self.ready],
        )

    def _restore(self):
        e = self.extraction
        if self.pinched_site is not None:
            t = e._topology
            self._launch(
                remember_pinched,
                len(e.points),
                [
                    t._first,
                    t._degree,
                    t._twins,
                    e.error,
                    e.diagnostics,
                    self.roles,
                    self.pinched_site,
                ],
            )
        wp.copy(e.selected, self.before)
        e._recompact()
        self.removed.zero_()
