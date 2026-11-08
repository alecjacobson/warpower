"""Transactional removal only of cells proved entirely exterior."""

import weakref

import warp as wp

from ._exterior_band import accept, references, reject_orphans
from ._exterior_proof_bvh import exterior
from ._surface_bvh import SurfaceTree

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.kernel
def begin(
    error: wp.array[int],
    diagnostics: wp.array[int],
    enabled: wp.array[int],
    ready: wp.array[int],
    removed: wp.array[int],
    rejected: wp.array[int],
):
    ready[0] = int(
        error[0] == 0 and enabled[0] != 0 and (diagnostics[0] == 2 or diagnostics[0] == 4)
    )
    removed[0] = 0
    rejected[0] = 0


@wp.kernel
def remove_proved(
    tree: SurfaceTree,
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    count: wp.array[int],
    selected: wp.array[int],
    distance: wp.array[wp.float64],
    removed: wp.array[int],
):
    i = wp.tid()
    if i < count[0] and selected[i] != 0 and distance[i] > wp.float64(0.0):
        ids = wp.vec4i(tets[i, 0], tets[i, 1], tets[i, 2], tets[i, 3])
        if exterior(tree, points, ids) != 0:
            selected[i] = 0
            wp.atomic_add(removed, 0, 1)


class ProvenExteriorRecovery:
    def __init__(self, extraction, enabled=None):
        self.e = weakref.proxy(extraction)
        self.device = extraction.device
        with wp.ScopedDevice(self.device):
            self.before = wp.empty_like(extraction.selected)
            self.volume = wp.empty_like(extraction.volume)
            self.references = wp.empty(len(extraction.points), dtype=int)
            self.ready = wp.zeros(1, dtype=int)
            self.removed = wp.zeros(1, dtype=int)
            self.rejected = wp.zeros(1, dtype=int)
        self.enabled = wp.ones(1, dtype=int, device=self.device) if enabled is None else enabled
        wp.load_module(module=__name__, device=self.device)

    def launch(self, kernel, dim, args):
        wp.launch(kernel, dim, inputs=args, device=self.device)

    def run(self):
        e = self.e
        self.launch(
            begin,
            1,
            [e.error, e.diagnostics, self.enabled, self.ready, self.removed, self.rejected],
        )
        wp.capture_if(self.ready, self.attempt)

    def attempt(self):
        e = self.e
        wp.copy(self.before, e.selected)
        wp.copy(self.volume, e.volume)
        self.references.zero_()
        self.launch(
            remove_proved,
            e.capacity,
            [
                e.domain.tree,
                e.points,
                e.input_tetrahedra,
                e.input_count,
                e.selected,
                e.signed_distance,
                self.removed,
            ],
        )
        self.launch(
            references,
            e.capacity,
            [e.input_tetrahedra, e.input_count, self.before, e.selected, self.references],
        )
        self.launch(reject_orphans, len(e.points), [self.references, self.rejected])
        wp.capture_if(self.removed, e._recompact)
        self.launch(accept, 1, [e.error, e.diagnostics, self.volume, e.volume, self.rejected])
        wp.capture_if(self.rejected, self.restore)

    def restore(self):
        e = self.e
        wp.copy(e.selected, self.before)
        e._recompact()
        self.removed.zero_()
