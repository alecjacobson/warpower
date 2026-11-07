"""Private VTM wall fallback sharing the existing exact-completion workspace."""
import warp as wp
from warpower import _active_delaunay as active
from warpower import _reuse_delaunay as reuse
from warpower import _validation
from warpower._delaunay_recovery import DelaunayWorkspace, reset
from local_dewall_v4 import normalize, initialize, walk

wp.set_module_options({'enable_backward': False})
ORIGINAL_REUSE = reuse.ReuseDelaunayWorkspace


@wp.kernel
def candidate_ready(count: wp.array[int], capacity: int, error: wp.array[int], ready: wp.array[int], candidates: wp.array[int]):
    candidates[0] = count[0]
    ready[0] = int(error[0] == 0 and count[0] > 0 and count[0] <= capacity)


@wp.kernel
def save_diagnostics(seeds: wp.array[int], rounds: wp.array[int], proofs: wp.array[int], faces: wp.array[int], saved: wp.array[int]):
    saved[0] = seeds[0]
    saved[1] = rounds[0]
    saved[2] = proofs[0]
    saved[3] = proofs[1]
    saved[4] = faces[0]
    saved[5] = faces[1]


@wp.kernel
def restore_diagnostics(seeds: wp.array[int], rounds: wp.array[int], proofs: wp.array[int], faces: wp.array[int], saved: wp.array[int], wall: wp.array[int]):
    wall[0] = seeds[0]
    wall[1] = rounds[0]
    wall[2] = proofs[0]
    wall[3] = proofs[1]
    wall[4] = faces[0]
    wall[5] = faces[1]
    seeds[0] = saved[0]
    rounds[0] = saved[1]
    proofs[0] = saved[2]
    proofs[1] = saved[3]
    faces[0] = saved[4]
    faces[1] = saved[5]


class BorrowedCandidate:
    def __init__(self, points, tree, tets, count):
        self.points, self.tree = points, tree
        n = len(points)
        self.raw_tets = tets[:min(len(tets), max(16, 8 * n))]
        self.raw_count = count
        with wp.ScopedDevice(points.device):
            self.normalized = wp.empty(n + 6, dtype=wp.vec3)
            self.seeds = wp.empty(2 * n, dtype=wp.vec3i)
            self.seed_status = wp.empty(2 * n, dtype=int)
            self.status = wp.empty(2 * n, dtype=int)
            self.seed_costs = wp.empty((2 * n, 2), dtype=int)
            self.costs = wp.empty((2 * n, 2), dtype=int)
            self.error = wp.zeros(1, dtype=int)
            wp.load_module(module='local_dewall_v4', device=points.device)

    def run(self):
        wp.launch(normalize, len(self.normalized), inputs=[self.points, self.tree, self.normalized], device=self.points.device)
        self.raw_count.zero_()
        self.error.zero_()
        wp.launch(initialize, 2 * len(self.points), inputs=[self.normalized, self.tree, self.seeds, self.seed_status, self.seed_costs], device=self.points.device, block_dim=32)
        wp.launch(walk, 2 * len(self.points), inputs=[self.normalized, self.tree, self.seeds, self.seed_status, self.status, self.costs, self.raw_tets, self.raw_count], device=self.points.device, block_dim=32)


class SharedWallWorkspace(ORIGINAL_REUSE):
    def __init__(self, *args, **kwargs):
        self._wall = None
        super().__init__(*args, **kwargs)
        with wp.ScopedDevice(self.device):
            self._wall = BorrowedCandidate(self.points, self._original.tree, self._previous, self._previous_count)
            self.wall_attempted = wp.zeros(1, dtype=int)
            self.wall_accepted = wp.zeros(1, dtype=int)
            self.wall_error = wp.zeros(1, dtype=int)
            self.wall_candidates = wp.zeros(1, dtype=int)
            self._wall_ready = wp.zeros(1, dtype=int)
            self._saved_diagnostics = wp.zeros(6, dtype=int)
            self.wall_diagnostics = wp.zeros(6, dtype=int)
            wp.load_module(module=__name__, device=self.device)
            with wp.ScopedCapture(device=self.device) as capture:
                self._record_reuse()
            self._reuse_graph = capture.graph
        wp.synchronize_device(self.device)

    def _record_reuse(self):
        if self._wall is None:
            return super()._record_reuse()
        self._launch(reset, 1, [self._state, self.cache_strategy])
        self.reuse_seeds.zero_()
        self.reuse_rounds.zero_()
        self.wall_attempted.zero_()
        self.wall_accepted.zero_()
        self.wall_error.zero_()
        self.wall_candidates.zero_()
        self.wall_diagnostics.zero_()
        self._launch(reuse.begin, 1, [self._previous_count, self._reuse_ready, self.reuse_attempted, self.reuse_accepted, self.reuse_error])
        wp.capture_if(self._reuse_ready, self._validate_reuse)
        self._launch(reuse.fallback, 1, [self.error, self._wall_ready])
        wp.capture_if(self._wall_ready, self._validate_wall)
        self._launch(reuse.fallback, 1, [self.error, self._reuse_ready])
        wp.capture_if(self._reuse_ready, lambda: DelaunayWorkspace.run(self))
        wp.copy(self._previous, self.tetrahedra)
        self._launch(reuse.remember, 1, [self.error, self.tetrahedra_count, self._previous_count])

    def _validate_wall(self):
        self._base._validation.zero_()
        self._launch(_validation.validate, self.n + 1, [self.points, self._base.weights, self._base._empty, self._base._empty, False, self._base._validation])
        self._launch(reuse.input_ready, 1, [self._base._validation, self._wall_ready, self.wall_error])
        wp.capture_if(self._wall_ready, self._attempt_wall)

    def _attempt_wall(self):
        self.wall_attempted.fill_(1)
        self._refresh_original()
        c = self._reuse_completion
        fields = [self.reuse_seeds, self.reuse_rounds, c._certificate.proof_statistics, c._certificate.face_statistics]
        wp.launch(save_diagnostics, 1, inputs=[*fields, self._saved_diagnostics], device=self.device)
        self._wall.run()
        c.error.fill_(1)
        wp.launch(candidate_ready, 1, inputs=[self._wall.raw_count, len(self._wall.raw_tets), self._wall.error, self._wall_ready, self.wall_candidates], device=self.device)
        wp.capture_if(self._wall_ready, self._complete_wall)
        self._launch(reuse.completed, 1, [self._reuse_filter.error, c.error, c.tetrahedra_count, len(self.tetrahedra), self.wall_error, self.wall_accepted])
        wp.capture_if(self.wall_accepted, self._publish_reuse)
        wp.launch(restore_diagnostics, 1, inputs=[*fields, self._saved_diagnostics, self.wall_diagnostics], device=self.device)

    def _complete_wall(self):
        # Only graph recording reads this Python setting. Retained-fragment
        # completion keeps its original budget; fresh wall repair gets eight.
        c = self._reuse_completion
        previous = c.max_rounds
        try:
            c.max_rounds = 8
            c.run()
        finally:
            c.max_rounds = previous


def install():
    reuse.ReuseDelaunayWorkspace = SharedWallWorkspace
    active.ReuseDelaunayWorkspace = SharedWallWorkspace
