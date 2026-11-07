"""Private unweighted wall candidates, exact completion, ordinary fallback."""
import warp as wp
from warpower._graph import DiagramWorkspace
from warpower._facet_completion import FacetCompletion
from warpower._proved_completion import ProvedCertificate, SeededFacetCompletion
from local_dewall_v4 import Candidate, normalize, initialize, walk
from warpower._facet_seeds import CertifiedSeeds

wp.set_module_options({'enable_backward': False})


@wp.kernel
def ready(error: wp.array[int], count: wp.array[int], capacity: int, flag: wp.array[int]):
    flag[0] = int(error[0] == 0 and count[0] > 0 and count[0] <= capacity)


@wp.kernel
def choose(candidate: wp.array[int], seed_error: wp.array[int], error: wp.array[int], accept: wp.array[int], fallback: wp.array[int]):
    accept[0] = int(candidate[0] != 0 and seed_error[0] == 0 and error[0] == 0)
    fallback[0] = 1 - accept[0]


class RawCandidate(Candidate):
    def run(self):
        wp.launch(normalize, len(self.normalized), inputs=[self.points, self.tree, self.normalized], device=self.points.device)
        self.raw_count.zero_()
        self.error.zero_()
        wp.launch(initialize, 2 * len(self.points), inputs=[self.normalized, self.tree, self.seeds, self.seed_status, self.seed_costs], device=self.points.device, block_dim=32)
        wp.launch(walk, 2 * len(self.points), inputs=[self.normalized, self.tree, self.seeds, self.seed_status, self.status, self.costs, self.raw_tets, self.raw_count], device=self.points.device, block_dim=32)


class WallWorkspace(DiagramWorkspace):
    def _init_tetrahedra(self, capacity):
        super()._init_tetrahedra(capacity)
        self._wall = None
        if not self.weighted and not self._adjacency:
            self._wall = RawCandidate(self.points, self._hierarchy, capacity=capacity)
            self._wall_filter = CertifiedSeeds(self.points, self._wall.raw_tets, self._wall.raw_count, self._hierarchy, block_dim=32, prepare_geometry=True)
            self._wall_completion = SeededFacetCompletion(self._wall_filter, capacity=capacity, max_rounds=8, query_block_dim=32)
            self._wall_completion._certificate.point_parallel = True
            self.wall_ready = wp.zeros(1, dtype=int)
            self.wall_accepted = wp.zeros(1, dtype=int)
            self.wall_fallback = wp.zeros(1, dtype=int)

    def _record_tetrahedra(self):
        if self._wall is None:
            return super()._record_tetrahedra()
        self._wall.run()
        self._wall_completion.error.fill_(1)
        wp.launch(ready, 1, inputs=[self._wall.error, self._wall.raw_count, len(self._wall.raw_tets), self.wall_ready], device=self.device)
        wp.capture_if(self.wall_ready, self._wall_completion.run)
        wp.launch(choose, 1, inputs=[self.wall_ready, self._wall_filter.error, self._wall_completion.error, self.wall_accepted, self.wall_fallback], device=self.device)
        wp.capture_if(self.wall_accepted, self._publish_wall)
        wp.capture_if(self.wall_fallback, super()._record_tetrahedra)

    def _publish_wall(self):
        wp.copy(self.tetrahedra, self._wall_completion.tetrahedra)
        wp.copy(self.tetrahedra_count, self._wall_completion.tetrahedra_count)
        self.tetrahedra_expansions.zero_()


def install():
    from warpower import _delaunay_recovery
    _delaunay_recovery.DiagramWorkspace = WallWorkspace
