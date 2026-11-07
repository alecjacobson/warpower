"""Private wall integration: current proof publication, bounded raw storage.

The archived v4 candidate kernels remain unchanged. Publication consumes the
current completion's arrays directly, including its matching face index; it
never treats an earlier successful replay as a current proof.
"""
import warp as wp
from warpower import _delaunay_recovery as dr
from warpower._graph import DiagramWorkspace
from warpower._facet_seeds import CertifiedSeeds
from warpower._proved_completion import SeededFacetCompletion
from local_dewall_v4 import normalize, initialize, walk

wp.set_module_options({'enable_backward': False})
ORIGINAL_BASE_ATTEMPT = dr.DelaunayWorkspace._base_attempt


@wp.kernel
def ready(error: wp.array[int], count: wp.array[int], capacity: int, flag: wp.array[int]):
    flag[0] = int(error[0] == 0 and count[0] > 0 and count[0] <= capacity)


@wp.kernel
def choose(candidate: wp.array[int], seed_error: wp.array[int], error: wp.array[int], accept: wp.array[int], fallback: wp.array[int]):
    accept[0] = int(candidate[0] != 0 and seed_error[0] == 0 and error[0] == 0)
    fallback[0] = 1 - accept[0]


@wp.kernel
def publication_ready(base_error: wp.array[int], accepted: wp.array[int], seed_error: wp.array[int], completion_error: wp.array[int], certificate_error: wp.array[int], count: wp.array[int], capacity: int, publish: wp.array[int], certify: wp.array[int]):
    valid = base_error[0] == 0
    publish[0] = int(valid and accepted[0] != 0 and seed_error[0] == 0 and completion_error[0] == 0 and certificate_error[0] == 0 and count[0] > 0 and count[0] <= capacity)
    certify[0] = int(valid and publish[0] == 0)


class RawCandidate:
    def __init__(self, points, tree, capacity):
        self.points, self.tree = points, tree
        n = len(points)
        capacity = min(capacity, max(16, 8 * n))
        with wp.ScopedDevice(points.device):
            self.normalized = wp.empty(n + 6, dtype=wp.vec3)
            self.seeds = wp.empty(2 * n, dtype=wp.vec3i)
            self.seed_status = wp.empty(2 * n, dtype=int)
            self.status = wp.empty(2 * n, dtype=int)
            self.seed_costs = wp.empty((2 * n, 2), dtype=int)
            self.costs = wp.empty((2 * n, 2), dtype=int)
            self.raw_tets = wp.empty((capacity, 4), dtype=int)
            self.raw_count = wp.zeros(1, dtype=int)
            self.error = wp.zeros(1, dtype=int)
            wp.load_module(module='local_dewall_v4', device=points.device)

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
            self._wall = RawCandidate(self.points, self._hierarchy, capacity)
            self._wall_filter = CertifiedSeeds(self.points, self._wall.raw_tets, self._wall.raw_count, self._hierarchy, block_dim=32, prepare_geometry=True)
            self._wall_completion = SeededFacetCompletion(self._wall_filter, capacity=capacity, max_rounds=8, query_block_dim=32)
            self._wall_completion._certificate.point_parallel = True
            self.wall_ready = wp.zeros(1, dtype=int)
            self.wall_accepted = wp.zeros(1, dtype=int)
            self.wall_fallback = wp.zeros(1, dtype=int)
            self.wall_publish = wp.zeros(1, dtype=int)
            self.wall_certify = wp.zeros(1, dtype=int)

    def _record(self):
        if self._wall is not None:
            self.wall_ready.zero_()
            self.wall_accepted.zero_()
            self.wall_fallback.zero_()
            self.wall_publish.zero_()
            self.wall_certify.zero_()
        super()._record()

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


def publish_certificate(workspace):
    source = workspace._base._wall_completion
    destination = workspace._certificate
    wp.copy(workspace.tetrahedra, source.tetrahedra)
    wp.copy(workspace.tetrahedra_count, source.tetrahedra_count)
    wp.copy(workspace._candidate, source.tetrahedra)
    wp.copy(workspace._candidate_count, source.tetrahedra_count)
    workspace.certificate_error.zero_()
    for name in ('statistics', '_used', '_table', '_occurrences', '_balance', '_parents', '_boundary', '_boundary_count'):
        wp.copy(getattr(destination, name), getattr(source._certificate, name))
    if destination._anchors is not None:
        destination._anchors.zero_()
        destination._local.zero_()
        destination._global.fill_(1)


def base_attempt(workspace):
    base = workspace._base
    if not isinstance(base, WallWorkspace) or base._wall is None:
        return ORIGINAL_BASE_ATTEMPT(workspace)
    base.run()
    workspace._launch(dr.prepare, 1, [workspace.base_error, workspace.certificate_error, workspace._ready])
    wp.copy(workspace._candidate, base.tetrahedra)
    wp.copy(workspace._candidate_count, base.tetrahedra_count)
    completion = base._wall_completion
    wp.launch(publication_ready, 1, inputs=[base.error, base.wall_accepted, base._wall_filter.error, completion.error, completion.certificate_error, completion.tetrahedra_count, len(workspace.tetrahedra), base.wall_publish, base.wall_certify], device=workspace.device)
    wp.capture_if(base.wall_publish, lambda: publish_certificate(workspace))
    wp.capture_if(base.wall_certify, workspace._certificate.run)
    workspace._launch(dr.begin_retry, 1, [workspace.base_error, workspace.certificate_error, workspace.error, workspace._running])


def install():
    dr.DiagramWorkspace = WallWorkspace
    dr.DelaunayWorkspace._base_attempt = base_attempt
