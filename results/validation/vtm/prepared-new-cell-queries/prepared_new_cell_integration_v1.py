"""Private #233 binding: borrow consumed seed geometry, with no extra device arrays."""

import prepared_new_cell_v1 as primitive
import warp as wp

from warpower._proved_completion import ProvedCertificate, SeededFacetCompletion


class SharedQueries:
    def __init__(self, certificate, seed_queries):
        c = certificate
        self.capacity = len(c.tetrahedra)
        self.device = c.device
        self.shapes = seed_queries.shapes
        self.boxes = seed_queries.boxes
        self.segment = seed_queries.segment
        if len(self.shapes) < self.capacity or len(self.boxes) < self.capacity:
            raise ValueError("Seed geometry must cover every completion cell slot")
        if self.segment.n != len(c.points) or self.segment.ids.ptr != c.tree.ids.ptr:
            raise ValueError("Completion must use the current shared site segment")
        self._prepare = [
            c.points,
            c.tetrahedra,
            c.tetrahedra_count,
            c._proved,
            self.shapes,
            self.boxes,
        ]
        self._prefix = [c.points, c.tetrahedra, c.tetrahedra_count]
        self._suffix = [
            c.error,
            c.statistics,
            c._proved,
            c.proof_statistics,
            self.shapes,
            self.boxes,
            self.segment,
        ]
        wp.load_module(module=primitive, device=self.device, block_dim=32)

    def run(self, tree):
        if tree.ids.ptr != self.segment.ids.ptr or tree.n != self.segment.n:
            raise ValueError("Completion query tree changed its segment association")
        wp.launch(
            primitive.prepare_unproved,
            self.capacity,
            inputs=self._prepare,
            device=self.device,
            block_dim=32,
        )
        wp.launch(
            primitive.check_prepared_unproved,
            32 * self.capacity,
            inputs=[*self._prefix, tree, *self._suffix],
            device=self.device,
            block_dim=32,
        )


class PreparedProvedCertificate(ProvedCertificate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._shared_unproved = None

    def _global_spheres(self, tree):
        if self._shared_unproved is None or not self.point_parallel:
            return super()._global_spheres(tree)
        self._shared_unproved.run(tree)


def install():
    original = SeededFacetCompletion.__init__

    def setup(self, seed_filter, **kwargs):
        original(self, seed_filter, **kwargs)
        seed = seed_filter._prepared
        if (
            seed is not None
            and hasattr(seed, "segment")
            and 4096 < len(seed_filter.points) <= 16384
            and len(self._certificate.tetrahedra) <= len(seed.shapes)
            and len(self._certificate.tetrahedra) <= len(seed.boxes)
        ):
            shared = SharedQueries(self._certificate, seed)
            assert shared.shapes is seed.shapes and shared.boxes is seed.boxes
            self._certificate._shared_unproved = shared

    SeededFacetCompletion._certificate_type = PreparedProvedCertificate
    SeededFacetCompletion.__init__ = setup
