"""Install private sample-parallel quadrature before capture; no new device arrays."""
import warp as wp
from warpower import _extraction
from surface_sample_parallel_v1 import surface_error_samples

ORIGINAL = _extraction.ExtractionWorkspace._surface_metrics


def parallel(self):
    self._launch(surface_error_samples, 3 * len(self.boundary_faces), [self.domain.tree, self.points, self.boundary_faces, self.boundary_count, self._surface_integral, self.surface_max_sample_distance, self.error], block_dim=32)


def install(enabled=True):
    if enabled:
        wp.load_module(module=surface_error_samples.module, device='cuda:0', block_dim=32)
    _extraction.ExtractionWorkspace._surface_metrics = parallel if enabled else ORIGINAL
