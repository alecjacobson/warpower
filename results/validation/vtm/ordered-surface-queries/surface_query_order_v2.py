"""Use the v1 unchanged scalar kernels, recording sort directly into a child capture.

The v1 prebuilt sort graph attempted lazy instantiation inside an enclosing
capture. Follow SiteHierarchy's eager initialization/child-capture pattern;
retain every v1 kernel expression and keep sorting cost in the measured path.
"""
import warp as wp
from warp.utils import radix_sort_pairs

from warpower import _surface_query_cache as original
from surface_query_order_v1 import centroid_keys, ORDERED


class OrderedSurfaceQueryCache(original.SurfaceQueryCache):
    def __init__(self, domain, capacity):
        super().__init__(domain, capacity)
        with wp.ScopedDevice(self.device):
            self.order_keys = wp.zeros(2 * capacity, dtype=wp.uint64)
            self.order_ids = wp.zeros(2 * capacity, dtype=int)
            self.order_enabled = wp.ones(1, dtype=int)
            for block in [32, 256]:
                wp.load_module(module="surface_query_order_v1", device=self.device, block_dim=block)
            self._sort()
        wp.synchronize_device(self.device)

    def _sort(self):
        radix_sort_pairs(self.order_keys, self.order_ids, self.capacity, end_bit=64)

    def launch(self, kernel, dim, args, block_dim=32):
        assert kernel in ORDERED and dim == self.capacity
        wp.launch(centroid_keys, self.capacity, inputs=[args[0], args[1], args[2], args[3], args[5], self.order_keys, self.order_ids], device=self.device, block_dim=256)
        if wp.get_stream(self.device).is_capturing:
            wp.capture_if(self.order_enabled, self._sort)
        else:
            self._sort()
        wp.launch(ORDERED[kernel], dim, inputs=[*args, self.current, self.previous, self.table, self.order_ids], device=self.device, block_dim=block_dim)
        self.after(args[5])


def install():
    original.SurfaceQueryCache = OrderedSurfaceQueryCache
