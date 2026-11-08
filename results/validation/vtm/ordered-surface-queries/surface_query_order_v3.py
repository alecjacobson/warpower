"""Unqualified size-gated query-order probe; preserve original small-cache paths.

Large captured caches choose on device from the current count and error. Eager
private calls retain v2 ordering; full public mesher queries use recorded graphs.
Cache-hit sensitivity still requires complete-solve measurement before adoption.
"""
import warp as wp

from warpower import _surface_query_cache as original
from surface_query_order_v1 import centroid_keys, ORDERED
from surface_query_order_v2 import OrderedSurfaceQueryCache

BaseCache = original.SurfaceQueryCache
wp.set_module_options({"enable_backward": False, "fast_math": False})


@wp.kernel
def choose_order(count: wp.array[int], error: wp.array[int], minimum: int, enabled: wp.array[int]):
    enabled[0] = int(error[0] == 0 and count[0] >= minimum)


class SizeGatedSurfaceQueryCache(OrderedSurfaceQueryCache):
    def __init__(self, domain, capacity):
        self.minimum_query_count = 16384
        self._has_order_storage = capacity >= self.minimum_query_count
        if self._has_order_storage:
            super().__init__(domain, capacity)
            wp.load_module(module=__name__, device=self.device, block_dim=32)
        else:
            BaseCache.__init__(self, domain, capacity)
            self.order_keys = self.order_ids = self.order_enabled = None

    def _ordered(self, kernel, dim, args, block_dim):
        wp.launch(centroid_keys, self.capacity, inputs=[args[0], args[1], args[2], args[3], args[5], self.order_keys, self.order_ids], device=self.device, block_dim=256)
        # Called in a captured conditional body or eagerly outside capture.
        self._sort()
        wp.launch(ORDERED[kernel], dim, inputs=[*args, self.current, self.previous, self.table, self.order_ids], device=self.device, block_dim=block_dim)
        self.after(args[5])

    def launch(self, kernel, dim, args, block_dim=32):
        if not self._has_order_storage:
            return BaseCache.launch(self, kernel, dim, args, block_dim)
        assert kernel in ORDERED and dim == self.capacity
        if wp.get_stream(self.device).is_capturing:
            wp.launch(choose_order, 1, inputs=[args[3], args[5], self.minimum_query_count, self.order_enabled], device=self.device, block_dim=32)
            wp.capture_if(self.order_enabled, lambda: self._ordered(kernel, dim, args, block_dim), lambda: BaseCache.launch(self, kernel, dim, args, block_dim))
        else:
            self._ordered(kernel, dim, args, block_dim)


def install():
    original.SurfaceQueryCache = SizeGatedSurfaceQueryCache
