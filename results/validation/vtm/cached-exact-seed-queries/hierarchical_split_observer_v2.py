"""Diagnostic pending-mask counts; extra observer work is excluded from timing evidence."""

import warp as wp
from hierarchical_split_v2 import CachedHierarchicalSplitQueries

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.kernel
def observe(
    pending: wp.array[wp.uint32],
    unresolved: wp.array2d[int],
    count: wp.array[int],
    stats: wp.array[wp.int64],
):
    i = wp.tid()
    if i == 0:
        wp.atomic_add(stats, 0, wp.int64(1))
        wp.atomic_add(stats, 1, wp.int64(count[0]))
    if i >= count[0]:
        return
    mask = pending[i]
    if mask != wp.uint32(0):
        lanes = int(0)
        for k in range(32):
            if (mask & (wp.uint32(1) << wp.uint32(k))) != wp.uint32(0):
                lanes += 1
                if unresolved[i, k] >= 0:
                    wp.atomic_add(stats, 4, wp.int64(1))
                elif unresolved[i, k] == -2:
                    wp.atomic_add(stats, 5, wp.int64(1))
        wp.atomic_add(stats, 2, wp.int64(1))
        wp.atomic_add(stats, 3, wp.int64(lanes))


instances = []


class ObservedSplit(CachedHierarchicalSplitQueries):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stats = wp.zeros(6, dtype=wp.int64, device=self.device)
        wp.load_module(module=__name__, device=self.device, block_dim=self.block_dim)
        instances.append(self)

    def run(self):
        super().run()
        wp.launch(
            observe,
            self.capacity,
            inputs=[self.pending, self.unresolved, self._check_inputs[2], self.stats],
            device=self.device,
            block_dim=self.block_dim,
        )


def install():
    from warpower import _stackless_seeds

    _stackless_seeds.StacklessSeedQueries = ObservedSplit
