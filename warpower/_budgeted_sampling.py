"""Private initial sampler with a device-selected prefix for corner budgeting.

The fixed-count public sampler and its device kernels remain unchanged.
"""

import warp as wp

from . import _canonical
from ._sampling import InitialSamplingWorkspace, begin, random64, ready_samples
from ._sizing import SizingData, sizing_value
from ._surface_bvh import SurfaceTree, material_side

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.kernel
def sample_prefix(
    requested: wp.array[int],
    tree: SurfaceTree,
    field: SizingData,
    graded: bool,
    resolution: int,
    cdf: wp.array[wp.float64],
    minimum: wp.array[wp.float64],
    seed: wp.array[int],
    max_attempts: int,
    points: wp.array[wp.vec3],
    attempts: wp.array[int],
    error: wp.array[int],
):
    i = wp.tid()
    points[i] = wp.vec3(0.0)
    attempts[i] = 0
    if i >= requested[0]:
        return
    # Site i owns 2**32 successive states. At most 5*max_attempts (<2**23)
    # variates are consumed, so variable rejection counts cannot meet another
    # site's stream. The odd increment is invertible modulo 2**64.
    state = wp.uint64(wp.uint32(seed[0])) + (wp.uint64(i) << wp.uint64(32)) * wp.uint64(
        0x9E3779B97F4A7C15
    )
    lo, hi = wp.vec3d(tree.lower[1]), wp.vec3d(tree.upper[1])
    width = (hi - lo) / wp.float64(resolution)
    total = cdf[cdf.shape[0] - 1]
    accepted = bool(False)
    for attempt in range(max_attempts):
        next_state, u = random64(state)
        state = wp.uint64(next_state)
        target = u * total
        if target >= total:
            target = total * wp.float64(0.9999999999999998)
        left, right = int(0), int(cdf.shape[0] - 1)
        while left < right:
            middle = (left + right) // 2
            if cdf[middle] <= target:
                left = middle + 1
            else:
                right = middle
        cell = wp.vec3i(
            left % resolution, (left // resolution) % resolution, left // (resolution * resolution)
        )
        fraction = wp.vec3d(0.0)
        for k in range(3):
            next_state, u = random64(state)
            state = wp.uint64(next_state)
            fraction[k] = u
        q = wp.vec3(lo + wp.cw_mul(width, wp.vec3d(cell) + fraction))
        attempts[i] = attempt + 1
        if material_side(tree, wp.vec3d(q)) != 1:
            continue
        mu = wp.float64(1.0)
        if graded:
            mu = sizing_value(field, wp.vec3d(q))
        lower = minimum[left]
        if mu <= wp.float64(0.0) or not wp.isfinite(mu) or lower > mu * wp.float64(1.000000000002):
            wp.atomic_or(error, 0, 32)
            break
        ratio = wp.min(wp.float64(1.0), lower / mu)
        next_state, u = random64(state)
        state = wp.uint64(next_state)
        if u < ratio * ratio * ratio:
            points[i] = q
            accepted = True
            break
    if not accepted:
        wp.atomic_or(error, 0, 8)


@wp.kernel
def validate_prefix(
    requested: wp.array[int], capacity: int, error: wp.array[int], ready: wp.array[int]
):
    if requested[0] < 1 or requested[0] > capacity:
        wp.atomic_or(error, 0, 64)
    ready[0] = int(error[0] == 0)


@wp.kernel
def check_prefix_duplicates(owners: wp.array[int], requested: wp.array[int], error: wp.array[int]):
    i = wp.tid()
    if i < requested[0] and owners[i] != i:
        wp.atomic_or(error, 0, 16)


@wp.kernel
def publish_prefix(error: wp.array[int], count: wp.array[int], requested: wp.array[int]):
    count[0] = 0
    if error[0] == 0:
        count[0] = requested[0]


@wp.kernel(enable_backward=False)
def insert_prefix(
    requested: wp.array[int],
    points: wp.array[wp.vec3],
    weights: wp.array[float],
    bits: wp.array2d[wp.uint32],
    table: wp.array[int],
    buckets: wp.array[int],
):
    i = wp.tid()
    if i >= requested[0]:
        return
    h = wp.uint32(0)
    for axis in range(3):
        value = bits[i, axis]
        # Positive and negative zero describe the same position.
        if (value & wp.uint32(0x7FFFFFFF)) == wp.uint32(0):
            value = wp.uint32(0)
        h = _canonical.mix(h ^ value)
    bucket = int(h & wp.uint32(table.shape[0] - 1))
    p = points[i]
    while True:
        previous = wp.atomic_cas(table, bucket, -1, i)
        if previous == -1:
            break
        other = points[previous]
        if p[0] == other[0] and p[1] == other[1] and p[2] == other[2]:
            if weights[i] > weights[previous] or (weights[i] == weights[previous] and i < previous):
                if wp.atomic_cas(table, bucket, previous, i) == previous:
                    break
            else:
                break
        else:
            bucket = (bucket + 1) & (table.shape[0] - 1)
    buckets[i] = bucket


@wp.kernel(enable_backward=False)
def resolve_prefix(
    requested: wp.array[int], table: wp.array[int], buckets: wp.array[int], owners: wp.array[int]
):
    i = wp.tid()
    if i >= requested[0]:
        return
    owners[i] = table[buckets[i]]


class BudgetedSamplingWorkspace(InitialSamplingWorkspace):
    def __init__(self, domain, capacity, requested, **options):
        self.requested = requested
        wp.load_module(module=__name__, device=domain.device)
        super().__init__(domain, capacity, **options)

    def _record(self):
        self.attempts.zero_()
        self._launch(
            begin,
            1,
            [
                self.domain.error,
                self._field,
                self.field is not None,
                self.count,
                self.error,
                self._ready,
            ],
        )
        self._launch(validate_prefix, 1, [self.requested, self.n, self.error, self._ready])
        wp.capture_if(self._ready, self._sample)
        self._launch(publish_prefix, 1, [self.error, self.count, self.requested])

    def _draw(self):
        self._launch(
            sample_prefix,
            self.n,
            [
                self.requested,
                self.domain.tree,
                self._field,
                self.field is not None,
                self.resolution,
                self._cdf,
                self._minimum,
                self.seed,
                self.max_attempts,
                self.points,
                self.attempts,
                self.error,
            ],
        )
        self._launch(ready_samples, 1, [self.error, self._ready])
        wp.capture_if(self._ready, self._unique)

    def _unique(self):
        self._table.fill_(-1)
        self._launch(
            insert_prefix,
            self.n,
            [self.requested, self.points, self._weights, self._bits, self._table, self._buckets],
        )
        self._launch(
            resolve_prefix, self.n, [self.requested, self._table, self._buckets, self._owners]
        )
        self._launch(check_prefix_duplicates, self.n, [self._owners, self.requested, self.error])
