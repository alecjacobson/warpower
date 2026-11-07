"""Device sampling of material with optional relative sizing density."""

import operator

import warp as wp

from . import _canonical
from ._sizing import SizingData, sizing_value
from ._surface_bvh import SurfaceTree, material_side

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.func(inline=True)
def random64(state: wp.uint64):
    # SplitMix64's bijective mixer; Vigna's public-domain reference:
    # https://prng.di.unimi.it/splitmix.c
    state += wp.uint64(0x9E3779B97F4A7C15)
    word = (state ^ (state >> wp.uint64(30))) * wp.uint64(0xBF58476D1CE4E5B9)
    word = (word ^ (word >> wp.uint64(27))) * wp.uint64(0x94D049BB133111EB)
    word ^= word >> wp.uint64(31)
    value = wp.float64(word >> wp.uint64(11)) * wp.float64(1.1102230246251565e-16)
    return state, value


@wp.func(inline=True)
def boundary_box(tree: SurfaceTree, lower: wp.vec3d, upper: wp.vec3d):
    node = int(1)
    while node > 0:
        lo, hi = wp.vec3d(tree.lower[node]), wp.vec3d(tree.upper[node])
        overlap = True
        for k in range(3):
            overlap = overlap and lo[k] <= upper[k] and hi[k] >= lower[k]
        if not overlap:
            node = tree.escape[node]
            continue
        if node >= tree.leaf_base:
            for j in range(
                4 * (node - tree.leaf_base), wp.min(4 * (node - tree.leaf_base) + 4, tree.n)
            ):
                face = tree.ids[j]
                a = wp.vec3d(tree.points[tree.indices[3 * face]])
                b = wp.vec3d(tree.points[tree.indices[3 * face + 1]])
                c = wp.vec3d(tree.points[tree.indices[3 * face + 2]])
                lo, hi = wp.min(a, wp.min(b, c)), wp.max(a, wp.max(b, c))
                overlap = True
                for k in range(3):
                    overlap = overlap and lo[k] <= upper[k] and hi[k] >= lower[k]
                if overlap:
                    return True
            node = tree.escape[node]
        else:
            node = tree.right[node]
    return False


@wp.kernel
def begin(
    domain_error: wp.array[int],
    field: SizingData,
    graded: bool,
    count: wp.array[int],
    error: wp.array[int],
    ready: wp.array[int],
):
    count[0] = 0
    error[0] = int(domain_error[0] != 0)
    if graded and field.error[0] != 0:
        error[0] |= 2
    ready[0] = int(error[0] == 0)


@wp.kernel
def proposal(
    tree: SurfaceTree,
    field: SizingData,
    graded: bool,
    resolution: int,
    minimum: wp.array[wp.float64],
    mass: wp.array[wp.float64],
    error: wp.array[int],
):
    i = wp.tid()
    lo, hi = wp.vec3d(tree.lower[1]), wp.vec3d(tree.upper[1])
    width = (hi - lo) / wp.float64(resolution)
    cell = wp.vec3i(i % resolution, (i // resolution) % resolution, i // (resolution * resolution))
    lower = lo + wp.cw_mul(width, wp.vec3d(cell))
    upper = lo + wp.cw_mul(width, wp.vec3d(cell + wp.vec3i(1)))
    center = (lower + upper) * wp.float64(0.5)
    scale = wp.max(wp.max(wp.abs(lower), wp.abs(upper)), wp.vec3d(1.0e-45))
    margin = scale * wp.float64(1.1920928955078125e-7) + wp.vec3d(1.0e-44)
    lower -= margin
    upper += margin
    retain = boundary_box(tree, lower, upper)
    if not retain:
        retain = material_side(tree, center) != 0
    value = wp.float64(1.0)
    if graded:
        global_minimum = wp.clamp(-wp.float64(field.tree.weight[1]), field.minimum, field.maximum)
        middle = sizing_value(field, center)
        radius = wp.length((upper - lower) * wp.float64(0.5))
        correction = (wp.abs(middle) + field.gradation * radius) * wp.float64(1.0e-12)
        value = wp.max(global_minimum, middle - field.gradation * radius - correction)
    minimum[i] = value
    weight = wp.float64(0.0)
    if retain:
        inverse = wp.float64(1.0) / value
        weight = inverse * inverse * inverse
    if not wp.isfinite(weight) or value <= wp.float64(0.0):
        wp.atomic_or(error, 0, 4)
        weight = wp.float64(0.0)
    mass[i] = weight


@wp.kernel
def scan_blocks(
    values: wp.array[wp.float64],
    output: wp.array[wp.float64],
    sums: wp.array[wp.float64],
    maximum: bool,
):
    # Warp's cooperative tile scan currently only accepts 32-bit values.
    # Independent short blocks keep this double-precision scan on device.
    block = wp.tid()
    total = wp.float64(0.0)
    for i in range(block * 64, wp.min((block + 1) * 64, values.shape[0])):
        if maximum:
            total = wp.max(total, values[i])
        else:
            total += values[i]
        output[i] = total
    sums[block] = total


@wp.kernel
def scan_add(output: wp.array[wp.float64], sums: wp.array[wp.float64], maximum: bool):
    i = wp.tid()
    if i // 64 > 0:
        if maximum:
            output[i] = wp.max(output[i], sums[i // 64 - 1])
        else:
            output[i] += sums[i // 64 - 1]


@wp.kernel
def check_cdf(mass: wp.array[wp.float64], cdf: wp.array[wp.float64], error: wp.array[int]):
    i = wp.tid()
    previous = wp.float64(0.0)
    if i > 0:
        previous = cdf[i - 1]
    value = cdf[i]
    if not wp.isfinite(value) or value < previous:
        wp.atomic_or(error, 0, 4)
    if mass[i] > wp.float64(0.0) and value <= previous:
        wp.atomic_or(error, 0, 4)
    if i == cdf.shape[0] - 1 and value <= wp.float64(0.0):
        wp.atomic_or(error, 0, 4)


@wp.kernel
def ready_samples(error: wp.array[int], ready: wp.array[int]):
    ready[0] = int(error[0] == 0)


@wp.kernel
def sample_sites(
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
def check_duplicates(owners: wp.array[int], error: wp.array[int]):
    i = wp.tid()
    if owners[i] != i:
        wp.atomic_or(error, 0, 16)


@wp.kernel
def publish(error: wp.array[int], count: wp.array[int], n: int):
    count[0] = 0
    if error[0] == 0:
        count[0] = n


class DoubleScan:
    def __init__(self, n, device):
        self.levels = []
        while True:
            blocks = (n + 63) // 64
            self.levels.append(
                (
                    wp.empty(blocks, dtype=wp.float64, device=device),
                    wp.empty(blocks, dtype=wp.float64, device=device),
                )
            )
            if blocks == 1:
                break
            n = blocks

    def run(self, values, output, level=0, *, maximum=False):
        sums, scanned = self.levels[level]
        wp.launch(
            scan_blocks,
            dim=len(sums),
            inputs=[values, output, sums, maximum],
            device=values.device,
        )
        if len(sums) > 1:
            self.run(sums, scanned, level + 1, maximum=maximum)
            wp.launch(
                scan_add, len(output), inputs=[output, scanned, maximum], device=values.device
            )


class InitialSamplingWorkspace:
    """Preallocated sampling from material with density proportional to mu^-3.

    Caller refits the domain and optional field before run(). The seed is a
    one-element device int array; unchanged seed and geometry repeat a sample.
    Failure publishes count zero, never a silently shortened point cloud.
    """

    def __init__(self, domain, n, *, sizing=None, resolution=32, seed=2026, max_attempts=4096):
        self.domain, self.field, self.device = domain, sizing, domain.device
        self.n = operator.index(n)
        self.resolution = operator.index(resolution)
        self.max_attempts = operator.index(max_attempts)
        if not 1 <= self.n < 2**29:
            raise ValueError("Sample count must be in [1,2**29)")
        if not 1 <= self.resolution <= 128:
            raise ValueError("Proposal grid resolution must be in [1,128]")
        if not 1 <= self.max_attempts <= 1000000:
            raise ValueError("max_attempts must be in [1,1000000]")
        if sizing is not None and sizing.device != self.device:
            raise ValueError("Sizing field and domain must share a device")
        with wp.ScopedDevice(self.device):
            if isinstance(seed, wp.array):
                if (
                    seed.dtype != wp.int32
                    or seed.shape != (1,)
                    or seed.device != self.device
                    or not seed.is_contiguous
                ):
                    raise ValueError("Seed must be one contiguous device int32")
                self.seed = seed
            else:
                self.seed = wp.full(1, operator.index(seed), dtype=int)
            self.points = wp.zeros(self.n, dtype=wp.vec3)
            self.count = wp.zeros(1, dtype=int)
            self.error = wp.zeros(1, dtype=int)
            self.attempts = wp.zeros(self.n, dtype=int)
            self._ready = wp.zeros(1, dtype=int)
            self._dummy = wp.zeros(1, dtype=int)
            self._field = SizingData()
            self._field.error = self._dummy
            if sizing is not None:
                self._field = sizing.data
            self._cells = self.resolution**3
            self._minimum = wp.empty(self._cells, dtype=wp.float64)
            self._mass = wp.empty_like(self._minimum)
            self._cdf = wp.empty_like(self._minimum)
            self._scan = DoubleScan(self._cells, self.device)
            self._weights = wp.zeros(self.n, dtype=float)
            self._bits = wp.array(
                ptr=self.points.ptr, shape=(self.n, 3), dtype=wp.uint32, device=self.device
            )
            self._table = wp.empty(1 << (2 * self.n - 1).bit_length(), dtype=int)
            self._buckets = wp.empty(self.n, dtype=int)
            self._owners = wp.empty(self.n, dtype=int)
            wp.load_module(module=__name__, device=self.device)
            with wp.ScopedCapture() as capture:
                self._record()
            self.graph = capture.graph
        wp.synchronize_device(self.device)

    def _launch(self, kernel, n, inputs):
        wp.launch(kernel, n, inputs=inputs, device=self.device)

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
        wp.capture_if(self._ready, self._sample)
        self._launch(publish, 1, [self.error, self.count, self.n])

    def _sample(self):
        self._launch(
            proposal,
            self._cells,
            [
                self.domain.tree,
                self._field,
                self.field is not None,
                self.resolution,
                self._minimum,
                self._mass,
                self.error,
            ],
        )
        self._scan.run(self._mass, self._cdf)
        # Hierarchical floating addition can make a zero-mass plateau decrease
        # by an ulp where block carries associate differently. A prefix maximum
        # restores a valid CDF; lost positive increments still fail check_cdf.
        self._scan.run(self._cdf, self._cdf, maximum=True)
        self._launch(check_cdf, self._cells, [self._mass, self._cdf, self.error])
        self._launch(ready_samples, 1, [self.error, self._ready])
        wp.capture_if(self._ready, self._draw)

    def _draw(self):
        self._launch(
            sample_sites,
            self.n,
            [
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
            _canonical.insert,
            self.n,
            [self.points, self._weights, self._bits, self._table, self._buckets],
        )
        self._launch(_canonical.resolve, self.n, [self._table, self._buckets, self._owners])
        self._launch(check_duplicates, self.n, [self._owners, self.error])

    def run(self, stream=None):
        stream = wp.get_stream(self.device) if stream is None else stream
        if stream.device != self.device:
            raise ValueError("Sampling stream must use the domain device")
        with wp.ScopedStream(stream):
            if stream.is_capturing:
                if stream not in self.device.captures:
                    raise RuntimeError("Use Warp ScopedCapture for conditional graphs")
                self._record()
            else:
                wp.capture_launch(self.graph, stream=stream)
        return self
