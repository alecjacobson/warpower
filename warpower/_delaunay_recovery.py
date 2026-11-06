"""Optional certified Delaunay construction with device-only finite retries."""

import operator

import warp as wp

from . import _canonical, _validation
from ._delaunay_certificate import DelaunayCertificate, direction
from ._graph import DiagramWorkspace, input_ready
from ._site_bvh import SiteHierarchy

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.kernel
def prepare(
    base_error: wp.array[int],
    certificate_error: wp.array[int],
    ready: wp.array[int],
):
    certificate_error[0] = 0
    ready[0] = int(base_error[0] == 0)


@wp.kernel
def begin_retry(
    base_error: wp.array[int],
    certificate_error: wp.array[int],
    error: wp.array[int],
    running: wp.array[int],
):
    error[0] = 0
    if base_error[0] == 1:
        error[0] = 1
    elif base_error[0] != 0 or certificate_error[0] != 0:
        error[0] = 2
    running[0] = int(error[0] == 2)


@wp.kernel
def choose_strategy(
    preferred: wp.array[int], cache: bool, normal: wp.array[int], recovery_first: wp.array[int]
):
    normal[0] = int(not (cache and preferred[0] != 0))
    recovery_first[0] = 1 - normal[0]


@wp.kernel
def fallback_ready(error: wp.array[int], ready: wp.array[int]):
    ready[0] = int(error[0] == 2)


@wp.kernel
def normalize(
    points: wp.array[wp.vec3],
    bounds: wp.array[wp.vec3],
    owners: wp.array[int],
    attempts: wp.array[int],
    normalized: wp.array[wp.vec3],
    weights: wp.array[float],
):
    i = wp.tid()
    lo, hi = wp.vec3d(bounds[0]), wp.vec3d(bounds[1])
    extent = wp.max(hi[0] - lo[0], wp.max(hi[1] - lo[1], hi[2] - lo[2]))
    scale = wp.max(extent, wp.float64(1.0e-300))
    normalized[i] = wp.vec3((wp.vec3d(points[i]) - (lo + hi) * wp.float64(0.5)) / scale)
    amplitude = float(1.0e-5)
    for _ in range(attempts[0]):
        amplitude *= 0.1
    weights[i] = amplitude * direction(owners[i])


@wp.kernel
def candidate_ready(
    candidate_error: wp.array[int], certificate_error: wp.array[int], ready: wp.array[int]
):
    certificate_error[0] = 0
    ready[0] = int(candidate_error[0] == 0)


@wp.kernel
def finish_retry(
    candidate_error: wp.array[int],
    certificate_error: wp.array[int],
    attempts: wp.array[int],
    recovered: wp.array[int],
    running: wp.array[int],
    error: wp.array[int],
    maximum: int,
):
    attempts[0] += 1
    if candidate_error[0] == 0 and certificate_error[0] == 0:
        recovered[0] = 1
        running[0] = 0
        error[0] = 0
    elif attempts[0] >= maximum:
        error[0] = 2
        running[0] = 0


@wp.kernel
def finalize(
    error: wp.array[int],
    count: wp.array[int],
    recovered: wp.array[int],
    preferred: wp.array[int],
    cache: bool,
):
    preferred[0] = int(cache and error[0] == 0 and recovered[0] != 0)
    if error[0] != 0:
        count[0] = 0


class DelaunayWorkspace:
    """Exact-certified ordinary Delaunay output, with bounded recovery attempts.

    Setup allocates two construction workspaces and all certificate storage.
    Replay leaves input positions untouched and performs no host decisions.
    Exhaustion is explicit: error != 0 and tetrahedra_count == 0. This is not
    a universal exact constructor; finite candidate generation can still fail.
    """

    def __init__(
        self,
        points,
        *,
        tetrahedra_capacity=None,
        tetrahedra_cell_capacity=1024,
        chunk_size=None,
        max_attempts=6,
        cache_strategy=True,
    ):
        self.max_attempts = operator.index(max_attempts)
        self.cache_strategy = bool(cache_strategy)
        if not 1 <= self.max_attempts <= 12:
            raise ValueError("max_attempts must be in [1, 12]")
        settings = dict(
            adjacency=False,
            return_tetrahedra=True,
            tetrahedra_capacity=tetrahedra_capacity,
            tetrahedra_cell_capacity=tetrahedra_cell_capacity,
            chunk_size=chunk_size,
        )
        self._base = DiagramWorkspace(points, **settings)
        self.points, self.device, self.n = self._base.points, self._base.device, self._base.n
        if self.n < 4:
            raise ValueError("DelaunayWorkspace requires at least four sites")
        with wp.ScopedDevice(self.device):
            self._normalized = wp.zeros_like(self.points)
            self._weights = wp.zeros(self.n, dtype=float)
            self._auxiliary = DiagramWorkspace(self._normalized, self._weights, **settings)
            self._original = SiteHierarchy(self.points)
            self._original.tree.owners = self._base._hierarchy.owners
            self._candidate = wp.empty_like(self._base.tetrahedra)
            self._candidate_count = wp.empty(1, dtype=int)
            self._certificate = DelaunayCertificate(
                self.points, self._candidate, self._candidate_count, self._base._hierarchy
            )
            self.tetrahedra = self._certificate.tetrahedra
            self.tetrahedra_count = self._certificate.tetrahedra_count
            self.certificate_error = self._certificate.error
            self.base_error = self._base.error
            self.candidate_error = self._auxiliary.error
            self.error = wp.zeros(1, dtype=int)
            self.attempts = wp.zeros(1, dtype=int)
            self.recovered = wp.zeros(1, dtype=int)
            self.base_attempted = wp.zeros(1, dtype=int)
            self.recovery_first = wp.zeros(1, dtype=int)
            self._preferred = wp.zeros(1, dtype=int)
            self._normal = wp.empty(1, dtype=int)
            self._ready = wp.empty(1, dtype=int)
            self._running = wp.empty(1, dtype=int)
            wp.load_module(module=__name__, device=self.device)
            with wp.ScopedCapture() as capture:
                self._record()
            self.graph = capture.graph
        wp.synchronize_device(self.device)

    def _launch(self, kernel, dim, args):
        wp.launch(kernel, dim, inputs=args, device=self.device)

    def _record(self):
        self.tetrahedra_count.zero_()
        self.error.zero_()
        self.attempts.zero_()
        self.recovered.zero_()
        self.base_attempted.zero_()
        self.base_error.zero_()
        self.candidate_error.zero_()
        self.certificate_error.zero_()
        self._running.zero_()
        self._launch(
            choose_strategy,
            1,
            [self._preferred, self.cache_strategy, self._normal, self.recovery_first],
        )
        wp.capture_if(self._normal, self._normal_path)
        wp.capture_if(self.recovery_first, self._preferred_path)
        self._launch(
            finalize,
            1,
            [
                self.error,
                self.tetrahedra_count,
                self.recovered,
                self._preferred,
                self.cache_strategy,
            ],
        )

    def _base_path(self):
        self.base_attempted.fill_(1)
        self._base.run()
        self._launch(
            prepare,
            1,
            [
                self.base_error,
                self.certificate_error,
                self._ready,
            ],
        )
        wp.copy(self._candidate, self._base.tetrahedra)
        wp.copy(self._candidate_count, self._base.tetrahedra_count)
        wp.capture_if(self._ready, self._certificate.run)
        self._launch(
            begin_retry, 1, [self.base_error, self.certificate_error, self.error, self._running]
        )

    def _normal_path(self):
        self._base_path()
        wp.capture_while(
            self._running, lambda: self._retry(self._base._hierarchy, self._base._bounds)
        )

    def _preferred_path(self):
        self._base._validation.zero_()
        self._launch(
            _validation.validate,
            self.n + 1,
            [
                self.points,
                self._base.weights,
                self._base._empty,
                self._base._empty,
                False,
                self._base._validation,
            ],
        )
        self._launch(input_ready, 1, [self._base._validation, self.error, self._ready])
        wp.copy(self._running, self._ready)
        wp.capture_if(self._ready, self._refresh_original)
        wp.capture_while(
            self._running, lambda: self._retry(self._original.tree, self._original._bounds)
        )
        # A changed input may favor the original builder. Retry it once after
        # auxiliary exhaustion; never turn stale connectivity into a success.
        self._launch(fallback_ready, 1, [self.error, self._ready])
        wp.capture_if(self._ready, self._base_path)

    def _refresh_original(self):
        self._original.rebuild()
        self._base._owner_table.fill_(-1)
        self._launch(
            _canonical.insert,
            self.n,
            [
                self.points,
                self._base.weights,
                self._base._bits,
                self._base._owner_table,
                self._base._buckets,
            ],
        )
        self._launch(
            _canonical.resolve,
            self.n,
            [self._base._owner_table, self._base._buckets, self._original.tree.owners],
        )

    def _retry(self, tree, bounds):
        self._launch(
            normalize,
            self.n,
            [
                self.points,
                bounds,
                tree.owners,
                self.attempts,
                self._normalized,
                self._weights,
            ],
        )
        self._auxiliary.run()
        wp.copy(self._candidate, self._auxiliary.tetrahedra)
        wp.copy(self._candidate_count, self._auxiliary.tetrahedra_count)
        self._launch(
            candidate_ready, 1, [self.candidate_error, self.certificate_error, self._ready]
        )
        wp.capture_if(self._ready, lambda: self._certificate.run(tree))
        self._launch(
            finish_retry,
            1,
            [
                self.candidate_error,
                self.certificate_error,
                self.attempts,
                self.recovered,
                self._running,
                self.error,
                self.max_attempts,
            ],
        )

    def run(self, stream=None):
        stream = wp.get_stream(self.device) if stream is None else stream
        if stream.device != self.device:
            raise ValueError("Workspace stream must use the input device")
        with wp.ScopedStream(stream):
            if stream.is_capturing:
                if stream not in self.device.captures:
                    raise RuntimeError(
                        "Use Warp ScopedCapture on the shared stream for conditional graphs"
                    )
                self._record()
            else:
                wp.capture_launch(self.graph, stream=stream)
        return self
