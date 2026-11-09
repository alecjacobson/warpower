"""Optional bounded symbolic candidate construction sharing existing scratch."""

import warp as wp

from . import _filtered_predicates as fp
from ._graph import begin_chunks, next_chunk
from ._symbolic_tetrahedra import build_cells

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.kernel
def affine_rank(points: wp.array[wp.vec3], error: wp.array[int], ready: wp.array[int]):
    ready[0] = 0
    a = points[0]
    second = int(-1)
    third = int(-1)
    for i in range(1, points.shape[0]):
        p = points[i]
        if p[0] != a[0] or p[1] != a[1] or p[2] != a[2]:
            second = i
            break
    if second >= 0:
        for i in range(1, points.shape[0]):
            independent = False
            for k in range(4):
                q = wp.vec3(0.0)
                if k < 3:
                    q[k] = 1.0
                o = fp.orient3d(a, points[second], points[i], q)
                if o.valid == 0:
                    error[0] = 4
                    return
                if o.sign != 0:
                    independent = True
            if independent:
                third = i
                break
    if third >= 0:
        for i in range(1, points.shape[0]):
            o = fp.orient3d(a, points[second], points[third], points[i])
            if o.valid == 0:
                error[0] = 4
                return
            if o.sign != 0:
                ready[0] = 1
                return
    error[0] = 1


@wp.kernel
def domain_box(
    bounds: wp.array[wp.vec3],
    factor: wp.array[wp.float64],
    domain: wp.array[wp.vec3],
    error: wp.array[int],
    running: wp.array[int],
    allow_reset: bool,
):
    if error[0] != 0:
        running[0] = 0
        return
    lo, hi = wp.vec3d(bounds[0]), wp.vec3d(bounds[1])
    span = wp.max(hi[0] - lo[0], wp.max(hi[1] - lo[1], hi[2] - lo[2]))
    padding = wp.max(span * factor[0], wp.float64(1.17549435e-38))
    lower = wp.vec3(lo - wp.vec3d(padding))
    upper = wp.vec3(hi + wp.vec3d(padding))
    finite = True
    for k in range(3):
        finite = finite and wp.isfinite(lower[k]) and wp.isfinite(upper[k])
    if not finite and allow_reset and factor[0] > wp.float64(1.0):
        factor[0] = wp.float64(1.0)
        lower = wp.vec3(lo - wp.vec3d(span))
        upper = wp.vec3(hi + wp.vec3d(span))
        finite = True
        for k in range(3):
            finite = finite and wp.isfinite(lower[k]) and wp.isfinite(upper[k])
    if not finite:
        error[0] = 64
        running[0] = 0
    else:
        domain[0] = lower
        domain[1] = upper
        running[0] = 1


@wp.kernel
def cell_errors(status: wp.array[int], error: wp.array[int]):
    s = status[wp.tid()]
    if s == 6:
        wp.atomic_or(error, 0, 4)
    elif s != 0 and s != 3:
        wp.atomic_or(error, 0, 2)


@wp.kernel
def candidate_ready(
    count: wp.array[int], capacity: int, error: wp.array[int], ready: wp.array[int]
):
    if count[0] > capacity:
        wp.atomic_or(error, 0, 8)
    ready[0] = int(error[0] == 0)


@wp.kernel
def orient_candidates(
    points: wp.array[wp.vec3],
    source: wp.array2d[int],
    count: wp.array[int],
    output: wp.array2d[int],
    error: wp.array[int],
):
    i = wp.tid()
    if i >= count[0]:
        return
    a, b, c, d = source[i, 0], source[i, 1], source[i, 2], source[i, 3]
    result = fp.orient3d(points[a], points[b], points[c], points[d])
    if result.valid == 0:
        wp.atomic_or(error, 0, 4)
    output[i, 0] = a
    output[i, 1] = b
    output[i, 2] = c
    output[i, 3] = d
    if result.sign < 0:
        output[i, 2] = d
        output[i, 3] = c


@wp.kernel
def finish_round(
    certificate: wp.array[int],
    error: wp.array[int],
    count: wp.array[int],
    factor: wp.array[wp.float64],
    expansions: wp.array[int],
    maximum: int,
    running: wp.array[int],
):
    running[0] = 0
    if error[0] == 0 and certificate[0] != 0:
        # Only missing support/coverage can be repaired by a larger box.
        if (certificate[0] & (~192)) != 0:
            error[0] = 16
        elif expansions[0] + 1 >= maximum:
            error[0] = 32
        else:
            expansions[0] += 1
            factor[0] *= wp.float64(8.0)
            running[0] = 1
    if error[0] != 0 or running[0] != 0:
        count[0] = 0


class SymbolicConstructor:
    """Private replay layer reusing base cell scratch and certificate buffers."""

    def __init__(self, base, certificate, candidate, candidate_count, *, max_expansions=16):
        self.base, self.certificate = base, certificate
        self.candidate, self.candidate_count = candidate, candidate_count
        self.device, self.n = base.device, base.n
        self.max_expansions = max_expansions
        # Original-coordinate FP64 tuning must not silently change the
        # independent symbolic recovery schedule. Every scratch view remains.
        self._capacities = tuple(sorted(base._robust))
        with wp.ScopedDevice(self.device):
            self.error = wp.zeros(1, dtype=int)
            self.expansions = wp.zeros(1, dtype=int)
            self._factor = wp.ones(1, dtype=wp.float64)
            self._domain = wp.empty(2, dtype=wp.vec3)
            self._ready = wp.zeros(1, dtype=int)
            self._running = wp.zeros(1, dtype=int)
            wp.load_module(module=__name__, device=self.device, block_dim=32)
            wp.load_module(module="warpower._symbolic_tetrahedra", device=self.device, block_dim=32)

    def _launch(self, kernel, n, args):
        wp.launch(kernel, n, inputs=args, device=self.device, block_dim=32)

    def run(self, tree, bounds):
        self.error.zero_()
        self.expansions.zero_()
        self.certificate.tetrahedra_count.zero_()
        self._running.zero_()
        self._launch(affine_rank, 1, [self.base.points, self.error, self._ready])
        wp.capture_if(self._ready, lambda: self._start(bounds))
        wp.capture_while(self._running, lambda: self._round(tree, bounds))

    def _start(self, bounds):
        self._launch(
            domain_box, 1, [bounds, self._factor, self._domain, self.error, self._running, True]
        )

    def _round(self, tree, bounds):
        b = self.base
        b._candidate_count.zero_()
        b._tet_status.fill_(1)
        b._job_count.fill_(self.n)
        wp.copy(b._jobs, tree.ids, count=self.n)
        for capacity in self._capacities:
            self._launch(begin_chunks, 1, [b._job_count, b._cursor, b._chunks])

            def body(capacity=capacity):
                self._launch(
                    build_cells,
                    b.retry_chunk,
                    [
                        b.points,
                        b.weights,
                        tree,
                        b._jobs,
                        b._cursor,
                        b._job_count,
                        self._domain,
                        b._robust[capacity],
                        b._tet_status,
                        b._candidates,
                        b._candidate_count,
                    ],
                )
                self._launch(next_chunk, 1, [b._job_count, b._cursor, b._chunks, b.retry_chunk])

            wp.capture_while(b._chunks, body)
            b._failures(b._tet_status)
        self._launch(cell_errors, self.n, [b._tet_status, self.error])
        self._launch(
            candidate_ready, 1, [b._candidate_count, len(b._candidates), self.error, self._ready]
        )
        wp.capture_if(self._ready, lambda: self._certify(tree))
        self._launch(
            finish_round,
            1,
            [
                self.certificate.error,
                self.error,
                self.certificate.tetrahedra_count,
                self._factor,
                self.expansions,
                self.max_expansions,
                self._running,
            ],
        )
        wp.capture_if(
            self._running,
            lambda: self._launch(
                domain_box,
                1,
                [bounds, self._factor, self._domain, self.error, self._running, False],
            ),
        )

    def _certify(self, tree):
        b = self.base
        self._launch(
            orient_candidates,
            len(b._candidates),
            [b.points, b._candidates, b._candidate_count, self.candidate, self.error],
        )
        wp.copy(self.candidate_count, b._candidate_count)
        self.certificate.run(tree)
