"""Reusable CUDA workspaces with device-controlled graph execution."""

import operator

import warp as wp
from warp.utils import radix_sort_pairs

from . import _adjacency as adj
from . import _bvh as bvh
from . import _canonical as canonical
from . import _validation
from ._api import _chunk_size, _points_array, _weights_array
from ._cell import Scratch64, allocate_scratch
from ._construct import collect_failures, make_construct_kernel
from ._graph_construct import make_retry_kernel

wp.set_module_options({"enable_backward": False})


@wp.kernel
def scan_blocks(values: wp.array[int], output: wp.array[int], sums: wp.array[int]):
    block, lane = wp.tid()
    tile = wp.tile_load(values, shape=256, offset=block * 256)
    wp.tile_store(output, wp.tile_scan_inclusive(tile), offset=block * 256)
    total = wp.tile_sum(tile)[0]
    if lane == 0:
        sums[block] = total


@wp.kernel
def scan_add(output: wp.array[int], sums: wp.array[int]):
    i = wp.tid()
    block = i // 256
    if block > 0:
        output[i] += sums[block - 1]


@wp.kernel
def reset_bounds(bounds: wp.array[wp.vec3]):
    bounds[0] = wp.vec3(1.0e30)
    bounds[1] = wp.vec3(-1.0e30)


@wp.kernel
def input_ready(validation: wp.array[int], error: wp.array[int], ready: wp.array[int]):
    ready[0] = int(validation[0] == 0)
    if validation[0] != 0:
        error[0] = 1


@wp.kernel
def begin_chunks(count: wp.array[int], cursor: wp.array[int], condition: wp.array[int]):
    cursor[0] = 0
    condition[0] = int(count[0] > 0)


@wp.kernel
def next_chunk(count: wp.array[int], cursor: wp.array[int], condition: wp.array[int], size: int):
    cursor[0] += size
    condition[0] = int(cursor[0] < count[0])


@wp.kernel
def prepare_repairs(
    jobs: wp.array[int],
    active: wp.array[int],
    counts: wp.array[int],
    base: wp.array2d[int],
    retry: wp.array2d[int],
    sources: wp.array[int],
    status: wp.array[int],
):
    slot = wp.tid()
    if slot >= active[0]:
        return
    i = jobs[slot]
    for k in range(retry.shape[1]):
        value = int(-1)
        if k < counts[i]:
            value = adj.neighbor_at(i, k, base, retry, sources)
        retry[i, k] = value
    sources[i] = i
    status[i] = 1


@wp.kernel
def check_cells(status: wp.array[int], error: wp.array[int]):
    i = wp.tid()
    if status[i] != 0 and status[i] != 3:
        wp.atomic_or(error, 0, 2)


@wp.kernel
def adjacency_ready(
    offsets: wp.array[int],
    capacity: int,
    count: wp.array[int],
    error: wp.array[int],
    ready: wp.array[int],
):
    total = offsets[offsets.shape[0] - 1]
    count[0] = total
    ready[0] = int(total >= 0 and total <= capacity)
    if total < 0 or total > capacity:
        wp.atomic_or(error, 0, 4)


@wp.kernel
def finalize(
    error: wp.array[int],
    offsets: wp.array[int],
    adjacency_count: wp.array[int],
    tetrahedra_count: wp.array[int],
):
    i = wp.tid()
    if error[0] != 0:
        offsets[i] = 0
        if i == 0:
            adjacency_count[0] = 0
            tetrahedra_count[0] = 0


@wp.kernel
def start_tets(
    bounds: wp.array[wp.vec3],
    factor: wp.array[wp.float64],
    domain: wp.array[wp.vec3],
    expansions: wp.array[int],
    condition: wp.array[int],
):
    span = wp.float64(0.0)
    for axis in range(3):
        span = wp.max(span, wp.float64(bounds[1][axis]) - wp.float64(bounds[0][axis]))
    padding = wp.max(span * factor[0], wp.float64(1.17549435e-38))
    domain[0] = wp.vec3(wp.vec3d(bounds[0]) - wp.vec3d(padding))
    domain[1] = wp.vec3(wp.vec3d(bounds[1]) + wp.vec3d(padding))
    expansions[0] = 0
    condition[0] = int(span > wp.float64(0.0))


@wp.kernel
def candidates_ready(
    failed: wp.array[int],
    candidates: wp.array[int],
    capacity: int,
    error: wp.array[int],
    ready: wp.array[int],
):
    if failed[0] != 0:
        wp.atomic_or(error, 0, 2)
    if candidates[0] > capacity:
        wp.atomic_or(error, 0, 8)
    ready[0] = int(error[0] == 0)


@wp.kernel
def finish_tets(
    bounds: wp.array[wp.vec3],
    domain: wp.array[wp.vec3],
    factor: wp.array[wp.float64],
    expansions: wp.array[int],
    problems: wp.array[int],
    count: wp.array[int],
    error: wp.array[int],
    condition: wp.array[int],
    max_expansions: int,
):
    if problems[1] != 0 or problems[2] != 0:
        wp.atomic_or(error, 0, 16)
    if problems[0] != 0 or problems[3] != 0 or problems[5] != 0:
        wp.atomic_or(error, 0, 32)
    condition[0] = 0
    if error[0] == 0 and (count[0] == 0 or problems[4] != 0):
        expansions[0] += 1
        if expansions[0] >= max_expansions:
            wp.atomic_or(error, 0, 64)
        else:
            factor[0] *= wp.float64(8.0)
            span = wp.float64(0.0)
            for axis in range(3):
                span = wp.max(span, wp.float64(bounds[1][axis]) - wp.float64(bounds[0][axis]))
            padding = wp.max(span * factor[0], wp.float64(1.17549435e-38))
            domain[0] = wp.vec3(wp.vec3d(bounds[0]) - wp.vec3d(padding))
            domain[1] = wp.vec3(wp.vec3d(bounds[1]) + wp.vec3d(padding))
            condition[0] = 1
            for axis in range(3):
                if not wp.isfinite(domain[0][axis]) or not wp.isfinite(domain[1][axis]):
                    wp.atomic_or(error, 0, 64)
                    condition[0] = 0
    if error[0] != 0:
        factor[0] = wp.float64(0.5)
        count[0] = 0


def _positive(value, name):
    value = operator.index(value)
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


class DiagramWorkspace:
    """Preallocate once; call ``run()`` inside a Warp capture or asynchronously.

    Inputs keep fixed addresses and shapes. Outputs are capacity arrays plus
    device counts. ``error[0] == 0`` means valid output. Setup may allocate and
    synchronize; run/captured replay never copies results to the CPU.
    """

    def __init__(
        self,
        points,
        weights=None,
        *,
        return_tetrahedra=False,
        adjacency=True,
        adjacency_capacity=None,
        tetrahedra_capacity=None,
        chunk_size=None,
        leaf_size=4,
        max_expansions=12,
    ):
        self.points, self._torch = _points_array(points)
        self.device = self.points.device
        if not self.device.is_cuda:
            raise ValueError("DiagramWorkspace requires CUDA inputs")
        if not wp.is_conditional_graph_supported():
            raise RuntimeError("DiagramWorkspace requires CUDA conditional graphs (CUDA 12.4+)")
        # Copies would sever the producer's pointer relationship.
        if isinstance(points, wp.array) and not points.is_contiguous:
            raise ValueError("Workspace inputs must be contiguous")
        if self._torch and not points.is_contiguous():
            raise ValueError("Workspace inputs must be contiguous")
        self.n = len(self.points)
        self.weights = _weights_array(weights, self.n, self.device)
        if isinstance(weights, wp.array) and not weights.is_contiguous:
            raise ValueError("Workspace weights must be contiguous")
        if (
            weights is not None
            and type(weights).__module__.startswith("torch")
            and not weights.is_contiguous()
        ):
            raise ValueError("Workspace weights must be contiguous")
        self.weighted = weights is not None
        self._adjacency = bool(adjacency)
        self._tetrahedra = bool(return_tetrahedra)
        if not self._adjacency and not self._tetrahedra:
            raise ValueError("Request adjacency, tetrahedra, or both")
        self.chunk_size = max(1, _chunk_size(self.n, 64, self.device, chunk_size))
        self.retry_chunk = min(self.chunk_size, 4096)
        self.leaf_size = _positive(leaf_size, "leaf_size")
        self.max_expansions = _positive(max_expansions, "max_expansions")
        ac = (
            max(1, 48 * self.n)
            if adjacency_capacity is None
            else _positive(adjacency_capacity, "adjacency_capacity")
        )
        tc = (
            max(16, 8 * self.n)
            if tetrahedra_capacity is None
            else _positive(tetrahedra_capacity, "tetrahedra_capacity")
        )
        if max(ac, 8 * tc, self.n * 512) >= 2**31:
            raise ValueError("Workspace capacities exceed int32 indexing")
        self._input_owners = (points, weights)
        with wp.ScopedDevice(self.device):
            self.error = wp.zeros(1, dtype=int)
            self.adjacency_count = wp.zeros(1, dtype=int)
            self.tetrahedra_count = wp.zeros(1, dtype=int)
            self.adjacency = wp.empty(ac if self._adjacency else 0, dtype=int)
            self.offsets = wp.zeros(self.n + 1, dtype=int)
            self.status = wp.empty(self.n, dtype=int)
            self.tetrahedra = wp.empty((tc if self._tetrahedra else 0, 4), dtype=int)
            self._validation = wp.zeros(1, dtype=int)
            self._ready = wp.zeros(1, dtype=int)
            self._empty = wp.empty(0, dtype=int)
            self._jobs = wp.empty(self.n, dtype=int)
            self._job_count = wp.zeros(1, dtype=int)
            self._cursor = wp.zeros(1, dtype=int)
            self._chunks = wp.zeros(1, dtype=int)
            self._bounds = wp.empty(2, dtype=wp.vec3)
            self._groups = (self.n + 255) // 256
            self._lo = wp.empty(self._groups, dtype=wp.vec3)
            self._hi = wp.empty(self._groups, dtype=wp.vec3)
            self._keys = wp.zeros(2 * self.n, dtype=wp.uint64)
            self._ids = wp.zeros(2 * self.n, dtype=int)
            self._segment = self._tree(1 << max(0, (self.n - 1).bit_length()), 1)
            self._hierarchy = self._tree(max(1, self.n), self.leaf_size)
            self._owner_table = wp.empty(1 << max(0, (2 * self.n - 1).bit_length()), dtype=int)
            self._buckets = wp.empty(self.n, dtype=int)
            self._bits = wp.array(
                ptr=self.points.ptr, shape=(self.n, 3), dtype=wp.uint32, device=self.device
            )
            if self._adjacency:
                self._base = wp.empty((self.n, 64), dtype=int)
                self._retry = wp.empty((self.n, 512), dtype=int)
                self._counts = wp.empty(self.n, dtype=int)
                self._source = wp.empty(self.n, dtype=int)
                self._flags = wp.empty(self.n, dtype=int)
                self._sizes = wp.zeros(self.n + 1, dtype=int)
                self._heads = wp.empty(self.n + 1, dtype=int)
                self._scan_levels = []
                length = self.n + 1
                while True:
                    blocks = (length + 255) // 256
                    self._scan_levels.append(
                        (wp.empty(blocks, dtype=int), wp.empty(blocks, dtype=int))
                    )
                    if blocks == 1:
                        break
                    length = blocks
                self._scratch = allocate_scratch(self.chunk_size, 64, self.device)
                self._retry32 = allocate_scratch(self.retry_chunk, 128, self.device)
            self._scratch64 = allocate_scratch(self.retry_chunk, 512, self.device, robust=True)
            self._robust = {}
            for capacity in (128, 256, 512):
                scratch = Scratch64()
                for name in ("planes", "vertices", "positions", "cartesian", "next", "neighbors"):
                    setattr(scratch, name, getattr(self._scratch64, name))
                scratch.capacity = capacity
                self._robust[capacity] = scratch
            if self._tetrahedra:
                self._init_tetrahedra(tc)
            # Prewarm Warp's native primitive scratch outside any conditional body.
            if self.n:
                radix_sort_pairs(self._keys, self._ids, self.n, end_bit=63)
            # Compiling/capturing is setup, never part of replay.
            with wp.ScopedCapture(device=self.device) as capture:
                self._record()
            self.graph = capture.graph
            # Initial fills/native primitive setup must finish before run() may
            # use a different stream. This barrier is outside captured execution.
            wp.synchronize_stream(wp.get_stream(self.device))

    def _tree(self, base, leaf_size):
        tree = bvh.Tree()
        tree.n, tree.leaf_base, tree.leaf_size = self.n, base, leaf_size
        tree.ids = self._ids
        tree.lower = wp.empty(2 * base, dtype=wp.vec3)
        tree.upper = wp.empty(2 * base, dtype=wp.vec3)
        tree.weight = wp.empty(2 * base, dtype=float)
        tree.owners = wp.empty(self.n, dtype=int)
        tree.left = wp.empty(2 * base, dtype=int)
        tree.right = wp.empty(2 * base, dtype=int)
        tree.start = wp.empty(2 * base, dtype=int)
        tree.count = wp.empty(2 * base, dtype=int)
        return tree

    def _launch(self, kernel, dim, args, **kwargs):
        wp.launch(kernel, dim, inputs=args, device=self.device, **kwargs)

    def _record(self):
        self.error.zero_()
        self.adjacency_count.zero_()
        self.tetrahedra_count.zero_()
        self._validation.zero_()
        self.offsets.zero_()
        self.status.fill_(-1)
        self._launch(
            _validation.validate,
            self.n + 1,
            [self.points, self.weights, self._empty, self._empty, False, self._validation],
        )
        self._launch(input_ready, 1, [self._validation, self.error, self._ready])
        if self.n:
            wp.capture_if(self._ready, self._record_valid)
        self._launch(
            finalize,
            self.n + 1,
            [self.error, self.offsets, self.adjacency_count, self.tetrahedra_count],
        )

    def _record_valid(self):
        self._launch(reset_bounds, 1, [self._bounds])
        wp.launch_tiled(
            bvh.block_bounds,
            dim=self._groups,
            inputs=[self.points, self._lo, self._hi],
            block_dim=256,
            device=self.device,
        )
        self._launch(bvh.merge_bounds, self._groups, [self._lo, self._hi, self._bounds])
        self._launch(bvh.morton_keys, self.n, [self.points, self._bounds, self._keys, self._ids])
        radix_sort_pairs(self._keys, self._ids, self.n, end_bit=63)
        self._launch(
            bvh.make_leaves, self._segment.leaf_base, [self.points, self.weights, self._segment]
        )
        level = self._segment.leaf_base // 2
        while level:
            self._launch(bvh.merge_level, level, [self._segment, level])
            level //= 2
        self._launch(bvh.radix_nodes, self.n, [self._keys, self._segment, self._hierarchy])
        self._launch(bvh.radix_leaves, self.n, [self.points, self.weights, self._hierarchy])
        self._owner_table.fill_(-1)
        self._launch(
            canonical.insert,
            self.n,
            [self.points, self.weights, self._bits, self._owner_table, self._buckets],
        )
        self._launch(
            canonical.resolve, self.n, [self._owner_table, self._buckets, self._hierarchy.owners]
        )
        if self._adjacency:
            self._record_adjacency()
        if self._tetrahedra and self.n >= 4:
            self._record_tetrahedra()

    def _failures(self, status):
        self._job_count.zero_()
        self._launch(collect_failures, self.n, [status, self._jobs, self._job_count])

    def _retry_pass(self, capacity, robust):
        self._launch(begin_chunks, 1, [self._job_count, self._cursor, self._chunks])

        def body():
            self._launch(
                make_retry_kernel(self.weighted, capacity, robust, 32),
                self.retry_chunk * 32,
                [
                    self.points,
                    self.weights,
                    self._hierarchy,
                    self._jobs,
                    self._cursor,
                    self._job_count,
                    self._source,
                    True,
                    self._robust[capacity] if robust else self._retry32,
                    self._retry,
                    self._counts,
                    self.status,
                    self._base,
                    False,
                    self._empty,
                    self._empty,
                    False,
                ],
                block_dim=32,
            )
            self._launch(
                next_chunk, 1, [self._job_count, self._cursor, self._chunks, self.retry_chunk]
            )

        wp.capture_while(self._chunks, body)

    def _retries(self, repair=False):
        if repair:
            self._launch(
                prepare_repairs,
                self.n,
                [
                    self._jobs,
                    self._job_count,
                    self._counts,
                    self._base,
                    self._retry,
                    self._source,
                    self.status,
                ],
            )
        for capacity, robust in ((128, False), (128, True), (256, True), (512, True)):
            if repair and not robust:
                continue
            self._retry_pass(capacity, robust)
            self._failures(self.status)

    def _asymmetry(self):
        self._flags.zero_()
        self._job_count.zero_()
        self._launch(
            adj.mark_asymmetric,
            self.n,
            [self._counts, self._base, self._retry, self._source, self._flags],
        )
        self._launch(adj.gather_marked, self.n, [self._flags, self._jobs, self._job_count])

    def _record_adjacency(self):
        self._source.fill_(-1)
        for start in range(0, self.n, self.chunk_size):
            self._launch(
                make_construct_kernel(self.weighted, 128),
                min(self.chunk_size, self.n - start),
                [
                    self.points,
                    self.weights,
                    self._hierarchy,
                    self._ids,
                    start,
                    False,
                    self._scratch,
                    self._base,
                    self._counts,
                    self.status,
                    self._base,
                    False,
                    self._empty,
                    self._empty,
                    False,
                ],
                block_dim=32,
            )
        self._failures(self.status)
        wp.capture_if(self._job_count, self._retries)
        self._asymmetry()
        for _ in range(3):
            wp.capture_if(self._job_count, self._repair_cycle)
        self._launch(check_cells, self.n, [self.status, self.error])
        wp.capture_if(self._job_count, lambda: self._compact(False), lambda: self._compact(True))

    def _repair_cycle(self):
        self._retries(repair=True)
        self._asymmetry()

    def _compact(self, direct):
        self._sizes.zero_()
        if direct:
            wp.copy(self._sizes, self._counts, dest_offset=1, count=self.n)
        else:
            self._launch(
                adj.count_symmetric,
                self.n,
                [self._counts, self._base, self._retry, self._source, self._sizes],
            )
        self._scan(self._sizes, self.offsets)
        self._launch(
            adjacency_ready,
            1,
            [self.offsets, len(self.adjacency), self.adjacency_count, self.error, self._ready],
        )

        def fill():
            if direct:
                self._launch(
                    adj.fill_direct,
                    self.n,
                    [
                        self._counts,
                        self._base,
                        self._retry,
                        self._source,
                        self.offsets,
                        self.adjacency,
                    ],
                )
            else:
                wp.copy(self._heads, self.offsets)
                self._launch(
                    adj.fill_symmetric,
                    self.n,
                    [
                        self._counts,
                        self._base,
                        self._retry,
                        self._source,
                        self._heads,
                        self.adjacency,
                    ],
                )

        wp.capture_if(self._ready, fill)

    def _scan(self, values, output, level=0):
        sums, scanned = self._scan_levels[level]
        wp.launch_tiled(
            scan_blocks,
            dim=len(sums),
            inputs=[values, output, sums],
            block_dim=256,
            device=self.device,
        )
        if len(sums) > 1:
            self._scan(sums, scanned, level + 1)
            self._launch(scan_add, len(output), [output, scanned])

    def _init_tetrahedra(self, capacity):
        from . import _graph_tetrahedra, _tetrahedra

        self._gt, self._et = _graph_tetrahedra, _tetrahedra
        self._candidates = wp.empty((capacity, 4), dtype=int)
        self._candidate_count = wp.zeros(1, dtype=int)
        self._tet_table = wp.empty(1 << (2 * capacity - 1).bit_length(), dtype=int)
        self._faces = wp.empty((4 * capacity, 4), dtype=int)
        size = 1 << (8 * capacity - 1).bit_length()
        self._face_table = wp.empty(size, dtype=int)
        self._face_counts = wp.empty(size, dtype=int)
        self._face_sides = wp.empty(size, dtype=int)
        self._tet_status = wp.empty(self.n, dtype=int)
        self._problems = wp.zeros(6, dtype=int)
        self._domain = wp.empty(2, dtype=wp.vec3)
        self._factor = wp.full(1, 0.5, dtype=wp.float64)
        self.tetrahedra_expansions = wp.zeros(1, dtype=int)
        self._expand = wp.zeros(1, dtype=int)

    def _record_tetrahedra(self):
        self._launch(
            start_tets,
            1,
            [self._bounds, self._factor, self._domain, self.tetrahedra_expansions, self._expand],
        )
        wp.capture_while(self._expand, self._tet_iteration)

    def _tet_iteration(self):
        self._candidate_count.zero_()
        self.tetrahedra_count.zero_()
        self._problems.zero_()
        self._tet_status.fill_(1)
        self._job_count.fill_(self.n)
        wp.copy(self._jobs, self._ids, count=self.n)
        for capacity in (128, 256, 512):
            self._launch(begin_chunks, 1, [self._job_count, self._cursor, self._chunks])

            def body(capacity=capacity):
                self._launch(
                    self._gt.make_cells(capacity),
                    self.retry_chunk,
                    [
                        self.points,
                        self.weights,
                        self._hierarchy,
                        self._jobs,
                        self._cursor,
                        self._job_count,
                        self._domain,
                        self._robust[capacity],
                        self._tet_status,
                        self._candidates,
                        self._candidate_count,
                    ],
                    block_dim=32,
                )
                self._launch(
                    next_chunk, 1, [self._job_count, self._cursor, self._chunks, self.retry_chunk]
                )

            wp.capture_while(self._chunks, body)
            self._failures(self._tet_status)
        self._launch(
            candidates_ready,
            1,
            [
                self._job_count,
                self._candidate_count,
                len(self._candidates),
                self.error,
                self._ready,
            ],
        )
        wp.capture_if(self._ready, self._tet_validate)
        self._launch(
            finish_tets,
            1,
            [
                self._bounds,
                self._domain,
                self._factor,
                self.tetrahedra_expansions,
                self._problems,
                self.tetrahedra_count,
                self.error,
                self._expand,
                self.max_expansions,
            ],
        )

    def _tet_validate(self):
        self._tet_table.fill_(-1)
        self._launch(
            self._gt.unique_candidates,
            len(self._candidates),
            [self._candidates, self._tet_table, self._candidate_count],
        )
        self._launch(
            self._et.orient_unique,
            len(self._tet_table),
            [
                self.points,
                self._candidates,
                self._tet_table,
                self.tetrahedra,
                self.tetrahedra_count,
                self._problems,
            ],
        )
        self._launch(
            self._gt.validate_power_spheres,
            len(self.tetrahedra),
            [
                self.points,
                self.weights,
                self._hierarchy,
                self.tetrahedra,
                self._problems,
                self.tetrahedra_count,
            ],
            block_dim=32,
        )
        self._face_table.fill_(-1)
        self._face_counts.zero_()
        self._face_sides.zero_()
        self._launch(
            self._gt.make_faces,
            len(self._faces),
            [self.tetrahedra, self._faces, self.tetrahedra_count],
        )
        self._launch(
            self._gt.index_faces,
            len(self._faces),
            [
                self.points,
                self.tetrahedra,
                self._faces,
                self._face_table,
                self._face_counts,
                self._face_sides,
                self.tetrahedra_count,
            ],
        )
        self._launch(
            self._et.validate_faces,
            len(self._face_table),
            [
                self.points,
                self._hierarchy,
                self._faces,
                self._face_table,
                self._face_counts,
                self._face_sides,
                self._problems,
            ],
            block_dim=32,
        )

    def run(self, stream=None):
        """Enqueue on a Warp stream, or inline into its active Warp capture."""
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
