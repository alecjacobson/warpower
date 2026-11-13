"""Native Warp interface and optional zero-copy PyTorch compatibility."""

import operator
import os
import time
from dataclasses import dataclass, field

import numpy as np
import warp as wp

from . import _bvh, _retry, _validation
from ._adjacency import asymmetric_cells, compact, gather_repair_seeds
from ._cell import allocate_scratch
from ._construct import (
    collect_failures,
    init_retries,
    make_construct_kernel,
    merge_retry,
    retry_kinds,
)
from ._cooperative_cell import allocate_scratch as allocate_precision_scratch
from ._precision_construct import make_precision_kernel
from ._query_predicates import check_queries, collect_precision, collect_precision_retry


@dataclass(frozen=True)
class Diagram:
    adjacency: object
    offsets: object
    status: object
    diagnostics: dict = field(default_factory=dict)
    tetrahedra: object = None


def _points_array(points):
    if isinstance(points, wp.array):
        if points.dtype != wp.vec3 or points.ndim != 1:
            raise ValueError("Warp points must be a one-dimensional wp.vec3 array")
        if not points.is_contiguous:
            points = wp.clone(points)
        return points, False
    if isinstance(points, np.ndarray):
        if points.dtype != np.float32 or points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have dtype float32 and shape (N, 3)")
        return wp.array(points, dtype=wp.vec3, device="cuda:0"), False
    if type(points).__module__.startswith("torch"):
        import torch

        if not points.is_cuda or points.dtype != torch.float32:
            raise ValueError("points must be a float32 CUDA tensor")
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")
        return wp.from_torch(points.contiguous(), dtype=wp.vec3), True
    raise TypeError("points must be a Warp array, NumPy array, or PyTorch tensor")


def _weights_array(weights, n, device):
    if weights is None:
        return wp.zeros(n, dtype=float, device=device)
    if isinstance(weights, wp.array):
        result = weights
    elif isinstance(weights, np.ndarray):
        if weights.dtype != np.float32:
            raise ValueError("weights must have dtype float32")
        result = wp.array(weights, device=device)
    elif type(weights).__module__.startswith("torch"):
        import torch

        if not weights.is_cuda or weights.dtype != torch.float32:
            raise ValueError("weights must be a float32 CUDA tensor")
        result = wp.from_torch(weights.contiguous())
    else:
        raise TypeError("weights must be a Warp array, NumPy array, or PyTorch tensor")
    if result.dtype != wp.float32 or result.size != n or result.device != device:
        raise ValueError("weights must contain N float32 values on the points device")
    if result.shape not in ((n,), (n, 1)):
        raise ValueError("weights must have shape (N,) or (N, 1)")
    if not result.is_contiguous:
        result = wp.clone(result)
    return result.reshape((n,))


def _hints_array(value, device):
    if isinstance(value, wp.array):
        result = value
    elif isinstance(value, np.ndarray):
        result = wp.array(value, device=device)
    elif type(value).__module__.startswith("torch"):
        result = wp.from_torch(value.contiguous())
    else:
        raise TypeError("initial guesses must be array objects")
    if result.dtype != wp.int32 or result.ndim != 1 or result.device != device:
        raise ValueError(
            "initial guesses must be one-dimensional int32 arrays on the points device"
        )
    return result


def _chunk_size(n, capacity, device, chunk_size, bytes_per_slot=44):
    if chunk_size is not None:
        chunk_size = operator.index(chunk_size)
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        return min(n, chunk_size)
    forced = os.environ.get("WARPOWER_NUM_CHUNKS", os.environ.get("PARAGRAM_NUM_CHUNKS", "0"))
    if int(forced) > 0:
        return max(1, (n + int(forced) - 1) // int(forced))
    fraction = float(
        os.environ.get("WARPOWER_MEM_FRAC", os.environ.get("PARAGRAM_MEM_FRAC", "0.5"))
    )
    if not 0 < fraction < 1:
        raise ValueError("memory fraction must lie between 0 and 1")
    budget = int(device.free_memory * fraction)
    per_cell = capacity * bytes_per_slot
    return max(32, min(n, (budget // per_cell // 32) * 32))


def _build(
    points,
    weights=None,
    *,
    knn=8,
    debug=False,
    initial_guesses=None,
    initial_guesses_offsets=None,
    leaf_size=-1,
    chunk_size=None,
    _capacity=64,
    _queue_capacity=96,
):
    started = time.perf_counter()
    p, torch_output = _points_array(points)
    if not p.device.is_cuda:
        raise ValueError("diagram construction requires a CUDA device")
    knn = operator.index(knn)
    if knn < 0:
        raise ValueError("knn must be nonnegative")
    leaf_size = operator.index(leaf_size)
    leaf_size = 4 if leaf_size == -1 else leaf_size
    if leaf_size < 1:
        raise ValueError("leaf_size must be positive or -1")
    n, device = len(p), p.device
    if torch_output:
        import torch

        stream = wp.stream_from_torch(torch.cuda.current_stream(points.device))
    else:
        stream = wp.get_stream(device)
    with wp.ScopedStream(stream):
        w = _weights_array(weights, n, device)
        use_hints = initial_guesses is not None
        if use_hints != (initial_guesses_offsets is not None):
            raise ValueError(
                "initial_guesses and initial_guesses_offsets must be provided together"
            )
        hints = (
            _hints_array(initial_guesses, device)
            if use_hints
            else wp.empty(0, dtype=int, device=device)
        )
        offsets = (
            _hints_array(initial_guesses_offsets, device)
            if use_hints
            else wp.empty(0, dtype=int, device=device)
        )
        if use_hints and len(offsets) != n + 1:
            raise ValueError("initial_guesses_offsets must have shape (N + 1,)")
        _validation.check(p, w, hints, offsets, use_hints)
        if n == 0:
            result = Diagram(
                wp.empty(0, dtype=int, device=device),
                wp.zeros(1, dtype=int, device=device),
                wp.empty(0, dtype=int, device=device),
            )
        else:
            hierarchy = _bvh.build(p, w, leaf_size)
            neighbors = wp.empty((n, _capacity), dtype=int, device=device)
            counts = wp.empty(n, dtype=int, device=device)
            status = wp.empty(n, dtype=int, device=device)
            retry_states = wp.empty(n, dtype=wp.vec3i, device=device)
            source = wp.full(n, -1, dtype=int, device=device)
            size = _chunk_size(n, _capacity, device, chunk_size)
            scratch = allocate_scratch(size, _capacity, device)
            query_capacity = max(1024, n // 16)
            queries = wp.empty(query_capacity, dtype=wp.vec4i, device=device)
            query_owners = wp.empty(query_capacity, dtype=int, device=device)
            query_count = wp.zeros(1, dtype=int, device=device)
            scratch.queries = queries
            scratch.query_owners = query_owners
            scratch.query_count = query_count
            kernel = make_construct_kernel(
                weights is not None, _queue_capacity, track_retries=False
            )
            for start in range(0, n, size):
                wp.launch(
                    kernel,
                    min(size, n - start),
                    inputs=[
                        p,
                        w,
                        hierarchy.tree,
                        hierarchy.tree.ids,
                        start,
                        False,
                        scratch,
                        neighbors,
                        counts,
                        status,
                        neighbors,
                        False,
                        hints,
                        offsets,
                        use_hints,
                        retry_states,
                    ],
                    block_dim=32,
                    device=device,
                )
            wp.launch(
                check_queries,
                query_capacity,
                inputs=[
                    p,
                    w,
                    hierarchy.tree,
                    queries,
                    query_owners,
                    query_count,
                    status,
                    neighbors,
                    counts,
                    source,
                    False,
                ],
                device=device,
            )
            failures = wp.empty(n, dtype=int, device=device)
            failure_count = wp.zeros(6, dtype=int, device=device)
            failure_size = failure_count[:1]
            wp.launch(
                collect_precision,
                n,
                inputs=[
                    status,
                    failures,
                    failure_count,
                    retry_states,
                    _capacity,
                    True,
                    _queue_capacity,
                ],
                device=device,
            )
            precision_size = min(n, 512)
            precision_jobs = wp.empty(precision_size, dtype=int, device=device)
            precision_scratch = allocate_precision_scratch(precision_size, 128, device, robust=True)
            wp.launch(
                make_precision_kernel(weights is not None, 128, True, 32),
                precision_size * 32,
                inputs=[
                    p,
                    w,
                    hierarchy.tree,
                    failures,
                    0,
                    failure_count,
                    source,
                    False,
                    True,
                    precision_scratch,
                    neighbors,
                    counts,
                    status,
                    neighbors,
                    False,
                    hints,
                    offsets,
                    use_hints,
                    retry_states,
                ],
                block_dim=32,
                device=device,
            )
            failure_size.zero_()
            wp.launch(
                _retry.collect_retries,
                n,
                inputs=[status, retry_states, failures, failure_count],
                device=device,
            )
            (
                failed,
                cooperative_attempts,
                capacity_remaining,
                precision_remaining,
                queue32_remaining,
                queue64_remaining,
            ) = map(int, failure_count.numpy())
            # The scalar transfer has synchronized construction. Release large
            # temporary buffers before allocating retry storage.
            del scratch
            hierarchy.temporaries = ()
            retry_neighbors = wp.empty((0, 1), dtype=int, device=device)
            retried = 0
            robust_retried = 0
            if failed:
                # Storage and precision have independent per-site requirements.
                # Completed rows and established storage floors are retained.
                retried = failed
                retry_neighbors = wp.empty((failed, 512), dtype=int, device=device)
                retry_counts = wp.empty(failed, dtype=int, device=device)
                retry_status = wp.empty(failed, dtype=int, device=device)
                wp.launch(
                    init_retries,
                    failed,
                    inputs=[failures, neighbors, status, retry_neighbors, retry_status, source],
                    device=device,
                )
                retry_counts.zero_()
                remaining = failed
                for capacity, queue, robust in (
                    (capacity, queue, robust)
                    for robust in (False, True)
                    for queue in (128, 256, 512)
                    for capacity in (128, 256, 512)
                ):
                    if not robust and capacity_remaining == 0:
                        continue
                    if robust and precision_remaining == 0:
                        continue
                    if queue > 128 and (queue64_remaining if robust else queue32_remaining) == 0:
                        continue
                    if robust and robust_retried == 0:
                        robust_retried = precision_remaining
                    retry_size = _chunk_size(
                        failed, capacity, device, chunk_size, bytes_per_slot=100 if robust else 56
                    )
                    if not robust:
                        query_count.zero_()
                        retry_scratch = allocate_precision_scratch(
                            retry_size, capacity, device, robust=False
                        )
                        retry_job_count = wp.array([failed], dtype=int, device=device)
                        kernel = make_precision_kernel(
                            weights is not None, queue, False, 32, capacity
                        )
                        for start in range(0, failed, retry_size):
                            wp.launch(
                                kernel,
                                min(retry_size, failed - start) * 32,
                                inputs=[
                                    p,
                                    w,
                                    hierarchy.tree,
                                    failures,
                                    start,
                                    retry_job_count,
                                    source,
                                    True,
                                    True,
                                    retry_scratch,
                                    retry_neighbors,
                                    retry_counts,
                                    retry_status,
                                    retry_neighbors,
                                    False,
                                    hints,
                                    offsets,
                                    use_hints,
                                    retry_states,
                                ],
                                block_dim=32,
                                device=device,
                            )
                    else:
                        retry_scratch = allocate_scratch(retry_size, capacity, device, robust=True)
                        retry_stride = 32
                        kernel = make_construct_kernel(
                            weights is not None, queue, robust=robust, lane_stride=retry_stride
                        )
                        for start in range(0, failed, retry_size):
                            wp.launch(
                                kernel,
                                min(retry_size, failed - start) * retry_stride,
                                inputs=[
                                    p,
                                    w,
                                    hierarchy.tree,
                                    failures,
                                    start,
                                    True,
                                    retry_scratch,
                                    retry_neighbors,
                                    retry_counts,
                                    retry_status,
                                    retry_neighbors,
                                    True,
                                    hints,
                                    offsets,
                                    use_hints,
                                    retry_states,
                                ],
                                block_dim=32,
                                device=device,
                            )
                    if not robust:
                        wp.launch(
                            check_queries,
                            query_capacity,
                            inputs=[
                                p,
                                w,
                                hierarchy.tree,
                                queries,
                                query_owners,
                                query_count,
                                retry_status,
                                retry_neighbors,
                                retry_counts,
                                source,
                                True,
                            ],
                            device=device,
                        )
                        failure_size.zero_()
                        wp.launch(
                            collect_precision_retry,
                            failed,
                            inputs=[
                                retry_status,
                                failures,
                                precision_jobs,
                                failure_count,
                                retry_states,
                            ],
                            device=device,
                        )
                        wp.launch(
                            make_precision_kernel(weights is not None, 128, True, 32),
                            precision_size * 32,
                            inputs=[
                                p,
                                w,
                                hierarchy.tree,
                                precision_jobs,
                                0,
                                failure_count,
                                source,
                                True,
                                True,
                                precision_scratch,
                                retry_neighbors,
                                retry_counts,
                                retry_status,
                                retry_neighbors,
                                False,
                                hints,
                                offsets,
                                use_hints,
                                retry_states,
                            ],
                            block_dim=32,
                            device=device,
                        )
                    wp.launch(
                        merge_retry,
                        failed,
                        inputs=[failures, retry_counts, retry_status, counts, status, source, 0],
                        device=device,
                    )
                    # Only a scalar diagnostic is copied, rather than all statuses.
                    check = wp.zeros(6, dtype=int, device=device)
                    wp.launch(
                        retry_kinds,
                        failed,
                        inputs=[retry_status, check, failure_count, failures, retry_states],
                        device=device,
                    )
                    (
                        remaining,
                        capacity_remaining,
                        cooperative_attempts,
                        precision_remaining,
                        queue32_remaining,
                        queue64_remaining,
                    ) = map(int, check.numpy())
                    if remaining == 0:
                        break
            # A one-sided facet can survive single-precision clipping without
            # an explicit topology failure. Recompute both endpoints in float64
            # before making the output symmetric. New rows append to the retry
            # buffer so existing source indices remain valid.
            consistency_retried = 0
            repair_jobs, repair_count = asymmetric_cells(neighbors, retry_neighbors, source, counts)
            for _ in range(3):
                if repair_count == 0:
                    break
                first = len(retry_neighbors)
                total = first + repair_count
                jobs = wp.empty(total, dtype=int, device=device)
                wp.copy(jobs, repair_jobs, dest_offset=first, count=repair_count)
                repaired = wp.empty((total, 512), dtype=int, device=device)
                if first:
                    wp.copy(repaired, retry_neighbors, count=first * 512)
                wp.launch(
                    gather_repair_seeds,
                    (repair_count, 512),
                    inputs=[jobs, first, counts, neighbors, retry_neighbors, source, repaired],
                    device=device,
                )
                repair_counts = wp.empty(total, dtype=int, device=device)
                repair_status = wp.full(total, 1, dtype=int, device=device)
                wp.launch(
                    _retry.prepare_repairs,
                    repair_count,
                    inputs=[repair_jobs, retry_states],
                    device=device,
                )
                repair_counts.zero_()
                consistency_retried += repair_count
                for capacity in (128, 256, 512):
                    repair_size = _chunk_size(
                        repair_count, capacity, device, chunk_size, bytes_per_slot=100
                    )
                    repair_scratch = allocate_scratch(repair_size, capacity, device, robust=True)
                    kernel = make_construct_kernel(
                        weights is not None, capacity, robust=True, lane_stride=32
                    )
                    for start in range(0, repair_count, repair_size):
                        wp.launch(
                            kernel,
                            min(repair_size, repair_count - start) * 32,
                            inputs=[
                                p,
                                w,
                                hierarchy.tree,
                                jobs,
                                first + start,
                                True,
                                repair_scratch,
                                repaired,
                                repair_counts,
                                repair_status,
                                repaired,
                                True,
                                hints,
                                offsets,
                                use_hints,
                                retry_states,
                            ],
                            block_dim=32,
                            device=device,
                        )
                    check = wp.zeros(1, dtype=int, device=device)
                    tmp = wp.empty(repair_count, dtype=int, device=device)
                    wp.launch(
                        collect_failures,
                        repair_count,
                        inputs=[repair_status[first:], tmp, check],
                        device=device,
                    )
                    if int(check.numpy()[0]) == 0:
                        break
                wp.launch(
                    merge_retry,
                    repair_count,
                    inputs=[jobs, repair_counts, repair_status, counts, status, source, first],
                    device=device,
                )
                retry_neighbors = repaired
                repair_jobs, repair_count = asymmetric_cells(
                    neighbors, retry_neighbors, source, counts
                )
            adjacency, final_offsets = compact(
                neighbors, retry_neighbors, source, counts, already_symmetric=repair_count == 0
            )
            # Scratch and queue work complete before their Python owners leave scope.
            wp.synchronize_stream(stream)
            diagnostics = {
                "retried_cells": retried,
                "double_precision_cells": robust_retried,
                # Each site gets at most one bounded cooperative attempt. The
                # other counters describe the later host-dispatched stages.
                "cooperative_precision_attempts": cooperative_attempts,
                "consistency_retried_cells": consistency_retried,
                "asymmetric_cells_remaining": repair_count,
                "chunk_size": size,
                "leaf_size": leaf_size,
                "knn_requested": knn,
                "knn_effective": 0,
                "warm_start": "CSR hints" if use_hints else "nearest leaf",
            }
            result = Diagram(adjacency, final_offsets, status, diagnostics)
    if torch_output:
        result = Diagram(
            wp.to_torch(result.adjacency),
            wp.to_torch(result.offsets),
            wp.to_torch(result.status),
            result.diagnostics,
        )
    if debug:
        print(f"Warpower: {n} sites, {time.perf_counter() - started:.6f}s, {result.diagnostics}")
    return result


def _with_tetrahedra(result, points, weights, kwargs):
    # Keep all imports, allocations, and GPU work for meshing behind the opt-in.
    from ._tetrahedra import build

    p, torch_output = _points_array(points)
    if torch_output:
        import torch

        stream = wp.stream_from_torch(torch.cuda.current_stream(points.device))
    else:
        stream = wp.get_stream(p.device)
    with wp.ScopedStream(stream):
        w = _weights_array(weights, len(p), p.device)
        leaf_size = kwargs.get("leaf_size", -1)
        tetrahedra, extra = build(
            p, w, chunk_size=kwargs.get("chunk_size"), leaf_size=4 if leaf_size == -1 else leaf_size
        )
        wp.synchronize_stream(stream)
        if torch_output:
            tetrahedra = wp.to_torch(tetrahedra)
    return Diagram(
        result.adjacency, result.offsets, result.status, {**result.diagnostics, **extra}, tetrahedra
    )


def voronoi_diagram(points, *, return_tetrahedra=False, **kwargs):
    if not isinstance(return_tetrahedra, (bool, np.bool_)):
        raise TypeError("return_tetrahedra must be a boolean")
    result = _build(points, **kwargs)
    if return_tetrahedra:
        result = _with_tetrahedra(result, points, None, kwargs)
    return result


def power_diagram(points, weights, *, return_tetrahedra=False, **kwargs):
    if weights is None:
        raise TypeError("power_diagram requires weights")
    if not isinstance(return_tetrahedra, (bool, np.bool_)):
        raise TypeError("return_tetrahedra must be a boolean")
    result = _build(points, weights, **kwargs)
    if return_tetrahedra:
        result = _with_tetrahedra(result, points, weights, kwargs)
    return result
