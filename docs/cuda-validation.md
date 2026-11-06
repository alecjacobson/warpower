# CUDA validation environment

Tests run on the NVIDIA L40 with Warp 1.19.0 and CUDA driver 570.158.01.
Ordinary execution and performance measurements use the default CUDA context
settings. The geometry runtime contains no native CUDA extension or driver calls.

The 300-site torus integration test exposed an instrumented nested-graph failure
tracked in [#52](https://github.com/alecjacobson/warpower/issues/52). With the default
context stack allowance, CUDA memcheck 12.6 and 12.9 report an illegal address;
12.9 reports a hardware exception at instruction address zero. Submitting the
same 16 optimization steps separately passes. Disabling the memory pool or
synchronizing before capture does not resolve the full-loop failure.

Setting the diagnostic process's CUDA stack allowance to **32 KiB** resolves it
without changes to geometry, graph structure, or buffers. The complete set of
31 boundary/core/extraction tests then passes memcheck, including 32-byte
allocation padding, with zero errors and no kernel/test exclusions. These
observations indicate stack pressure during instrumented nested execution;
they do not identify a particular NVIDIA internal call frame. The separate
[upstream allocation-tracking report](https://github.com/NVIDIA/warp/issues/1406)
has a similar symptom, but its suggested pre-capture synchronization did not
resolve this reproducer, so they are not assumed to have the same cause.

Enabling `--track-stream-ordered-races all` reveals a second diagnostic issue:
it reports writes to persistent workspace bounds as occurring before allocation.
The same report reproduces in `benchmarks.nested_capture`, which imports no
Warpower code and only increments an integer through nested Warp loops. All
arrays are retained and explicitly synchronized before capture. With the enlarged
stack, ordinary memcheck passes this minimal program; allocation-order tracking
still fails with asynchronous pooled allocation. Switching the **diagnostic
process** to synchronous allocations makes the tracking run pass. This isolates
the report from the geometry implementation and records a checker/runtime
limitation rather than an ignored geometry error. The upstream allocation issue
is relevant, but pre-capture synchronization alone does not fix this nested case.

Use the diagnostic-only launcher to reproduce the instrumented test setting:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
  /usr/local/cuda-12.9/bin/compute-sanitizer --tool memcheck --error-exitcode 99 \
  --padding 32 python -m benchmarks.memcheck --stack-bytes 32768 -- -q \
  tests/test_boundary.py tests/test_vtm_core.py tests/test_extraction.py

# Additional lifetime checking, with synchronous diagnostic allocations:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
  /usr/local/cuda-12.9/bin/compute-sanitizer --tool memcheck --error-exitcode 99 \
  --padding 32 --track-stream-ordered-races all \
  python -m benchmarks.memcheck --stack-bytes 32768 --disable-mempool -- -q \
  tests/test_boundary.py tests/test_vtm_core.py tests/test_extraction.py
```

`benchmarks.memcheck` uses the public CUDA driver API only to configure and record
this **test-process** stack limit before running pytest. The optional pool flag
uses Warp's allocator configuration API. It is never imported by
`warpower` and changes no library or replay behavior. Native DOT/Nsight audits
separately check real replay for host copies, callbacks, synchronization, and
allocation/free nodes. Correctness tests also compare geometry against
independent CPU references. Passing memory checks alone is not a proof of
numerical or topological correctness.

The minimal reproducer can be run directly or under the checker; compare its
default asynchronous allocation with `--disable-mempool` when lifetime tracking
is enabled:

```bash
python -m benchmarks.nested_capture
compute-sanitizer --tool memcheck --track-stream-ordered-races all \
  python -m benchmarks.nested_capture --stack-bytes 32768
compute-sanitizer --tool memcheck --track-stream-ordered-races all \
  python -m benchmarks.nested_capture --stack-bytes 32768 --disable-mempool
```

The middle command reproduces the allocation-tracking report in the recorded
environment. These controls are validation settings, not required production
configuration, and no checker result here proves arbitrary pooled-memory
lifetime correctness.
