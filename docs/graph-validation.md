# CUDA graph validation and performance

Validated on 2026-11-06 using the NVIDIA L40 48 GB, Warp 1.19.0+cu12,
Python 3.10.12, NumPy 2.2.6, SciPy 1.15.3, and PyTorch 2.7.1+cu126.
The [workspace guide](graphs.md) defines the capture, capacity, and stream contract.

## Device-resident replay

A captured pipeline produces new positions and weights, builds the bounded power
diagram and full-hull regular mesh, consumes tetrahedra to compute volume, and
advances a device counter. Three warmed replays were profiled with Nsight Systems.
The captured graph contains 174 kernel nodes, 38 memset nodes, 27 conditional
nodes, and five **device-to-device** copy nodes. It contains no host-copy,
host-callback, allocation, or free nodes. Executed branches produced 360 kernel
activities and 18 memset activities across the three replays.

The recorded CPU CUDA calls were three graph launches, the profiler start, a
capture-state query, and one synchronization at the end of the measurement
region. Construction itself performs no host readback or synchronization.
Nsight omitted activity records for the graph's device-to-device copies, so copy
directions were verified from CUDA's verbose DOT graph inspection rather than
inferred from missing profiler records. The [sanitized audit](../results/validation/graphs/device-audit.json)
retains counts, API names, and hashes of the inspected local artifacts.

From the repository root, with development dependencies, PyTorch, and Nsight
Systems installed:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 nsys profile \
  --trace=cuda --cuda-graph-trace=node --cuda-event-trace=false \
  --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi \
  --capture-range-end=stop --output=results/local/device-loop \
  python -m benchmarks.graph_capture --dot results/local/device-loop.dot
nsys export --type sqlite --output results/local/device-loop.sqlite \
  results/local/device-loop.nsys-rep
python -m benchmarks.graph_capture --dot results/local/device-loop.dot \
  --analyze results/local/device-loop.sqlite --output results/local/device-audit.json
```

## Replay speed

Both implementations receive the same resident CUDA inputs. Measurements use
three warmups, alternating paired execution order, end-to-end wall time with
synchronization, and 32 GiB of Warp memory-pool retention. There are 20 pairs per
adjacency case and five per tetrahedron case. No measured case was CPU-quota
throttled. Setup, allocation, compilation, and capture are excluded from replay
timing; these costs are paid once when creating a workspace.

| Output / input | Eager mean (ms) | Graph mean (ms) | Speedup |
|---|---:|---:|---:|
| Voronoi adjacency, 100,000 uniform | 12.680 | 11.072 | 1.15× |
| Voronoi adjacency, 100,000 five-cluster | 29.694 | 27.569 | 1.08× |
| Voronoi adjacency, 1,000,000 uniform | 102.293 | 100.168 | 1.02× |
| Power adjacency, 1,000,000 uniform | 115.047 | 113.276 | 1.02× |
| Adjacency + Delaunay, 1,000 uniform | 127.461 | 11.118 | 11.46× |
| Adjacency + Delaunay, 10,000 uniform | 250.100 | 42.975 | 5.82× |
| Adjacency + Delaunay, 100,000 uniform | 2,442.804 | 487.385 | 5.01× |
| Adjacency + regular mesh, 100,000 uniform | 2,469.018 | 492.210 | 5.02× |

Each graph output matched the eager API's complete CSR rows and tetrahedron set.
At 100,000 sites this includes 671,908 unweighted tetrahedra and 667,413 weighted
tetrahedra. Those same seeded inputs also have the
[independent SciPy comparisons](tetrahedra-validation.md#independent-full-mesh-comparisons).

The tetrahedron timings are **warmed replay**: the workspace retains a successful
domain scale, reconstructs and validates each mesh, and expands again if needed.
The eager API starts its domain search afresh on every call. This reuse accounts
for much of the speedup. These timing inputs are fixed; changing-topology replay
is tested separately. These results do not claim a first-call speedup or a speedup
for every possible input distribution.

[Raw samples and measured source hashes](../results/validation/graphs/performance.json)
are reproducible with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python -m benchmarks.graphs \
  --output results/local/graph-performance.json
```

## Existing API performance

The eager `_build` AST and all its existing kernel modules are unchanged from
`6569d3887eaab8965b97a6f38a4886a869ca4564`. A lazy package export makes the new class
available without loading graph code during ordinary calls. Thirty paired
repetitions on the four adjacency cases above differed by -0.083% to +0.083%, with
identical GPU launch/allocation traces and no CPU throttling. The new module
remained unloaded throughout that separate check.

[Raw default-path results](../results/validation/graphs/default-performance.json)
can be reproduced with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python -m benchmarks.default_path \
  --baseline-ref 6569d3887eaab8965b97a6f38a4886a869ca4564 \
  --output results/local/graph-default-performance.json
```

## Correctness and memory safety

The [full suite](../results/validation/graphs/pytest.txt) passes **85 tests**.
The 15 new graph tests cover changing input and topology within captured
producer/construction/consumer pipelines, a device-controlled outer loop,
overflowed CSR totals, full independent weighted and
unweighted references, volume and face checks, fixed-capacity overflow, invalid
input followed by successful replay, duplicate ownership, hidden weighted sites,
far-away dual vertices, domain reuse, small inputs, multi-level prefix scans,
adjacency recovery, and PyTorch operations on a shared captured stream.
Tests also intercept host array reads, synchronizations, and Warp allocation
calls during capture and replay and fail if construction attempts any of them.

A built wheel also passed a captured CUDA smoke test outside the checkout,
including lazy import and full-hull tetrahedra. Every packaged source file matched
the hashes recorded by the performance run.

[CUDA memcheck](../results/validation/graphs/memcheck.txt) passes all 15 graph tests
with **zero errors and no kernel exclusions**:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python -m pytest -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 compute-sanitizer \
  --tool memcheck --error-exitcode 99 python -m pytest tests/test_graph.py -q
```

The workspace preserves the eager mesh's floating-point limitations. It uses
device error flags for unresolved degeneracy or capacity exhaustion and publishes
zero counts, rather than silently accepting a partial mesh. Persistent reserves
also increase resident memory: for adjacency, the 512-entry recovery row alone
requires 2.048 GB per million sites, in addition to ordinary rows, hierarchy,
output, and clipping scratch. `chunk_size` controls clipping scratch; reducing it
can trade throughput for memory. The workspace guide documents fixed input
shapes and the need for separate workspaces for concurrent executions.
