# VTM development validation

## Standalone geometry baseline

The first VTM preparation change adds lazy standalone `delaunay_tetrahedra` and
`regular_tetrahedra` functions. Every existing geometry implementation file is
byte-identical to commit `3091289464af368d2a61f843d169d7f4863d1cfd`; only the package's
lazy exports and the new wrapper module differ. No VTM optimization runs yet.

A paired baseline/current run on the NVIDIA L40 covers four eager diagram cases
(up to one million sites), two graph diagram cases, and four full-mesh graph
cases at 100,000 sites. Each output matches completely. Eager GPU launch and
allocation traces are identical. No case has CPU quota throttling or a measured
significant slowdown. Raw samples, paired confidence intervals, source hashes,
and workspace allocator deltas are in
[the baseline report](../results/validation/vtm/geometry-baseline.json).

| Captured output, 100,000 uniform sites | Baseline mean (ms) | Current mean (ms) |
|---|---:|---:|
| Delaunay tetrahedra only | 472.965 | 472.932 |
| Regular tetrahedra only | 477.439 | 477.438 |
| Voronoi adjacency and Delaunay tetrahedra | 486.952 | 487.014 |
| Power adjacency and regular tetrahedra | 492.100 | 492.291 |

Measurements use three warmups, twenty alternating paired repetitions for
adjacency, ten for tetrahedra, resident CUDA inputs, synchronized wall time, and
32 GiB allocator retention. Graph timings exclude setup/capture and reuse the
successful domain scale, while reconstructing and validating every mesh.
Tetrahedra-only execution saves the independent bounded-adjacency work; these
results do not establish a VTM speedup. Weighted adjacency showed a small timing
improvement despite identical implementation; it is not claimed as an optimization.

The regression harness imports the baseline checkout under a separate package
name in the same process. Persistent allocator deltas include capture runtime
storage; the current graph captures measured about 72 KB more than the baseline
captures despite identical workspace code. These deltas should not be confused
with a change in explicit geometry-buffer capacities.

The new tests compare standalone weighted/unweighted tetrahedra with independent
SciPy references, verify positive volume and hull coverage, ensure bounded diagram
construction is skipped, check lazy imports, invalid and small inputs, ambiguity
failure, and PyTorch stream interoperability. Reproduction commands are in the
[VTM development plan](vtm.md).

The complete suite passes **94 tests**, including the nine new standalone API
tests ([test output](../results/validation/vtm/geometry-tests.txt)).
A built wheel also passes a CUDA standalone-mesh smoke test from outside the
checkout; imports remain lazy and installed source files match the measured tree.
