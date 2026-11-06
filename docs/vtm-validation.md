# VTM development validation

## Standalone geometry baseline

The first VTM preparation change (#41) added lazy standalone `delaunay_tetrahedra`
and `regular_tetrahedra` functions. At that milestone, every existing geometry
implementation file was byte-identical to commit
`3091289464af368d2a61f843d169d7f4863d1cfd`; only the package's lazy exports and the
new wrapper module differed. Later milestones and their evidence appear below.

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

## Fixed-boundary ODT foundation

The uniform interior optimizer is implemented in the optional
[`ODTWorkspace`](odt.md). The complete suite passes **104 tests**
([output](../results/validation/vtm/odt-tests.txt)); all ten optimizer tests also
pass CUDA memcheck with **zero errors and no exclusions**
([output](../results/validation/vtm/odt-memcheck.txt)). Checks include independent
quadratic quadrature and circumcenter linear solves, an energy finite-difference
check, a nearly flat tetrahedron, changing topology, fixed hull/volume, energy
backtracking, and complete rollback after injected reconstruction failure.

A captured outer device loop produces new sites, optimizes them, and consumes
tetrahedra without host readback, allocation, or synchronization. The
[Nsight/DOT audit](../results/validation/vtm/odt-device-audit.json) records 146 kernel,
55 memset, 16 conditional, and 12 device-to-device copy nodes. There are no host
copy/callback/allocation/free nodes. Three profiled graph replays perform three
optimizer iterations each; CPU CUDA calls consist of graph submission, profiler
start, a capture-state query, and one final synchronization outside construction.

The initial trajectory measurements are deliberately reported separately from
full-mesher performance:

| Uniform sites | Accepted/requested steps | Energy decrease | Mean warmed run | Result |
|---|---:|---:|---:|---|
| 1,000 | 20/20 | 47.72% | 0.197 s | Iteration budget reached |
| 10,000 | 20/20 | 52.05% | 2.849 s | Iteration budget reached |
| 100,000 | 6/20 | 48.44% | 95.57 s, diagnostic only | Cell-construction failure; valid previous mesh restored |

[Raw records](../results/validation/vtm/odt-core.json) include source hashes,
energy histories, quality, rejection counts, and CPU quota counters. The 100k run
experienced CPU throttling and is **not a clean timing result**. It also did not
complete the requested iteration budget. Numerical recovery and trajectory costs
are open work in [#32](https://github.com/alecjacobson/warpower/issues/32) and
[#43](https://github.com/alecjacobson/warpower/issues/43). No CPU speedup is claimed.

All runs retain the original hull and volume. Minimum radius ratios remain
extremely poor (around 1e-13), and can worsen as sites approach the fixed boundary.
These measurements establish the interior energy/reconstruction loop, not a
high-quality boundary-fitted mesher. The later boundary and quality stages are
required before claiming the full VTM result.

After 20 iterations, complete optimized meshes at 1k and 10k sites match SciPy's
independent Delaunay result exactly (5,934 and 61,244 tetrahedra respectively;
[comparisons](../results/validation/vtm/odt-reference-comparisons.json)).
The same optimized meshes also match CGAL 6.2.1's complete tetrahedron sets
([native reference records](../results/validation/vtm/odt-cgal-comparisons.json),
[reference source and instructions](../benchmarks/cpu/README.md)). CGAL also
constructs the saved failing 100k candidate: the GPU failure is isolated to site
34817, whose final star has 305 tets and 157 neighbors. Its clipping status is a
vertex-capacity overflow, tracked in
[#44](https://github.com/alecjacobson/warpower/issues/44).

Generate an instrumented failure snapshot (when the failure still reproduces):

```bash
python -m benchmarks.odt_failure --output results/local/odt-failure.npz
```

The sidecar JSON says whether a failed candidate was captured. The `points` and
`status` arrays are present only on capture; `last_valid_points` always records
the optimizer's returned state. This diagnostic uses private mesh state and is
not part of the production API.

Reproduce the device audit with:

```bash
nsys profile --trace=cuda --cuda-graph-trace=node --cuda-event-trace=false \
  --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi \
  --capture-range-end=stop --output=results/local/odt-loop \
  python -m benchmarks.graph_capture --optimizer --dot results/local/odt-loop.dot
nsys export --type sqlite --output results/local/odt-loop.sqlite results/local/odt-loop.nsys-rep
python -m benchmarks.graph_capture --optimizer --dot results/local/odt-loop.dot \
  --analyze results/local/odt-loop.sqlite --output results/local/odt-audit.json
```

The built wheel passes a captured CUDA ODT smoke test from outside the checkout,
including lazy import, energy decrease, and exact packaged-source matching.
At the initial optimizer milestone (#45), all pre-existing geometry implementation
files were byte-identical to the pre-VTM revision ([source audit](../results/validation/vtm/odt-standalone-isolation.json));
only optional modules and lazy exports were added. The standalone performance
baseline above still describes those unchanged execution paths. The new optimizer
has its own costs and does not run during standalone geometry calls.

## Larger intermediate-cell recovery

`DiagramWorkspace(..., tetrahedra_cell_capacity=1024)` adds an optional recovery
stage while retaining the default 512-cell path. `ODTWorkspace` opts into the
larger reserve by default. The 10-bit packed plane indices bound the supported
maximum at 1024; no IDs are wrapped. Scratch allocation doubles for the opt-in
at a fixed retry chunk, and can be reduced with `chunk_size`.

The saved failing 100k input now produces all **626,008 tetrahedra**, exactly
matching CGAL 6.2.1
([full comparison](../results/validation/vtm/large-cells/full-mesh-recovery.json)).
A compact [401-site fixture](../tests/data/intermediate-cell-capacity.json)
preserves the clipping order and box: it reaches 512 distinct intermediate
vertices, then finishes with only 312 when given the larger capacity. All 305
finite dual tetrahedra incident to its seed match the independent CGAL star.
Captured tests force the larger recovery branch, including odd-sized chunks and
weighted/unweighted geometry.

The 100k optimizer now completes **20/20 iterations without rejection**, reducing
energy by **50.73%** and returning 610,832 tetrahedra with the original hull and
volume ([trajectory record](../results/validation/vtm/large-cells/trajectory.json)).
Its measured 58.06-second warm run encountered CPU throttling and remains a
diagnostic measurement, not a clean performance claim. Worst boundary quality
is still poor; boundary fitting and reconstruction speed remain open milestones.

The full suite passes 111 tests. The 17 optimizer and large-cell tests pass CUDA
memcheck without errors. All ten standalone paired performance cases have equal
outputs and no statistically significant regression above 2%, with no CPU
throttling ([measurements](../results/validation/vtm/large-cell-default-performance.json)).

## Captured surface-domain services

The domain milestone supplies `SurfaceDomain` and `DomainQueryWorkspace`, with
pure-Warp triangle hierarchy refitting, float64 closest-point and oriented-ray
queries, surface quadrature, crease/corner detection, and device input validation.
It does not yet perform boundary-fitted tetrahedral meshing.

The full suite passes **133 tests**. All **22 domain tests** pass CUDA memcheck
with zero errors. Independent references include brute-force triangle-distance
minima and all-triangle float64 solid-angle sums, alongside analytic sphere,
torus, box, cavity, and disconnected-component checks. Degree-two surface
integrals and crease integrals agree with closed-form values; tests cover sharp
corners, illegal shared-vertex/edge contact, nonmanifold vertex links, reversed
orientation, invalid indices, nonfinite inputs, and recovery after a bad refit.
Thin-triangle and `2e-7`-thick box cases exercise float64 geometric calculations.

The first native audit found allocation/free nodes inside `Warp.Mesh.refit()`;
[#47](https://github.com/alecjacobson/warpower/issues/47) records that finding.
The final implementation uses Warp-owned hierarchy arrays instead. Its
[replay audit](../results/validation/vtm/domain-device-audit.json) contains **33
kernel nodes, two memsets, and two device-to-device copies**, with no host,
allocation, or free nodes. Three changing-input replays produce 99 kernel
activities and six observed device-to-device copies. Runtime calls are three
graph launches, profiler/capture bookkeeping, and one final synchronization
outside the geometry operation. A test now inspects native DOT node types in
addition to forbidding Python allocation/readback/synchronization calls.

On NVIDIA L40, 100,000 signed queries against a 9,216-triangle torus take a median
**51.76 ms**. Deforming and refitting that surface, validating it, updating 110,592
surface samples, querying, and consuming the results takes **62.83 ms**. Neither
20-sample measurement encountered CPU throttling
([raw measurements](../results/validation/vtm/domain-performance.json)). These
are domain-service timings, not a full-mesher speedup result. Query performance
remains tracked in [#48](https://github.com/alecjacobson/warpower/issues/48).

All pre-existing geometry and ODT implementation modules are byte-identical to
revision `9692137`; package exports remain lazy
([isolation audit](../results/validation/vtm/domain-standalone-isolation.json)).
The standalone paired performance evidence from the preceding milestone still
applies to those unchanged execution paths.
An installed wheel passes a captured domain-refit/query smoke test outside the
checkout, with lazy imports and exact installed-source matching
([package record](../results/validation/vtm/domain-wheel.json)).

Reproduce the domain measurement and replay audit:

```bash
python -m benchmarks.domain --output results/local/domain-performance.json
nsys profile --trace=cuda --cuda-graph-trace=node --cuda-event-trace=false \
  --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi \
  --capture-range-end=stop --output=results/local/domain-loop \
  python -m benchmarks.domain --queries 1000 --repeats 1 --profile \
  --dot results/local/domain-loop.dot
nsys export --type sqlite --output results/local/domain-loop.sqlite results/local/domain-loop.nsys-rep
python -m benchmarks.graph_capture --domain --dot results/local/domain-loop.dot \
  --analyze results/local/domain-loop.sqlite --output results/local/domain-audit.json
```

## Boundary fitting and interior integration

The independent [`BoundaryWorkspace`](boundary.md) and internal uniform VTM core
pass **149 total tests** ([output](../results/validation/vtm/boundary-tests.txt)).
Their 16 tests pass CUDA memcheck with **zero errors**
([output](../results/validation/vtm/boundary-memcheck.txt)). Independent checks cover
nearest-site assignments, nonuniform weighted moments, corner conflicts, evolving
roles, device active counts, invalid-input recovery, and complete position and
connectivity rollback after injected reconstruction failures.

| Example | Sites | Accepted steps | Median resident loop | Boundary objective reduction |
|---|---:|---:|---:|---:|
| Torus | 1,000 | 20/20 | 313.35 ms | 66.33% surface error |
| Cube with boundary initialization | 200 | 12/12 | 51.87 ms | 63.67% crease error |

These NVIDIA L40 measurements include a changing-input producer, optimization,
and full-hull volume consumer. The cube also includes one complete boundary
initialization pass and preserves its eight corners exactly. Five warmed timing
samples per case have no CPU quota throttling
([torus](../results/validation/vtm/boundary-torus.json),
[cube](../results/validation/vtm/boundary-cube.json)). The torus uses 110,592
surface quadrature samples. Both final full tetrahedron sets match SciPy and
native CGAL 6.2.1 exactly
([CGAL check](../results/validation/vtm/boundary-cgal.json)).

These are intermediate pipeline measurements, not CPU speedup or final mesh
quality claims. Full-hull output fills nonconvex holes and contains very poor
boundary elements; extraction and worst-element improvements are still required.
Corner-only cube initialization exposes an unresolved cospherical-facet case
in [#32](https://github.com/alecjacobson/warpower/issues/32); complete boundary
initialization avoids that case without claiming a general numerical fix.

The [native replay audit](../results/validation/vtm/boundary-device-audit.json)
has 316 kernel, 109 memset, 28 conditional, and 24 device-to-device copy nodes.
It has no allocation/free nodes, host callbacks, or host transfers. Three profiled
replays execute 1,848 kernel activities, including device corner-matching and
optimization loops. A top-level native radix sort originally added graph memory
nodes ([#50](https://github.com/alecjacobson/warpower/issues/50)); the site hierarchy
now rebuilds inside the device input-validity conditional, matching the existing
diagram workspace's capture pattern.

All prior geometry, ODT, and domain implementation modules remain byte-identical
to `df60928` ([source audit](../results/validation/vtm/boundary-standalone-isolation.json)).
The prior standalone performance gate therefore still covers their unchanged
execution paths. An installed wheel passes captured boundary/core smoke checks
outside the source checkout, with exact source matching
([package record](../results/validation/vtm/boundary-wheel.json)).

```bash
python -m benchmarks.vtm_core --case torus --n 1000 --iterations 20 --reference \
  --output results/local/boundary-torus.json
python -m benchmarks.vtm_core --case cube --n 200 --iterations 12 --reference \
  --output results/local/boundary-cube.json
nsys profile --trace=cuda --cuda-graph-trace=node --cuda-event-trace=false \
  --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi \
  --capture-range-end=stop --output=results/local/boundary-loop \
  python -m benchmarks.vtm_core --case cube --n 200 --iterations 3 --repeats 1 \
  --profile --dot results/local/boundary-loop.dot
nsys export --type sqlite --output results/local/boundary-loop.sqlite results/local/boundary-loop.nsys-rep
python -m benchmarks.graph_capture --vtm-core --dot results/local/boundary-loop.dot \
  --analyze results/local/boundary-loop.sqlite --output results/local/boundary-audit.json
```

## Domain extraction and topology diagnostics

[`ExtractionWorkspace`](extraction.md) applies the paper's strict outside-distance
to circumradius threshold of 0.4, then compacts selected tetrahedra and exposes
their oriented boundary. The complete suite passes **164 tests**
([output](../results/validation/vtm/extraction-tests.txt)). Fifteen new tests cover
independent circumcenter solves, all-triangle signed winding and closest-distance
oracles, strict threshold behavior, volume and quality formulas, surface-error
quadrature, invalid inputs, changing device counts, and captured consumers.

The extracted cube has exact volume and surface area. An enclosed cavity has
one volume component, two boundary shells, and the correct negative inner-shell
volume. Disconnected cubes retain two components without a convex-hull bridge.
The fitted torus has one closed boundary with Euler characteristic zero. Tests
also detect duplicate tetrahedra, nonmanifold edge contacts, and disconnected
vertex links. These checks do not guarantee topology at insufficient resolution.

All **31 boundary/core/extraction tests** pass CUDA memcheck with 32-byte padding,
zero errors, and no exclusions
([output](../results/validation/vtm/extraction-memcheck.txt)). Instrumenting the
300-site, 16-step nested optimizer requires a 32 KiB CUDA stack allowance in the
test process; the default setting fails before extraction. Individual captured
steps pass without that setting. The [validation environment](cuda-validation.md)
records these observations and the diagnostic-only launcher. No runtime geometry
or default context setting was changed.

A second 31-test run also passes with explicit allocation-order checking and
synchronous diagnostic allocations
([output](../results/validation/vtm/extraction-lifetime.txt)). With asynchronous
allocation, that checker reports use-before-allocation even in the committed
minimal nested-counter example with no Warpower imports and an explicit setup
synchronization. The diagnostic documentation includes that reproducer and
distinguishes its settings from the normal pooled runtime.

On NVIDIA L40, the 1,000-site torus extraction, topology/quality diagnostics,
changing-input producer, and volume consumer take **23.22 ms** median over ten
samples. Including 20 boundary/interior optimization steps takes **336.23 ms**
median over five samples. Both measurements have no CPU quota throttling
([extraction](../results/validation/vtm/extraction-torus.json),
[combined loop](../results/validation/vtm/extraction-combined.json)). Selection
and topology match independent CPU calculations. These are resident-device
intermediate pipeline timings, not CPU speedup comparisons.

The combined case retains **4,186 tetrahedra** and **1,144 boundary faces**, with
median radius ratio **0.873** and worst ratio **0.025**. Its volume is **4.84% below**
the triangulated input domain; directed sampled boundary RMS distance is
**0.01861** in model units. Three cycling input translations produce slightly
different optimization trajectories and final quality minima; each JSON records
the final state for its particular replay count. Worst-element improvement,
gradation, and better approximation remain required.

The [combined native audit](../results/validation/vtm/extraction-device-audit.json)
records 322 kernel, 116 memset, 26 conditional, and 22 device-to-device copy nodes.
Three complete profiled replays execute 9,027 kernel activities. There are no
host transfers, host callbacks, or allocation/free nodes; CPU activity is three
graph submissions, profiler/capture bookkeeping, and one final synchronization
outside geometry. All prior implementation modules remain byte-identical to
`1cbea48` ([source audit](../results/validation/vtm/extraction-standalone-isolation.json)).
An installed wheel also passes a captured extraction/changing-count smoke test
outside the checkout, with exact source matching
([package record](../results/validation/vtm/extraction-wheel.json)).

```bash
python -m benchmarks.extraction --reference --output results/local/extraction.json
python -m benchmarks.extraction --combined --reference --repeats 5 \
  --output results/local/extraction-combined.json
nsys profile --trace=cuda --cuda-graph-trace=node --cuda-event-trace=false \
  --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi \
  --capture-range-end=stop --output=results/local/extraction-loop \
  python -m benchmarks.extraction --combined --iterations 20 --repeats 1 --profile \
  --dot results/local/extraction-loop.dot
nsys export --type sqlite --output results/local/extraction-loop.sqlite results/local/extraction-loop.nsys-rep
python -m benchmarks.graph_capture --extraction --dot results/local/extraction-loop.dot \
  --analyze results/local/extraction-loop.sqlite --output results/local/extraction-audit.json
```
