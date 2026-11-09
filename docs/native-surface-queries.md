# Native Warp surface queries

The optional `SurfaceDomain(query_backend="native")` backend uses Warp's native
FP32 closest-point queries for domain queries, tetrahedron selection/extraction,
and boundary-distance metrics. The default is `query_backend="reference"`.
This work follows the [smooth-input performance plan](https://github.com/alecjacobson/warpower/issues/264).

```python
from warpower import SurfaceDomain, VariationalMeshingWorkspace
import warp as wp

domain = SurfaceDomain(vertices, triangles, query_backend="native",
                       validate_geometry=False)  # Trusted valid input.
mesher = VariationalMeshingWorkspace(domain, 8192, initial_count=256)
with wp.ScopedCapture() as capture:
    # The input surface stays fixed; upstream device kernels may update the
    # mesher's documented live controls, and downstream kernels consume output.
    mesher.run()
wp.capture_launch(capture.graph)
```

`validate_geometry` is independent of backend selection and remains `True` by
default. Disabling it skips expensive input intersection/orientation validation;
index, topology, nonfinite and degenerate-face checks remain. It does not disable
generated-mesh certification, material topology checks or transactional recovery.

## Arithmetic and scope

Signed queries use `wp.mesh_query_point` with its three oriented-ray votes.
The initial normal-based query was rejected after it misclassified the center of
a cavity: normals from equally distant opposing faces cancel. Distance-only
queries use `wp.mesh_query_point_no_sign`. Successful native queries are not
rechecked with the reference traversal. Query points and local triangle arithmetic
are FP32; physical tetrahedron centers, volumes, radius ratios and metric sums
retain FP64 and the existing wide fallback. Delaunay predicates and certification
are unchanged.

Use well-scaled, watertight, consistently oriented, non-self-intersecting surfaces
with triangles that Warp's native query accepts. Warp excludes triangles with
relative twice-area below `1e-6`; this backend does not repair or conservatively
retest those triangles. The reference domain accepts triangles down to `1e-12`
and keeps its existing behavior. FP32 queries can change boundary decisions and
optimization trajectories. Equal connectivity or bitwise coordinates are not the
accuracy contract for this explicitly selected mode.

The reference surface hierarchy remains for input validation, initial sampling
and containment/exterior recovery. The native BVH is additional setup storage.
Native selection avoids allocating the reference query cache. This backend does
not change the nearest-site quadrature accumulation; selectable accumulation
precision is tracked separately in [#266](https://github.com/alecjacobson/warpower/issues/266).

## Capture limitation in Warp 1.19

Queries and complete meshing against a fixed surface remain allocation-free in
captured loops, including conditional subgraphs. They use stable public buffers.
If surface vertices change, call `domain.refit()` before querying, as usual.
Native refit works in a top-level capture and executes entirely on device, but
Warp's internal average-edge-length scan contributes device `MEM_ALLOC` and
`MEM_FREE` nodes. Consequently native refit cannot be placed inside a conditional
subgraph. Do not omit refit after deformation.

The default reference backend retains its stronger allocation-free refit contract,
including nested captures. [#267](https://github.com/alecjacobson/warpower/issues/267)
tracks native persistent scratch support. Wrapping native refit in a conditional
child does not resolve this: scan lacks the side-allocation cache used by sort.
No native pointer patching or custom C++/CUDA is introduced.

## Reproduction

```bash
python -m examples.vtm_gallery --case surface --input armadillo.off \
  --n 8192 --initial 256 --seed 2026 --subdivisions 1 \
  --crease-subdivisions 1 --feature-angle 120 --corner-angle 45 \
  --quality-rounds 4 --repeats 3 --query-backend native \
  --no-validate-geometry --output results/local/armadillo-native
```

Use `--query-backend reference` with the same other options for the backend
ablation. The input is the pinned reduced Stanford Armadillo described in the
[gallery](examples.md). The main README's earned Paragram result is unchanged.

## Complete-solve comparison

NVIDIA L40, Warp 1.19.0, runtime `0dc7217`, based on main `afec85d`.
Both backends use `validate_geometry=False`, seed 2026, 256 initial sites and four
quality rounds. Each value is the median of three additional fresh captured
solves, including refit, sampling, growth, fitting, quality moves and extraction.
Setup/first graph instantiation, transfers, export and independent checking are
outside these replay timings. CPU-quota windows are clean in all ten jobs.

| Case | Reference, ms | Native, ms | Reference/native |
|---|---:|---:|---:|
| Uniform Armadillo, 8,192 sites | 3455.885 | 2594.664 | 1.332× |
| Graded Armadillo, 8,192 sites | 3651.010 | 2750.585 | 1.327× |
| Torus, 1,000 sites | 485.272 | 484.032 | 1.003× |
| Mechanical plate, 1,000 sites | 829.187 | 746.223 | 1.111× |
| Cavity, 1,000 sites | 352.495 | 339.189 | 1.039× |

These are short GPU-backend comparisons, not CPU speedup measurements or a
universal gain. Dense Armadillo solves use about 25% less GPU time. The small
torus is effectively unchanged.

All ten meshes pass fresh independent full-hull orientation/Delaunay/hull checks,
positive material-cell and incidence checks, single-cycle boundary vertex links,
exact boundary self-intersection checks, and VTU round trips. Full/material
connectivity, oriented boundary and roles agree in all five pairs. Four pairs
have bitwise identical points. Uniform Armadillo differs by at most `3.58e-7` per
coordinate and takes three accepted selective steps instead of two (the
126 accepted global steps and rejection counters agree). Different valid
trajectories remain allowed for this backend.

Uniform/graded Armadillo retain 39,377/39,510 material tetrahedra and minimum
radius ratios of 0.031019/0.060836. Independently measured two-sided sampled
surface RMS is 0.14799%/0.16776% of the input box diagonal. These fitting metrics
are sampled estimates, not Hausdorff bounds, and do not promise minimum-quality
thresholds. The wide comparable-quality CPU speedup goal remains open.

## Existing APIs and capture checks

The ordinary test suite passes all 1,396 cases, including public routines, invalid-input recovery,
changing-domain queries and nested meshing against a static native surface.
Existing device definitions in the three edited runtime modules are unchanged;
the other 73 existing runtime files are byte identical to main. Backend selection
happens in Python at setup/capture construction, with no new default GPU work.

The paired standalone geometry gate passes for eager uniform/clustered diagrams
through one million sites, and weighted/unweighted captured adjacency,
tetrahedron-only and combined output. Outputs match, and eager kernel/allocation
traces match. No paired 95% interval lies wholly above the +2% regression threshold.
Individual means range from −1.85% to +1.56%; this is a measured gate, not a
universal performance guarantee. The compact record includes all samples,
intervals and allocator measurements.

Separate certified Delaunay controls change by −0.06% to +0.30% across 1K/100K
random points, a lattice and fitted hand. The default 1K torus mesher changes by
+0.05% (489.143 → 489.401 ms). All reported full-workspace timing windows are
unthrottled; the retained raw control also records throttling during the lattice's
separate base-builder measurement, which is not the reported full-workspace time.

A warmed native producer/refit/mesher/consumer trace executes five graph replays
through valid input, changed geometry/target/seed, invalid input and recovery.
It has 48,467 kernel activities, five graph launches and one final synchronization.
All observed transfers and all 761 graph copy nodes are device-to-device; there
are no host callbacks. Its one device allocation/free pair is the documented
native-refit limitation, not a pass of the default no-allocation validator.

[Compact evidence](../experiments/native-surface-queries/README.md) retains the
complete backend comparison, independent checks, public regression controls and
native capture audit. Raw meshes and traces remain outside Git.

Native lifetime instrumentation has a separate limitation tracked in
[#269](https://github.com/alecjacobson/warpower/issues/269): with
`--track-stream-ordered-races all`, a conditional query reports that the live
328-byte native mesh descriptor may be read before allocation. The
[minimal Warp-only reproducer](../experiments/native-surface-queries/native_mesh_lifetime_v1.py)
imports no Warpower geometry, keeps every owner alive, warms the query and
synchronizes before capture. Ordinary and padded nested controls pass; a
top-level query passes lifetime tracking; one conditional query fails even when
the surface is static and there is no refit in the graph. Disabling Warp's array
pool does not disable native stream-ordered descriptor allocation.

These controls isolate a native Mesh/nested-capture tracking limitation. They do
not justify claiming clean full-pipeline lifetime instrumentation or unrestricted
lifetime correctness. The failed reports are retained alongside the passing
controls, and the default reference backend remains available.

All eight new native cases pass padded memcheck in isolated jobs with a diagnostic
32 KiB stack allowance. The cube/torus complete solves take 230/356 seconds under
instrumentation; the torus passes an isolated 600-second retry after exceeding
the initial 300-second limit. These are diagnostic durations, not performance
samples. The six direct query cases also pass allocation-lifetime checking with
synchronous Warp array allocation. Full nested native lifetime checking remains
unqualified because of #269; no production allocator/stack settings are changed.

## Event-dependency control

An additional minimal Warp 1.19.0 / L40 diagnostic records an event after Mesh
construction and a synchronized warm query, then waits on it before replayed
queries. All Mesh, event and array owners remain alive. This does not fix #269:

| Query placement / dependency | Allocation-lifetime result |
|---|---|
| One conditional, no wait | Access-before-allocation reports |
| One conditional, top-level explicit event wait | Same reports |
| One conditional, top-level implicit event wait | Same reports |
| Two conditionals, top-level explicit event wait | Same reports |
| Event wait inside the conditional | Capture rejects unsupported event wait |
| Top-level query with explicit event wait | Three replays, zero errors |

Every successfully captured case returns the expected interior sign, including
the cases with lifetime diagnostics. Correct output does not waive those reports.
The unsupported event-wait case is a failed capture, not a passing memory test.
No runtime change follows from this negative result. Source, exact commands,
reported outcomes and raw-log hashes are retained in
[experiments/native-query-events](../experiments/native-query-events/).
