# Analytic filters in VTM facet search

Certified analytic determinant filters reduce the complete 4,096-site torus
solve from **3,785 to 2,825 ms**, a 25.4% reduction. The 1,000-site solve falls
from **824 to 737 ms**. Six further meshing examples improve by 2–25%, with
matching tetrahedron sets, roles, oriented boundaries and iteration work.
Standalone diagram/Delaunay measurements remain within 0.8% of their controls.
The [wide CPU speedup goal](cpu-mesher-baseline.md) remains open.

Only two predicate calls in the reuse-owned exposed-face search change.
The new float64 determinant/permanent filters return certified strict signs;
uncertainty uses the original interval, exact-integer and symbolic predicates.
No allocation, search order, hull-support check, topology check or proof lifetime
changes. The [bound analysis](analytic-predicate-bounds.md) describes the source,
sign conventions, compiler requirements and finite-float32 range argument.

## Complete solves

Measurements use NVIDIA L40 / Warp 1.19.0, baseline `fb795ac` and runtime
`0590626`. Every captured replay starts from fresh samples and includes current
domain/sizing refit, growth, fitting, quality work, extraction and diagnostics.
Allocation, compilation, transfers, validation and serialization are outside
replay timing. The 1k torus uses five timed replays; the other cases use three.
All accepted initial and measured CPU quota windows are unthrottled.

| Example | Sites | Before (ms) | After (ms) |
|---|---:|---:|---:|
| Uniform torus | 1,000 | 823.739 | 736.951 |
| Uniform torus | 4,096 | 3,785.001 | 2,824.748 |
| Graded torus | 4,096 | 4,238.005 | 3,167.491 |
| Thin hand | 1,000 | 2,381.579 | 2,334.538 |
| Cavity | 1,000 | 979.455 | 916.290 |
| Disconnected solids | 1,000 | 687.652 | 638.681 |
| Uniform Armadillo | 4,096 | 4,635.605 | 3,740.246 |
| Graded Armadillo | 4,096 | 8,170.487 | 7,275.240 |

All eight pairs retain identical input arrays, parameters, full/material
tetrahedron sets, roles, oriented boundary triangles and recorded iteration work.
Coordinates are bitwise identical in the two uniform torus cases, graded torus
and both Armadillo cases. The final hand differs in two coordinates by at most
4.88e-19; cavity and disconnected outputs each differ in one coordinate by one
float32 ulp (2.33e-10 and 4.66e-10). Fresh runs of the unchanged control reproduce
the cavity/disconnected prototype coordinates exactly and exhibit hand variation
up to 4.20e-19. These observations support ordinary repeat-run variation; no
bitwise identity is claimed for those three feature cases.

Independent CGAL global predicates check the entire convex hull, including
positive orientation, empty spheres, paired faces and supporting hull faces.
Independent NumPy/AABB checks cover material geometry, quality and boundary fit;
boundary vertex links are single cycles. All six gallery VTU exports round-trip
exactly, and their boundaries have zero exact self-intersections.

The explicit persistent workspace pools are unchanged at 273,862,732 bytes for
the 1k torus and 1,743,411,108 bytes for 4k. Warm-cache setup is 1.507→1.498 s
and 2.919→2.902 s; first graph execution is 1,064→960 ms and 4,540→3,565 ms.
These are separate from replay medians and do not establish a cold-setup gain.
The compiled search remains at 255 registers/thread; local storage changes from
15,808 to 15,800 bytes/thread. No explicit workspace buffer is added.

## Standalone operations

The source audit verifies that every existing production function except this
private search is unchanged, as are benchmark/example code and existing tests.
The new filter function bodies and constants match the validated private probe.
Independent APIs retain their ordinary defaults. Delaunay measurements use nine
complete-operation samples; diagram adjacency uses 30 samples after five warmups.
All compared inputs, work counters and diagnostics match, with clean quota
windows. The small timing differences below are not attributed to this change.

| Operation | Before (ms) | After (ms) | Change |
|---|---:|---:|---:|
| Delaunay, 1k random | 9.174560 | 9.188128 | +0.15% |
| Delaunay, 100k random | 182.389404 | 181.951904 | −0.24% |
| Delaunay, grid | 2.211200 | 2.193856 | −0.78% |
| Delaunay, fitted hand | 13.386496 | 13.475136 | +0.66% |
| Diagram adjacency, 100k | 18.098401 | 18.036377 | −0.34% |

## Validation and retained attempts

The full 694-test suite passes. The 44 new cases include independent rational
error enclosures and six-phase captured queries, with every strided lane checked
against the preceding exact/interval search. The initial integrated focused run
passed 95 tests; lane-by-lane assertions were then strengthened and included in
the successful full suite.

All 51 focused cases pass padded memcheck. The 44 new predicate/query cases also
pass API-memory and full stream-ordered lifetime checks with pooling disabled.
All seven routed cases pass separate API-memory/use-after-free checks, each in
a fresh instrumented process. The explicit 131,072-byte diagnostic CUDA stack
allowance belongs to the test harness, not production allocation.

Full allocation-order tracking remains limited by
[#91](https://github.com/alecjacobson/warpower/issues/91). The retained strict
4k duplicate-cube control exits 99 with five native CUB histogram atomics flagged
as potentially preceding allocation, followed by a Python crash. All five printed
diagnostics identify the same +0x1a40 instruction and 10,751-byte scratch as the
preceding-runtime and standalone controls. This is a retained failure, with no
suppression or production allocator workaround; full routed allocation-order
coverage is not claimed.

Native profiling checks one full fresh 4k torus solve and an eight-phase
device-controlled producer/reconstruction/consumer loop. The full solve has
60,862 kernel activities, 33,222 DOT kernel nodes and 2,932 device-to-device copy
nodes. The loop has 1,345 activities, 2,389 kernel nodes and 228 device-to-device
copy nodes. All six valid loop phases match independent SciPy hulls; invalid-count
and NaN phases return empty outputs and recover. Both traces have one graph
launch and one final synchronization, with no observed host transfers, callbacks
or allocation/free graph nodes. First-launch graph instantiation/upload APIs are
recorded, rather than described as already excluded setup.

The full native activity sum falls from 3,986.300 to 3,031.179 ms. The changed
search falls from 1,273.260 to 313.394 ms across the same 326 calls. Prepared seed
checks (736.293 ms) and hull-support recording (454.475 ms) are now larger costs.
These instrumented durations diagnose the change; the clean table above supplies
the performance measurements.

The initial graded Armadillo control reported four throttled CPU quota periods.
Its results are retained and excluded from the timing comparison; a fresh pair
has clean windows. The first repeat failed during Warp initialization because
the chosen compiler-cache directory was unwritable. No timing from that failure
is used. Only regenerable compiler artifacts were subsequently copied to a
writable ephemeral cache, hash-verified, and removed from the persistent cache.
Source, Git history, inputs and unique outputs remained persistent.

Two early comparison-helper runs stopped at overly strict bitwise coordinate
assertions for feature cases. Their source and logs are retained. The unchanged
control repeats above investigate those differences; connectivity/work checks
and independent geometry checks remain exact. The final report records every
changed coordinate rather than hiding it behind a tolerance.

The [manifest](../results/validation/vtm/analytic-filters/manifest.json) records
runtime/reference hashes, commands, raw samples, private probes, failed attempts,
source isolation, instrumentation and independent checks. The
[final comparisons](../results/validation/vtm/analytic-filters/analytic-final-independent.json)
contain all eight meshes and five standalone pairs. Raw timing inputs are shared
only after byte-identical copies are verified; the storage records identify each
retained persistent original. The main README keeps its earned Paragram result.

Tracked in [#166](https://github.com/alecjacobson/warpower/issues/166).
[#167](https://github.com/alecjacobson/warpower/issues/167) separately investigates
splitting the strict search and exact fallback into device kernels; no such split
is part of this implementation.
