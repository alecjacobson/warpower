# Grouped current-input seed queries

Private VTM reconstruction groups the current Morton-sorted points in batches
of 32 before checking retained tetrahedra. Each group gets an exact float32
coordinate-wise minimum and maximum. All 32 query lanes for a tetrahedron can
then reject that group with the existing conservative circumsphere box. This
avoids reading and testing individual points from excluded groups without a
per-lane hierarchy traversal or stack.

Every invocation refreshes the group boxes from the current point permutation.
No bounds or sphere proof survive an input change. Groups include duplicate and
inactive input slots; including extra points only enlarges a box. Each surviving
rank maps back to its original point ID before canonical-owner and defining-site
checks. A partial last group masks ranks outside the reserved point capacity.
Surviving queries use the unchanged point-specific Ball64 expression and complete
symbolic fallback, and rejected cells continue through the remaining queries.

The exclusion argument uses set containment: every query point is inside its
group box, and the existing sphere box encloses the closed physical sphere.
Strict separation of the two boxes proves every group point strictly outside
that sphere. Touching or overlapping boxes are retained. An invalid sphere
enclosure rejects no groups. Thus the optimization cannot discard a physical
tie that requires symbolic handling. All final topology, coverage and hull-support
certificates remain mandatory.

Only `CertifiedSeeds` with its private `prepare_geometry=True` option selects
the new subclass. The original prepared checker remains available as a reference.
Standalone diagram, Delaunay and ordinary seed-filter defaults keep their
existing kernels and dispatch. Each prepared workspace adds two vec3 arrays,
24 bytes per 32-point group, and one group-preparation launch per invocation.
All storage is allocated at setup and all decisions remain on device.

## Complete solves and storage

Runtime `b5d369c` is compared with the preceding `6a82244` implementation on
NVIDIA L40 / Warp 1.19.0. Controls come from the fresh private paired experiments;
the integrated implementation reproduces those improvements. Controls at main
`b0fb1d4` and `d8a9b33`, and the standalone baseline `ce51096`, have identical
runtime source. Every first and timed quota window is unthrottled.

Each replay includes domain refit, fresh sampling, growth, fitting, quality
stages, extraction and diagnostics. Setup, compilation, transfers and offline
checking are excluded. The 1k torus uses five measured replays; other complete
cases use three. These compare GPU versions, not a quality-matched CPU mesher.

| Complete fresh solve | Before, ms | Grouped queries, ms |
|---|---:|---:|
| Uniform torus, 1,000 sites | 719.067 | 704.399 |
| Uniform torus, 4,096 sites | 2,640.817 | 2,401.048 |
| Graded torus, 4,096 sites | 2,967.448 | 2,713.681 |
| Thin hand, 1,000 sites | 2,318.374 | 2,301.660 |
| Cavity, 1,000 sites | 904.674 | 897.539 |
| Disconnected solids, 1,000 sites | 632.211 | 626.802 |
| Armadillo uniform, 4,096 sites | 3,601.712 | 3,434.469 |
| Armadillo graded, 4,096 sites | 7,163.157 | 6,996.960 |

The largest complete-solve reduction is 9.1% on the 4k torus; the smaller
differences on the hand, cavity and disconnected solids are below 1.1%.
The persistent allocator-pool footprint increases by 768 bytes at 1k and 5,376
bytes at 4k, including the smaller routed workspaces. These are pool allocations,
not process-wide peak memory measurements. Warm-cache setup takes
1.587→1.513 seconds at 1k and 2.980→2.937 seconds at 4k. First graph executions
take 952.167→925.225 ms and 3,435.424→3,198.813 ms. Setup is not repeatedly
timed; no cold-start or setup-speed improvement is claimed.

All eight final pairs preserve full/material tetrahedron sets, oriented
boundaries, site roles, work counters, parameters and input arrays. Six preserve
positions bitwise. The hand differs in two coordinates by at most 2.846e-19;
the cavity differs in one coordinate by one float32 ULP (2.328e-10). Every final
full hull passes independent CGAL exact global sphere, orientation, incidence
and support checks. NumPy/AABB checks measure material quality and fit and verify
single-cycle boundary-vertex links. All six gallery outputs round-trip through
VTU exactly and have zero CGAL boundary self-intersection pairs.

## Standalone routines

Standalone Delaunay and diagram code paths are unchanged. Delaunay measurements
use nine replays per initial pair; the diagram uses thirty. No speedup is
attributed to their timing differences, and all measured quota windows are clean.

| Standalone operation | Before, ms | After, ms |
|---|---:|---:|
| Certified Delaunay, 1,000 sites | 9.181152 | 9.170144 |
| Certified Delaunay, 100,000 sites | 183.707199 | 182.013214 |
| Certified Delaunay, grid ties | 2.200704 | 2.201344 |
| Certified Delaunay, fitted hand, pooled | 13.434896 | 13.386512 |
| Diagram, 100,000 sites | 18.176844 | 18.071173 |

The initial fitted-hand control varies from 13.373 to 13.815 ms, producing a
13.698→13.387 ms median pair. One fresh reverse-order pair gives 13.402→13.386 ms.
The table pools all eighteen samples per version; no sample is discarded, and
the initial results remain separately recorded. Every operation preserves its
input/configuration and output diagnostics. The largest pooled difference is
0.93%; no standalone regression is observed.

## Tests and device execution

The integrated focused set passes **119 tests** in 89.94 seconds; the full suite
passes **786 tests** in 296.76 seconds. The new tests comprise 36 captured-mask
comparisons against previous/current exhaustive checkers and independent rational
symbolic masks for small inputs, plus 14 direct group-bound cases. They cover
changed positions, permutations and duplicate owners, empty/shrinking/restored
prefixes, poisoned old bounds/live masks, block sizes 32/256 and extreme finite
float32 bounds. The unchanged point-enclosure tests remain in the full suite.

An instrumented 4k torus diagnostic records 624 query invocations and 12,882,912
cell instances over three complete graph executions. It skips 1,402,978,068 of
1,604,092,896 group decisions (87.46%). Surviving groups contain 6,357,290,319
eligible point visits after canonical/defining-site checks; point boxes reduce
those to 1,816,319,529 strict Ball64 evaluations. No exact fallback is needed
on this trajectory. All counters are int64, the saved output is bitwise equal
to the uninstrumented final mesh, and work counters match. Instrumentation
timings are excluded from performance evidence. These counts do not claim that
other geometries avoid symbolic fallback; the tie tests explicitly require it.

All 50 new cases pass padded Compute Sanitizer memcheck and API-memory/full
allocation-lifetime checking with zero errors. A routed 2,050-site integration
case also passes padded checking and API-memory/use-after-free tracking. These
six completed jobs use a 128 KiB diagnostic CUDA stack; lifetime runs disable
the diagnostic array pool. Production stack and allocation settings are unchanged.
The routed integration is not a new full allocation-order pass: the existing
[conditional CUB instrumentation limitation](cuda-validation.md#conditional-native-radix-sort-scratch)
remains separately documented.

The native full-solve trace contains 61,070 kernel activities, including 208
group preparations and 208 grouped checks. Its graph contains 33,270 kernel
nodes and 2,932 device-to-device copies. The separate eight-phase device-controlled
producer/constructor/consumer loop has 1,348 kernel activities, 2,393 graph
kernel nodes and 228 device-to-device copies. Both traces have one graph launch
and one final synchronization, no observed host transfers, and no host callback
or allocation/free graph nodes. The first launches include graph instantiation
and upload API calls; these traces are capture audits, not clean timing samples.
All six valid loop phases match independent SciPy meshes, while both invalid
phases publish empty output before valid recovery.

The full native kernel-activity sum is 2,603.920 ms. Grouped checking accounts for
308.460 ms, compared with 554.292 ms for the preceding checker in its archived
trace. Both use 255 registers per thread at block size 32. Hull-support recording
now accounts for 454.014 ms, facet search 313.965 ms and cell construction
294.480 ms. These diagnostic activity durations identify remaining costs;
the clean complete-solve measurement is the 2,401.048 ms median above.

## Evidence and scope

The [manifest](../results/validation/vtm/grouped-seed-queries/manifest.json)
records source isolation, prototypes, all timings, meshes, independent checks,
test/audit logs and reproduction commands. The numerical kernels and workspace
match the tested prototype's syntax tree. The only existing runtime edit selects
the grouped subclass inside the private preparation option; the ordinary
prepared checker, standalone kernels and benchmark implementations are unchanged.

The [near-4k CPU comparison](cpu-size-frontier.md) remains a separate experiment
with different quality and stopping behavior. This improvement does not meet
the wide CPU outperformance goal. No fitting schedule, quality target or README
flagship result changes.

The earlier [DFS traversal layouts](prepared-hierarchy-probes.md) and
[strict/exact split](prepared-split-probe.md) were slower and remain unadopted.
Their retained results are separate from this flat-group design.

Tracked in [#172](https://github.com/alecjacobson/warpower/issues/172).
The separate [#177](https://github.com/alecjacobson/warpower/issues/177) follow-up
investigates existing segment-tree bounds and stackless traversal; it is not
implemented by this change.
