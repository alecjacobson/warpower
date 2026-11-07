# Certified fragment reuse through 4,096 sites

The active VTM mesher now uses certified fragment reuse on its full reconstruction
route at reserved capacities from 2,049 through 4,096 sites. This reduces complete
4k torus and Armadillo solves by 1.82–2.96× in the measured cases. Only two host
classes change; all existing Warp kernels/functions and independently callable
power-diagram, Delaunay and fixed-boundary ODT implementations stay unchanged.

Every call filters previous tetrahedra against **current original-coordinate
input**, including canonical duplicate ownership. The completion then retains
those just-established cell proofs through compaction and deduplication. New
cells receive full sphere checks, and the entire result still passes orientation,
coverage, face-pairing, connectivity and hull-support checks. No sphere proof
survives an input change. Failed fragment completion uses the ordinary certified
constructor; exhausted output capacity still produces an error and empty output.

Fresh mesher initialization clears the new full-route fragment as well as the
existing 1,024/2,048-site child fragments. Larger reserved capacities retain
ordinary full reconstruction. The small child paths remain unchanged. Reserved
capacity determines the full-route strategy; requesting 4,096 active sites in
an 8,192-slot workspace does not select this new path.

## Complete solves and storage

Runtime `f8b5cfa` is compared with baseline `cca7a1d` (runtime unchanged from
`9228a3e`) on NVIDIA L40 / Warp 1.19.0. Three measured complete fresh replays
include domain refit, fresh sampling, progressive fitting, refinement, quality
stages and extraction. Setup, compilation, transfers and offline checking are
excluded. All first and measured timing windows have zero CPU quota throttling.

| Complete fresh solve | Before (ms) | After (ms) | Ratio |
| --- | ---: | ---: | ---: |
| Uniform torus, 2,050 sites | 3,287.902 | 1,490.729 | 2.206× |
| Uniform torus, 3,072 sites | 4,620.510 | 2,472.733 | 1.869× |
| Uniform torus, 4,096 sites | 7,065.010 | 3,787.116 | 1.866× |
| Graded torus, 4,096 sites | 7,730.856 | 4,237.962 | 1.824× |
| Armadillo uniform, 4,096 sites | 13,732.105 | 4,635.680 | 2.962× |
| Armadillo graded, 4,096 sites | 15,205.459 | 8,176.069 | 1.860× |

The last four use preceding control runs from the same unchanged baseline
runtime; the 2,050/3,072 pairs are newly interleaved before/after runs. A private
prototype was also paired with each control before integrating the change.
Final runtime measurements reproduce the prototype gains. These compare GPU
implementations; the [wide CPU speed goal](cpu-mesher-baseline.md) remains unmet.

The 4,096-slot torus allocator-pool measurement grows from 1,709,304,125 to
1,743,411,108 bytes: an additional 34,106,983 bytes (32.53 MiB). This also applies
when reserving 4,096 slots for a smaller target that never enters the full route.
At 2,050/3,072 slots the extra pool storage is 18,766,073/25,909,607 bytes.
The 1,000-slot and 8,192-slot measurements are identical before and after.
These measure persistent workspace pool allocations, not process-wide peaks.

For the 4k torus, setup takes 2.787→2.968 s and first execution 7.845→4.578 s.
Setup is not timed repeatedly and no setup-speed improvement is claimed.
All raw samples and separate setup/first-execution results are retained.

All ten complete-solve pairs preserve coordinates bitwise, roles, full/material
tetrahedron sets, oriented boundaries, input arrays and iteration work. Every
final hull passes independent CGAL exact orientation, global empty-sphere,
face-pairing and hull-support checks. Independent NumPy/AABB measurements check
material quality and fit, and every boundary vertex has a single-cycle link.
The three gallery exports round-trip exactly through VTU and have zero CGAL
boundary self-intersection pairs. These are offline checks; they do not contribute
to any runtime decision. The [independent result record](../results/validation/vtm/large-fragment-reuse/large-reuse-final-independent.json)
includes all comparisons and per-mesh measurements.

## Unchanged paths

The 1k default uses five measured replays; reserved-prefix comparisons use three.
Their changes are below 0.15% and treated as timing variation.

| Complete fresh solve | Before (ms) | After (ms) |
| --- | ---: | ---: |
| Torus, capacity/target 1,000 | 822.443 | 823.249 |
| Torus, capacity 4,096 / target 1,000 | 890.399 | 889.250 |
| Torus, capacity 4,096 / target 2,048 | 1,522.928 | 1,520.647 |
| Torus, capacity 8,192 / target 4,096 | 14,271.323 | 14,265.623 |

Standalone full-operation timings use nine Delaunay replays and thirty diagram
runs. Differences are below 0.6%, with zero timed CPU quota throttling.

| Standalone operation | Before (ms) | After (ms) |
| --- | ---: | ---: |
| Certified Delaunay, 1,000 sites | 9.204768 | 9.187360 |
| Certified Delaunay, 100,000 sites | 182.407974 | 182.585571 |
| Certified Delaunay, grid ties | 2.204800 | 2.198048 |
| Certified Delaunay, fitted-hand cloud | 13.492512 | 13.421792 |
| Diagram, 100,000 sites | 18.222226 | 18.125774 |

## Why the initial extension was slower

Merely selecting full-route fragment reuse made the 4k torus slower:
7,065.010→12,008.697 ms. The existing completion disabled parallel proof
consumption above 2,048 sites, causing the final certificate to repeat sphere
checks for cells the current seed filter had already certified. One native
diagnostic records 8,338.888 ms of `check_spheres` activity across 405 calls.
The original full-construction trace spends 5,280.140 ms in `build_cells` over
190 calls. Activity sums explain costs; they are not clean end-to-end timings.

A separate diagnostic records 168 accepted attempts out of 169 reuse attempts
across 170 full-route calls. Of those attempts, 128 retain at least 99% of previous
cells; 30 retain every previous cell. Reuse failure frequency was not the
dominant problem.

A cooperative BVH seed-filter prototype passed 25 existing and 18 additional
oracle tests, including changing captured inputs and scales from 2^-60 to 2^60.
It nevertheless took 13,169.271 ms on the same complete solve and was rejected.
The accepted change enables existing current-input proof consumption before the
reuse completion is captured. No query kernel or precision policy changes.

## Validation and capture

The seven new tests cover captured producers and consumers at 2,050/4,096 sites,
nearby/unrelated input, shrinking/invalid prefixes, NaN recovery, fixed buffer
addresses and fresh reset. A deliberate corruption of a newly introduced,
unproved cell causes certificate rejection and ordinary reconstruction; an exact
rational predicate independently proves the injected nonempty sphere. Other
cases cover insufficient output capacity and cospherical duplicate cubes at
scales 2^-60, 1 and 2^60. The seven tests pass in 11.93 s; the full suite passes
**650 tests** in 288.73 s.

All seven new cases pass padded Compute Sanitizer memcheck individually, each
with zero errors. These use 32-byte padding and a 128 KiB diagnostic CUDA stack;
production settings are unchanged.

The initial combined-process run is **incomplete**. After three cases it stopped
making visible progress during the capacity-failure case, with one CPU core busy
and the GPU idle. It was deliberately terminated at 11 minutes 52 seconds;
no zero-error exit was obtained. That same unchanged case passes in 21.65 seconds
in a fresh instrumented process. The combined log, unsuccessful debugger attach,
process snapshot and interruption record are retained under
[#155](https://github.com/alecjacobson/warpower/issues/155). These observations do
not establish a root cause. All seven cases were rerun separately with timed
Python stack diagnostics and bounded process-group cleanup.

All seven cases also have zero-error API-memory and use-after-free coverage with
the diagnostic array pool disabled. Full allocation-order tracking passes the
three 2,050-site cases. It fails the four 4,096-site cases in native CUB radix
sorting; those four pass use-after-free tracking separately. Full tracking is
**not** claimed green.

Fresh controls reproduce the same displayed native histogram-kernel instruction,
access-before-allocation diagnostic and 10,751-byte sort scratch in the preceding
runtime and in a standalone conditional Warp sort with no Warpower geometry.
The standalone 4,096-key top-level/full-tracking, conditional/ordinary-memory and
conditional/use-after-free controls each pass with exact keys and payloads.
This supports the existing [#91 limitation](cuda-validation.md#conditional-native-radix-sort-scratch);
it does not prove arbitrary allocation order safe or justify suppressing errors.
The [lifetime analysis](../results/validation/vtm/large-fragment-reuse/large-reuse-lifetime-analysis.json)
retains every failure and identifies the exact validated mode for each test.

The native complete 4k torus trace contains 60,862 kernel activities; the graph
has 33,222 kernel nodes and 2,932 device-to-device copy nodes. A second trace
executes eight producer/routed-constructor/consumer phases inside one
device-controlled loop: 1,345 kernel activities, 2,389 graph kernel nodes and
228 device-to-device copy nodes. Both have one graph launch and one final
synchronization, no observed host transfers, and no host/callback or
allocation/free graph nodes. The first launches include graph instantiation and
upload API calls; these traces are capture audits, not clean timing samples.
All valid loop phases match independent SciPy meshes, and both invalid phases
publish empty output before valid recovery.

The integrated full-solve activity sum is 3,986.300 ms. The new sphere-proof path
uses 126.032 ms across 534 certificate calls, while face candidate search now
accounts for 1,273.260 ms. These diagnostic durations identify remaining work;
the clean complete-solve timing is the 3,787.116 ms median above. The final full
reconstruction consumes 73,496 current-input cell proofs and checks 27 newly
introduced cells across its completion rounds.

The [full native audit](../results/validation/vtm/large-fragment-reuse/large-reuse-full-native-audit.json),
[device-loop audit](../results/validation/vtm/large-fragment-reuse/large-reuse-loop-native-audit.json)
and [kernel attribution](../results/validation/vtm/large-fragment-reuse/large-reuse-native-audit-summary.json)
retain the counts and observed API calls.

## Evidence and reproduction

The [evidence manifest](../results/validation/vtm/large-fragment-reuse/manifest.json)
records raw samples, exact commands, private prototypes,
unsuccessful variants, native reports, independently checked meshes, test/audit
logs, source isolation and a hash manifest. Original native traces remain
persistent; exported SQLite files and graph DOT files are losslessly compressed.
The temporary SQLite export uses reproducible scratch only.

```bash
python -m benchmarks.mesher --n 4096 --repeats 3 --save-mesh torus4096.npz --output torus4096.json
python -m examples.vtm_gallery --case graded-torus --n 4096 --repeats 3 --output graded4096
python -m pytest -q tests/test_large_reuse.py
```

Tracked in [#161](https://github.com/alecjacobson/warpower/issues/161) and
[#163](https://github.com/alecjacobson/warpower/pull/163). The main README retains
the independently earned Paragram result.
