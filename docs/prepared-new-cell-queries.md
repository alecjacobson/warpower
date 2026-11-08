# Shared geometry for new-cell sphere checks

VTM completion now prepares each unproved tetrahedron's sphere geometry once,
then queries the existing current-site segment hierarchy in groups of 32 sites.
It borrows consumed seed geometry arrays and adds no device storage. The original
point predicates, symbolic tie rule, error bits and query-counter definitions
remain unchanged. This applies to the existing private 4,097–16,384-site route;
standalone diagram and Delaunay construction keep their existing paths.

## Complete resident solves

Production `3b68a60` is compared with main `a4f7215` on NVIDIA L40 / Warp 1.19.0.
Seven fresh interleaved pairs each record three complete measured replays,
including domain refit, sampling, fitting, refinement/quality and extraction.
Setup, compilation, transfers and offline validation are excluded. All initial
and measured windows have zero CPU quota throttling. Raw samples and setup costs
are retained in the [comparison](../results/validation/vtm/prepared-new-cell-queries/prepared-new-cell-final-v2-independent.json).

The first batch stopped when the graded candidate recorded one CPU-quota
throttling period (about 54 ms). Its valid mesh and raw timings are retained as
diagnostic evidence. A fresh graded control/candidate pair passed the same
zero-throttling gate; three earlier clean pairs were retained without repeating
them. The primary cohort therefore contains seven complete clean pairs. The
[affected pair](../results/validation/vtm/prepared-new-cell-queries/prepared-new-cell-throttled-pair-independent-v1.json)
is independently validated as well and is excluded from the primary timing table.

| Sites and schedule | Before (ms) | After (ms) | Change |
| --- | ---: | ---: | ---: |
| Uniform torus 16,384, fixed 32 | 5,256.581 | 4,900.978 | -6.76% |
| Uniform torus 8,192, progressive | 3,217.651 | 3,074.205 | -4.46% |
| Uniform torus 8,192, fixed 32 | 1,595.350 | 1,499.436 | -6.01% |
| Torus target 4,096, reserved 16,384 | 7,265.960 | 7,220.299 | -0.63% |
| Graded torus 16,384, progressive | 7,319.497 | 6,976.309 | -4.69% |
| Uniform torus 4,096 | 1,511.306 | 1,510.678 | -0.04% |
| Uniform torus 1,000 | 513.282 | 512.634 | -0.13% |

All fourteen saved meshes pass independent exact full-hull/orientation/empty-sphere,
material topology/quality, boundary vertex-link and boundary self-intersection
checks. Coordinates, canonical full/material cells, oriented boundary, roles and
solver work match in every pair; graded VTU exports round-trip. These are GPU
improvements. The [wide comparable-quality CPU goal](fixed-site-current.md)
remains unmet, and README retains its independently earned Paragram flagship.

## Current-input proof and scratch ownership

The old checker repeated identical Sphere/Box construction in all 32 lanes and
scanned every site. The two [production kernels](../warpower/_unproved_queries.py)
use those same operations once per current unproved cell. Fast math and fused
operations stay disabled. A segment node encloses its descendants: strict box
separation implies the old point-box test would reject every descendant, so
skipped nodes remove no counted queries. Remaining point-box, general Ball64
outside and exact/symbolic checks are unchanged. Query order changes, while
integer additions and error-bit ORs retain their meaning for a given proof mask.
This does not promise identical existing atomic duplicate-winner provenance
across unrelated runs. The [proof note](../results/validation/vtm/prepared-new-cell-queries/prepared-new-cell-proof-v1.md)
records the finite original-input and current-hierarchy prerequisites.

Neither preparation nor queries stop on a preexisting error bit. Invalid
predicates and nonempty spheres still accumulate bits 2 and 8. Each active cell
increments exactly one skipped/checked proof counter. Proved cells never read
stale geometry; each unproved entry is prepared immediately before use.

[Completion](../warpower/_proved_completion.py) borrows the seed checker's
Sphere/Box arrays only after its queries and compaction finish. Every subsequent
seed or certificate pass refreshes its own current geometry. The helper owns
direct array references, with no parent cycle or global registry. Capacity,
device and current permutation association are checked; insufficient-capacity
or unsupported routes retain the old checker. No proof crosses an input update.

| Reserved sites | Before pool bytes | After pool bytes |
| --- | ---: | ---: |
| 8,192 | 2,044,662,012 | 2,044,662,012 |
| 16,384 | 2,538,128,443 | 2,538,128,443 |

These are persistent workspace pool measurements, not process peak memory.
No device arrays are added. The new preparation launch is included in all timings.

## Why the strict/exact split was not adopted

A second private variant added existing certified strict filters after the
original query-count increment, borrowing consumed seed masks/index storage for
exact fallback. It passed 56 primitive/dispatch and 77 integration tests, including
lane 31, overflow rescans and a later query that rejects a cell after an earlier
query accepts it. All eight saved full meshes and both grid outputs passed
independent checks and matched their controls.

It provided no useful extra speed. In diagnostic fixed 16k profiles the simple
checker took 464.820 ms; the split strict pass took 434.386 ms plus 40.593 ms exact.
Both additionally prepared geometry in about 6.75 ms. The strict saving was smaller
than the added exact work. Clean fixed 16k and graded results likewise favored the
simpler version. Additional simple 8k/grid candidates used retained fresh split
controls, rather than new interleaved pairs; their outputs independently match
both controls and split candidates. The [decision](../results/validation/vtm/prepared-new-cell-queries/prepared-new-cell-adoption-decision-v1.md)
and [manifest](../results/validation/vtm/prepared-new-cell-queries/manifest.json)
preserve both variants and their exact experimental scope.

The first simple-grid wrapper followed the virtual-environment interpreter
symlink to system Python and failed at NumPy import before GPU setup. That failed
attempt is retained with the successful explicit-venv retry; no meshing failure
was discarded.

## Tests, memory and captured loops

The complete suite passes **1,279 tests**, including 119 focused completion/query
cases. The 42 new cases cover six captured phases with changing coordinates,
counts and proof flags; poisoned scratch; exact error/query/proof-counter
comparison; independent rational small cases; ties, duplicates, flat cells,
scales 2^-60/1/2^60, partial groups, 16K points and 32/256-thread blocks. Capacity
fallback and storage/association guards are included. Existing large-route tests
verify shared-array identity while exercising actual changing-input captures.

All **28 isolated memory jobs** pass with zero memcheck errors, including padded
access checks and pool-disabled lifetime checks. See the [commands and results](../results/validation/vtm/prepared-new-cell-queries/prepared-new-cell-memory-v1-commands.json).
No timeout or failed memory job is represented as a pass.

Native traces cover a warmed complete 16K solve and a device-controlled
producer/mesher/consumer loop. The latter changes target size and domain,
rejects an invalid count and collapsed input, then restores the original frame.
All three valid loop meshes pass independent checks; rejected frames publish
zero outputs/consumed volume, pointers remain stable, and recovery is exact.
The [full audit](../results/validation/vtm/prepared-new-cell-queries/prepared-new-cell-full-native-v1-audit.json)
and [loop audit](../results/validation/vtm/prepared-new-cell-queries/prepared-new-cell-loop-native-v1-audit.json)
record 69,096 and 209,224
kernel activities. DOT nodes are device kernels, conditional work, memset and
DtoD copies; no host callback, host transfer or explicit allocation/free occurs
inside the measured work. Nsight may omit graph-copy activity records, so every
DOT copy node is also checked. Each trace has one graph launch and final external
synchronization. The first loop launch includes deferred instantiation/upload
before that launch; this is not a claim of zero driver-internal host work.

The warmed full trace totals 4,888.240 ms of kernel
activity. It is diagnostic, not clean performance evidence:

| Kernel | Calls | Registers/thread | Total ms |
| --- | ---: | ---: | ---: |
| prepare_unproved | 554 | 92 | 6.789 |
| check_prepared_unproved | 554 | 255 | 463.950 |
| strict_check | 227 | 80 | 1,163.948 |
| exact_check | 227 | 255 | 16.899 |
| retained_cached | 194 | 255 | 931.632 |
| search_unproved_faces | 327 | 255 | 720.654 |
| record_face_support | 554 | 255 | 440.425 |

## Independent API and structured-grid guards

| Callable workload | Before (ms) | After (ms) | Change |
| --- | ---: | ---: | ---: |
| delaunay1000 | 9.178 | 9.184 | +0.07% |
| delaunay100000 | 180.771 | 181.643 | +0.48% |
| delaunay-grid | 2.201 | 2.201 | +0.00% |
| delaunay-fitted-hand | 13.416 | 13.714 | +2.23% |
| diagram100000 | 18.017 | 18.149 | +0.74% |

The standalone pairs use nine complete Delaunay replays and thirty diagram runs,
with matching inputs/counts/status/diagnostics and clean timing windows. Their
implementation files are unchanged; small timing differences are not attributed
to this private optimization. Source/kernel AST checks cover 149 source hashes
and verify 141 other runtime/benchmark/example files unchanged.

The primary fitted-hand pair measured a 2.23% increase with a bimodal candidate
sample set. A predeclared ABBA repeat kept every original sample and repeated
both source versions, with the second pair in reverse order:

| Follow-up pair | Before (ms) | After (ms) | Change |
| --- | ---: | ---: | ---: |
| AB pair | 13.406 | 13.363 | -0.32% |
| BA pair | 13.376 | 13.712 | +2.51% |

All follow-up diagnostics match and their timing windows are clean. The
[repeat record](../results/validation/vtm/prepared-new-cell-queries/prepared-new-cell-hand-repeat-v1/comparison.json)
retains the order and raw samples; the original primary measurement remains in
the table above. This guard uses the unchanged independent Delaunay path.

A [dependency and trace audit](../results/validation/vtm/prepared-new-cell-queries/prepared-new-cell-hand-trace-v1/nonregression-conclusion-v1.json)
finds 40 byte-identical loaded project modules and identical input fixture bytes;
neither modified runtime module is loaded by this workload. All 18 profiled full
replays have the same 143-kernel configuration signature. The slow/fast behavior
appears in the existing local-sphere checker in both versions (baseline
0.503–0.912 ms, candidate 0.512–0.894 ms). This establishes path independence and
localizes the observed variation; it does not prove a low-level cause such as row
ordering or scheduling. [#235](https://github.com/alecjacobson/warpower/issues/235)
tracks that existing variability. Profiling perturbs timings and does not replace
the clean samples above.

A 17³ retained-grid guard (4,913 sites) measures nine certified replays after
initial construction: **2.694976 → 2.698496 ms**, a +0.13% change.
Both outputs independently certify the complete hull and physical quality and
have identical coordinates/canonical cells. This is retained-mesh stress, not a
fresh VTM solve or CPU speedup comparison.

Tracked in [#233](https://github.com/alecjacobson/warpower/issues/233). Larger
active-prefix routing [#230](https://github.com/alecjacobson/warpower/issues/230),
scratch sharing [#122](https://github.com/alecjacobson/warpower/issues/122), and
Armadillo extraction failures [#228](https://github.com/alecjacobson/warpower/issues/228)
remain open. No extraction guard or quality criterion is weakened.

An [additional cross-reservation comparison](../results/validation/vtm/prepared-new-cell-queries/larger-prefix-reservation-comparison-v2.json)
finds identical validated coordinates, connectivity, boundary and solver work for
the 4,096-site target with reservations of 4,096 and 16,384. Current medians are
1,510.678 and 7,220.299 ms (4.78×). This motivates #230; it is a reservation-cost
observation, not a result of a proposed larger-prefix implementation.
