# Larger device-selected active prefixes

VTM now preallocates fitting 4,096- and 8,192-site constructors alongside its
existing 1,024/2,048 routes and full constructor. The device active count selects
the smallest fitting branch. A 16K reservation can therefore run a 4K target
through the 4K constructor while keeping stable public outputs for later growth.
The tradeoff is additional persistent memory and setup; each child has separate
storage. This is not the scratch-sharing work in [#122](https://github.com/alecjacobson/warpower/issues/122).

## Complete solves and costs

Production `670bd3b` is compared with `df74493` on NVIDIA L40 / Warp 1.19.0.
Nine fresh interleaved complete-solve pairs record three measured replays each,
including domain refit, fresh sampling, fitting, refinement/quality and extraction.
Setup, compilation, upload and offline checks are excluded from replay. The
[primary comparison](../results/validation/vtm/larger-active-prefixes/larger-prefix-final-v1-independent.json) preserves
samples, setup, first replay, workspace memory and independent outputs. Every
initial and measured primary timing window passes the zero-CPU-quota-throttling
gate. These are gains over the preceding GPU implementation; the
[wide comparable-quality CPU goal](fixed-site-current.md) remains unmet.

| Complete solve | Before ms | After ms | Before / after |
| --- | ---: | ---: | ---: |
| Torus target 4,096; reserved 16,384 | 7,223.438 | 1,545.766 | 4.673× |
| Torus target 4,096; reserved 8,192 | 3,260.200 | 1,531.204 | 2.129× |
| Torus target 8,192; reserved 16,384 | 8,339.644 | 2,925.402 | 2.851× |
| Torus 8,192; progressive | 3,088.004 | 2,900.250 | 1.065× |
| Graded torus 16,384; progressive | 6,975.787 | 5,733.512 | 1.217× |
| Torus 16,384; fixed 32 | 4,881.551 | 4,890.743 | 0.998× |
| Torus 8,192; fixed 32 | 1,496.812 | 1,497.411 | 1.000× |
| Torus 4,096 | 1,508.215 | 1,510.186 | 0.999× |
| Torus 1,000 | 512.685 | 514.283 | 0.997× |

All eighteen successful saved meshes independently pass original-coordinate
full-hull/orientation/empty-sphere, material quality/topology, boundary vertex-link
and boundary self-intersection checks. Graded exports round-trip. Every pair has
bitwise-identical coordinates, canonical full/material cells, oriented boundary,
roles and solver work. Routing counts change intentionally and remain in the raw
JSON; reduced constructor capacity does not mean fewer optimizer iterations.

A [cross-reservation check](../results/validation/vtm/larger-active-prefixes/larger-prefix-current-reservation-comparison-v1.json)
also finds identical geometry and work across the saved post-change reservations.
Using the primary medians, a 4K target reserved at 16K records
1.024× its dedicated 4K time; the 8K target reserved
at 16K records 1.009× its dedicated 8K time. These reuse
the existing samples; no separate interleaved cross-reservation timing experiment
was performed.

| Solve | Before pool bytes | After pool bytes | Setup seconds |
| --- | ---: | ---: | ---: |
| Torus target 4,096; reserved 16,384 | 2,551,578,988 | 4,404,748,550 | 3.952 → 5.174 |
| Torus 8,192; progressive | 2,044,662,012 | 2,939,146,889 | 3.942 → 4.539 |
| Torus 16,384; fixed 32 | 2,538,128,443 | 4,391,298,005 | 3.495 → 4.262 |

These are persistent workspace pool bytes, not process peak memory. Both extra
children add **1,853,169,562 bytes** at reservation 16K; the 4K child adds
**894,484,877 bytes** at reservation 8K in these configurations. Even the fixed 16K
case pays for unused children because a device-controlled target may change in
a later graph invocation. Standalone diagram/Delaunay construction allocates
neither additional child. README retains its earned Paragram flagship.

## Prefix proof and ownership

VTM first copies active coordinates to separate mesh scratch and fills every
inactive slot with an exact copy of site zero. Taking a fitting prefix removes
only later duplicates: every distinct active position, canonical first owner
and original site ID remains. Each selected constructor still certifies the
complete current active convex hull with the existing original-coordinate and
symbolic predicates. [The proof note](../results/validation/vtm/larger-active-prefixes/larger-prefix-proof-v1.md) states
these prerequisites; arbitrary unprepared buffers do not satisfy the contract.

The [VTM owner](../warpower/_vtm.py) supplies explicit increasing levels filtered
below its full reservation. The [private router](../warpower/_active_delaunay.py)
keeps its direct default of 1,024/2,048 and accepts explicit levels through 8,192.
The full constructor remains output owner and fallback. Every child has separate
retained-fragment storage; public result copies are DtoD. A fresh solve clears
fragments/preferences for every route. Reuse always re-establishes current-input
proofs after movement, growth, shrinkage, rejection or unrelated input.

All constructors, views and conditional graphs exist before capture. Selection,
execution, result copies and fallback stay on device. No new GPU kernel,
predicate, certificate, reset, publish or fallback method is introduced. An
[AST/source audit](../results/validation/vtm/larger-active-prefixes/larger-prefix-production-source-audit-v2.json) binds the
production setup to the measured private prototype, verifies 148 source hashes
and checks 141 other runtime/benchmark/example files unchanged.

This proof concerns geometric equivalence. Existing parallel cell ordering and
floating reductions do not guarantee identical VTM trajectories on every possible
input. Strict equal-output/work checks are reported for the measured cases;
independent validity is checked separately.

## Tests and captured execution

The full suite passes **1,292 tests** and the focused routing suite passes
**47 tests**. Thirteen new cases cover captured 4,095/4,096/4,097 and
8,191/8,192/8,193 thresholds, growth/shrinkage, unrelated inputs, invalid counts,
nonfinite/flat data, recovery, exact ties, duplicate first owners at sparse
indices, one-ULP changes, scales 2^-60/1/2^60, exhausted outputs, forced child
failures and the non-reusing full route above 16K. Capture tests forbid host
allocation/readback/synchronization and use a non-default stream.

All **28 isolated memory jobs** pass padded-access and pool-disabled lifetime
checks with zero memcheck errors. These diagnostic processes use a 131,072-byte
CUDA stack for instrumentation; production execution settings are unchanged.
[Commands and logs](../results/validation/vtm/larger-active-prefixes/larger-prefix-memory-v1-commands.json)
retain the individual results and 600-second process limits. Periodic 60-second
Python stack dumps are diagnostic snapshots, not termination or completion.
A terminated or failed job is never counted as a pass.

Native traces cover a warmed complete fixed 16K solve and a device-controlled
producer/mesher/consumer loop with reservation 16K. The loop runs targets 4K and 8K
with a domain deformation, rejects target 16,385 and a collapsed domain, then
recovers the original 4K solve. All three valid frames independently validate,
invalid frames publish zero outputs/consumed volume, output pointers stay stable,
and recovery matches exactly.

The [full audit](../results/validation/vtm/larger-active-prefixes/larger-prefix-full-native-v1-audit.json) and
[loop audit](../results/validation/vtm/larger-active-prefixes/larger-prefix-loop-native-v1-audit.json) record
69,556 and 223,338 kernel
activities. Captured nodes are device kernels, conditional work, memset and DtoD
copies. No host transfer, callback or explicit allocation/free occurs inside
measured execution. Every DOT copy node is checked because Nsight can omit copy
activity records. Each trace has one graph launch and external final
synchronization. The first loop launch can include deferred graph instantiation
and upload before that launch; this is not a claim of zero driver-internal host
work. Native/sanitizer timings are diagnostic rather than clean speed samples.

## Standalone APIs and remaining failures

| Callable workload | Before ms | After ms | Before / after |
| --- | ---: | ---: | ---: |
| delaunay1000 | 9.182 | 9.183 | 1.000× |
| delaunay100000 | 182.243 | 183.299 | 0.994× |
| delaunay-grid | 2.201 | 2.191 | 1.004× |
| delaunay-fitted-hand | 13.412 | 13.425 | 0.999× |
| diagram100000 | 18.084 | 18.095 | 0.999× |

The standalone pairs record nine complete Delaunay replays and thirty diagram
runs, with matching inputs/counts/status/diagnostics. Their construction paths
are unchanged. Existing fitted-hand timing variability remains tracked in
[#235](https://github.com/alecjacobson/warpower/issues/235); no samples are dropped.
A retained 17³ grid guard measures **2.688320 → 2.685024 ms** after
initial construction. Both outputs independently certify the complete hull and
physical quality and have identical canonical cells. This is retained-mesh
stress rather than a fresh VTM solve or CPU comparison.

Two additional fresh pairs retain the known uniform/graded 8K Armadillo rejection
cases. Both versions report error 64 with zero published outputs and matching
failed state/work. Their timings are not successful-mesh speedups. The extraction
problem remains [#228](https://github.com/alecjacobson/warpower/issues/228); no
quality or topology guard is weakened.

The [private decision](../results/validation/vtm/larger-active-prefixes/larger-prefix-adoption-decision-v1.md) records the
earlier four clean pilot pairs, all eight independently validated pilot meshes,
twelve private correctness cases and the explicit memory/setup tradeoff.
[The manifest](../results/validation/vtm/larger-active-prefixes/manifest.json) links exact sources, raw results, proofs,
independent validators, native traces and memory logs. Tracked in
[#230](https://github.com/alecjacobson/warpower/issues/230), supporting
[#29](https://github.com/alecjacobson/warpower/issues/29),
[#40](https://github.com/alecjacobson/warpower/issues/40) and
[#43](https://github.com/alecjacobson/warpower/issues/43).
