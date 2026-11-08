# VTM line-search observations

The current complete mesher spends substantial work rejecting selective updates.
This study supports a private experiment with remembered selective step scales;
it does not change the production optimizer or establish a speedup. All 267
audited production, test, benchmark, example and installation files match runtime
`7bd8542d1292512b0594a6ac5e6c75451b77aa50`. Standalone diagram and Delaunay routines,
and the README's earned Paragram result, remain unchanged.

Tracking: [observation #255](https://github.com/alecjacobson/warpower/issues/255),
[bounded step-history experiment #257](https://github.com/alecjacobson/warpower/issues/257),
[uniform-scan selection failures #258](https://github.com/alecjacobson/warpower/issues/258).
The [previous nearest-query change](surface-push-pruning.md) is already merged.

## What was measured

The original complete benchmark/gallery runs were executed in paired processes
on the NVIDIA L40 with Warp 1.19.0. The private observer adds snapshots around
original graph recording sites. At replay, one-thread Warp kernels read solver
state and write separate, preallocated trace buffers. They never modify solver
arrays or replace its kernels, arguments or acceptance checks. Host hooks choose
recording sites; replay paths and buffer writes remain device controlled.

Each fresh complete solve resets its trace. Events identify stages, core calls,
iteration starts, actual trial alpha before halving, objective acceptance,
selective guard results, local retries, finish decisions and reconstruction
entry/exit. A bounded buffer reports exhaustion explicitly. Floating-point
objective fields are unavailable outside actual step events, and other fields
are interpreted only within the relevant state lifecycle. This observer is
private instrumentation for scheduler-owned cores, not a public standalone API.

The first 1K pair and the remaining ten paired cases retain their original replay
counts. Final warmed outputs and work are compared to uninstrumented references.
Recorded instrumented times are **not performance results**. Initialization,
refinement, jitter, interior moves and recovery reconstructions are counted
separately from line-search trials.

## Observed work

The last two columns count rejected prefixes on the *observed trajectory* that
precede a proposed next starting scale. History is restricted to the same core
invocation. "Same" uses the preceding accepted alpha; "twice" uses
`min(0.5, 2 * previous_accepted_alpha)`. When a proposed scale lies below the
current observed successful scale, no prefix saving is counted for that step.

| Complete case | Reconstructions | Trials | Accepted | Rejected | Prefix: same | Prefix: twice |
|---|---:|---:|---:|---:|---:|---:|
| Torus 1K | 141 | 115 | 48 | 67 | 16 | 12 |
| Torus 8K, fixed 32 global iterations | 134 | 120 | 39 | 81 | 30 | 25 |
| Torus 16K, fixed 32 global iterations | 227 | 214 | 57 | 157 | 121 | 99 |
| Torus 8K, progressive | 249 | 218 | 77 | 141 | 93 | 83 |
| Thin hand, seed 7 | 190 | 167 | 119 | 48 | 13 | 9 |
| Cavity | 146 | 122 | 48 | 74 | 24 | 20 |
| Disconnected solids | 113 | 92 | 40 | 52 | 0 | 0 |
| Counterbored plate | 178 | 157 | 51 | 106 | 53 | 45 |
| Graded scan 4K | 214 | 182 | 155 | 27 | 16 | 9 |
| Graded scan 8K | 269 | 233 | 141 | 92 | 50 | 38 |
| Uniform scan 8K | 242 | 185 | 128 | 57 | 5 | 3 |

Every observed global trial accepts its initial 0.5 scale. All rejections occur
in selective mode. Except for the uniform scan, the rejected trials fail the
minimum-quality guard (reason bit 4). The uniform scan has selective error 16,
zero retained count and reason bits 3 on all 57 rejections. That code does not
identify the original failure: recovered selection also adds bit 16 when
extraction is invalid. Issue #258 will inspect the underlying diagnostics and
geometry before proposing a remedy.

Disconnected solids accept no selective trials, so within-call success history
offers no saving. The plate shows one case where unchanged scale reuse would
start below the observed successful scale. The graded 4K scan shows two such
cases even with doubling: a prior accepted scale of 0.03125 is followed by
successes at 0.125 and 0.25. Quadrupling still misses the latter. These are
explicit counterexamples to assuming that previously small steps remain best.

The counts are opportunity bounds, not predicted saved milliseconds. Trial
costs differ, and skipping rejected trials can change Delaunay reuse history,
rounded reductions, later acceptance and final quality. Neither rejection
monotonicity nor unchanged future trajectories is established.

## Geometry and graph qualification

All 22 final meshes pass independent geometry qualification: 19 exact proof
transfers and three fresh validations. Material cell/boundary connectivity,
site roles, input geometry, work counts and minimum material quality match.
The paired hand coordinates differ in four scalar entries by at most
`1.0408340855860843e-17`; disconnected solids differ in one entry by
`4.656612873077393e-10`. These variations are retained, without assigning a
causal kernel.

The observed hand also replaces four full-hull cells outside the material
domain. The changed full hull passes exact original-coordinate orientation,
global empty-sphere and hull-coverage checks. Material topology, boundary vertex
links, and exact boundary self-intersection checks pass. It is explicitly **not**
an identical full-hull result. Exact transfers are never used across these
differences; witnesses and fresh proofs are included.

Nine unchanged complete-mesher contracts pass with the observer installed,
including changing device-produced inputs, bounded failures, recovery, rollback,
and an outer device-controlled loop. Existing tests forbid host allocation,
readback and synchronization during their captured scope. The optional local
rollback tests record 10 and 11 local retries, separately from alpha halving.

Four isolated padded memory jobs pass with zero errors: two buffer-exhaustion
tests, the changing-input producer/consumer fixture, and the two local-retry
fixtures. Their wall times are 2.12, 382.21, 72.06 and 74.81 seconds. These are
instrumentation costs, not meshing performance. The first batch omitted the
established explicit stack setup and was stopped incomplete; the configured
multi-workspace batch also stopped in setup synchronization. An isolated
producer attempt reached its 300-second deadline. The final full-workspace
jobs use a 1,200-second allowance and the existing 128 KiB diagnostic stack.
All incomplete runs and stack traces are retained. No root cause is assigned;
the synchronization behavior is related tracking work in
[issue #155](https://github.com/alecjacobson/warpower/issues/155).

The initial exact-only analysis stopped on a nonbitwise hand reference rerun.
A later analysis stopped on the changed full-hull cells. Both partial reports
are retained; the final analysis requires fresh certification rather than
claiming exact equivalence. An initial validation invocation omitted
`PYTHONPATH`; its import error and corrected invocation log are also retained.

## Next experiment and evidence

Issue #257 will test selective-only, bounded scale history privately. It must
retain original geometry, feature, material-quality and rollback guards, reset
history on appropriate device state transitions, and preserve bounded failure.
A smaller starting scale must not cause false convergence through the existing
actual-displacement tolerance: an explicit default-scale check or a justified
equivalent is required before such termination. Adoption requires complete
time-to-quality and geometry comparisons, native capture, memory/lifetime and
standalone nonregression checks.

The [artifact manifest](../results/validation/vtm/step-observation/manifest.json)
binds raw traces, paired outputs, scripts, partial failures, source hashes and
validation. The [summary](../results/validation/vtm/step-observation/step-observer-summary-v1.json)
includes stage attribution, exhausted core calls, scale distributions and the
understep witnesses. The wide comparable-quality CPU speedup goal remains open.
