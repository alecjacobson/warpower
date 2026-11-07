# Surface-query work and bounded arithmetic probes

The remaining full-distance surface queries perform substantial repeated box
arithmetic. This investigation measures that work and compares three private
nearest-query prototypes. The float32 filter and tail traversal remain
experimental. Production validation for the simpler exact reuse of child
distances is reported separately in [surface-bound reuse](surface-bound-reuse.md).

All probes here ran on merged baseline `18ecb03c10cb10e42a5089e8253ed92a24c3bae1`,
NVIDIA L40 and Warp 1.19. They must be reproduced against that baseline: their
reference imports resolve the current implementation at execution time. These
records do not establish a comparable-quality CPU speedup.

## Where the work occurs

An instrumented copy of the original traversal counts operations and compares
its results with the original query. On final-mesh circumcenter snapshots, all
nearest points, squared distances, face indices and ray results match. The
retained-query column evaluates the ordinary bounded-query rule on these same
centers; it is not a measurement of the actual solver's retained-call traffic.

| Snapshot | Full nearest boxes / triangles | Bounded retained boxes / triangles | Rays boxes / triangles |
|---|---:|---:|---:|
| torus 1k | 279.899 / 138.398 | 15.091 / 1.173 | 50.951 / 26.373 |
| torus 4096 | 246.903 / 119.956 | 11.292 / 0.837 | 53.692 / 28.025 |
| Armadillo uniform | 951.726 / 308.901 | 19.458 / 1.178 | 166.936 / 48.728 |
| Armadillo graded | 1022.015 / 328.730 | 18.924 / 1.125 | 186.958 / 54.608 |

Values are means per input query, including zero-work retained queries. Ray
counts include the conditional retries in the original three-direction rule.
Operation counts are not elapsed-time attribution.

A separate observer runs after the original full classifier during two complete
graded Armadillo solves. Every production kernel still executes. Each replay
contains 39 full classification calls and 962,610 observed queries, with zero
query/reference or actual classification mismatches. Positions are bitwise
equal to the saved baseline; topology, roles and solver work also match.

Each replay counts 983,785,762 nearest-box evaluations and 316,418,958 nearest
triangle evaluations, plus 180,056,830 ray-box and 52,594,600 ray-triangle
evaluations. The maximum nearest stack occupancy is 22 on this trajectory.
The two replays have identical counters. This observation excludes private
recovery classification, ordinary retained classification and surface-summary
queries; no instrumented timing is used as a speed measurement.

## Conservative float32 prototype

For finite query and box coordinates with absolute values at most 16, the
prototype computes a float32 squared box distance and an error interval. Its
scale is the maximum absolute float32 query/box coordinate and `2^-30`; its
stored error is `(B * B) * 2^-16`. Pruning and child ordering use the interval
only when it establishes the original float64 comparison. Overlap, unsupported
scales and nonfinite data use the original float64 computation.

A rational-arithmetic derivation bounds the difference between the two rounded
calculations by less than `2^-17 * B^2`, including final interval-endpoint
rounding. It assumes IEEE round-to-nearest errors with unit roundoffs `2^-24`
and `2^-53`, conservative absolute underflow terms `2^-126` and `2^-1022`, and a
five-operation dot-product bound that covers FMA contraction. This is a proof
under that arithmetic model, not verification of a compiler or the whole mesher.

The 38 prototype tests pass: primitive interval/pruning/ordering checks over
32,768 boxes and six threshold sets with FMA enabled and disabled, plus 36
full-nearest shape/scale/FMA cases with six search limits. The cases include
boundary points, adjacent representable values, unsupported scales and invalid
inputs. These finite tests supplement the arithmetic argument.

## Static nearest-query timings

Each experiment uses three warmups and 20 interleaved timed samples on identical
resident inputs. All recorded quota windows are clean. Nearest point, distance
and face outputs match exactly. Each candidate has its own contemporaneous
reference; these are isolated query timings, excluding geometry preparation,
setup and full solver execution.

| Input | Float32 filter: reference → candidate ms | Exact reuse: reference → candidate ms | Tail traversal: reference → candidate ms |
|---|---:|---:|---:|
| torus 1k | 3.508 → 3.539 | 3.501 → 3.352 | 3.497 → 3.338 |
| torus 4096 | 7.734 → 7.018 | 7.737 → 7.262 | 7.737 → 7.253 |
| Armadillo uniform | 17.123 → 14.564 | 17.121 → 15.497 | 17.123 → 15.414 |
| Armadillo graded | 18.484 → 15.315 | 18.484 → 16.342 | 18.484 → 16.258 |

The float32 filter slows the smallest case and needs complete-solve, capture and
memory validation before adoption. Its progress remains tracked in
[#198](https://github.com/alecjacobson/warpower/issues/198).

Exact reuse stores the already computed child distances alongside the original
node stack. The tail variant additionally keeps the next child in local scalar
variables and pushes only the deferred child. The tail variant passes 46 tests,
including extreme/nonfinite bounds and limits, but offers only a small additional
isolated gain and changes more control flow. It was not selected for production.
The simpler exact-reuse change is tracked in
[#200](https://github.com/alecjacobson/warpower/issues/200).

The [artifact manifest](../results/validation/vtm/surface-bound-reuse/manifest.json)
links the raw counters, samples, rational derivation, probe sources and logs.
Earlier cooperative ray experiments are documented in
[the cooperative-surface probe](cooperative-surface-probe.md).
