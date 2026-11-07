# Reusing current hull-support results

This page records the preceding exhaustive support-mask reuse stage. The current
[exact exterior-witness shortcut](hull-witness-stop.md) additionally stops work
once support of a face has been disproved.

VTM's private fragment completion globally checks every boundary face against the
current sites. A supported face has no site on its outward side, so the following
facet search cannot produce a new tetrahedron there. The private certificate now
records this result and skips that redundant search. Unsupported faces retain the
complete original candidate search and symbolic sphere comparisons.

The mask belongs to one current boundary ordering. Before each hull check, every
slot is initialized to supported. Any positive exact orientation clears the slot
atomically. The original traversal still visits every live radix-tree piece and
strided face slot, and keeps all predicate/stack error reporting. A completion
round starts only if the preceding certificate has no geometric errors; incomplete
traversals therefore cannot turn initial mask values into trusted proofs.

No face proof survives into a new certificate or input frame. Newly exposed faces
receive a fresh global check. Cell orientation, empty-sphere, incidence, coverage
and final full-hull certification remain in place. This complements
[same-input cell proofs](completion-proofs.md); it is separate from cross-frame
proof caching in [issue #130](https://github.com/alecjacobson/warpower/issues/130).

Two host dispatch hooks leave the ordinary certificate and facet kernels unchanged.
Only VTM's private coordinator uses the new kernels. There is no public mask or
trust switch. Added storage is four integers per cell-capacity slot plus two
integer search counters: 128,008 bytes at 1,000-site capacity and 393,232 bytes across
the smaller constructors reserved by the 4,096-site mesher. All storage is allocated
before capture. These are persistent workspace pool deltas, not total process
memory peaks; per-thread kernel temporaries are separate. The counters accumulate
supported faces skipped and unsupported faces searched across one completion;
they are not counts of predicate calls.

## Validation

The full suite passes 564 tests. Eleven new cases compare masks with independent
rational orientation determinants and compare every candidate lane with the
unmodified search. They include coplanar hull facets, duplicate sites, scales
2^-60/1/2^60, early tree leaves, strided face tails, padded output slots, each
round's changed boundary ordering, nonfinite predicates, exhausted traversal
stacks and failure/restoration. Existing exact symbolic cell-proof tests and
full global certificates also pass. AST comparisons confirm that geometric kernel
bodies are unchanged after removing the mask write, skip and diagnostic counters.

All 45 focused cases pass unfiltered padded memory checking in four independent
processes, with zero errors. A separate six-phase allocation/lifetime audit with
memory pooling disabled also reports zero errors. The moved-input phase skips
89 supported-face searches and performs 223 unsupported-face searches.
The nearby complete fragment needs no candidate search. Fresh, invalid and restored
frames do not report stale private counters. See the
[memory record](../results/validation/vtm/support-reuse-memory-summary.json) and
[resident records](../results/validation/vtm/support-reuse-replay-proof-counts.json).

All eight saved full hulls pass independent CGAL exact orientation, global
empty-sphere, face-pairing and hull-support checks. NumPy checks material topology
and quality; six gallery VTU round trips reproduce the saved arrays. All paired
parameter/iteration counts, full/material cell sets, roles and oriented boundaries
agree. Five pairs preserve coordinates bitwise; the hand differs near zero by
at most `1.09e-18`. Details are retained in the
[comparison record](../results/validation/vtm/support-reuse-summary.json).

## Complete-solve measurements

Baseline `4582115` and runtime `c961661` were measured on NVIDIA L40 / Warp 1.19.0.
Each median uses three warm, complete fresh resident solves, including domain
refit, sampling, growth, fitting, four quality rounds and extraction. Setup,
transfers, export and independent validation are outside replay timing. All
first/warm timing windows have zero CPU quota throttling.

| Complete solve | Before (ms) | After (ms) | Speedup |
| --- | ---: | ---: | ---: |
| Torus, 1,000 sites | 1,206.828 | 1,190.348 | 1.014× |
| Torus, 4,096 sites | 7,805.976 | 7,800.038 | 1.001× |
| Graded torus, 1,000 sites | 1,997.018 | 1,966.368 | 1.016× |
| Thin hand, quadrature 20/20 | 2,622.005 | 2,613.931 | 1.003× |
| Cavity, quadrature 20/20 | 1,139.624 | 1,131.886 | 1.007× |
| Disconnected, quadrature 20/20 | 898.271 | 896.070 | 1.002× |

These are incremental savings, with little change at 4,096 sites where most work
uses the larger ordinary constructor. The records include setup, storage deltas,
raw samples and quality. Coarse quadrature-4 cavity and disconnected recovery
examples also reproduce their preceding repaired meshes.

Across three profiled complete torus1000 solves, candidate search time changes
from 389.415 ms to 344.107 ms. Original hull support time is
580.188 ms; the new recording check plus remaining ordinary support checks
take 578.116 ms. Profile overhead applies; the table uses separate clean
measurements. The [kernel profile](../results/validation/vtm/support-reuse-kernels.json)
retains all work, including proof bookkeeping. Bookkeeping increases kernel
activity count from 97,404 to 98,775; its cost is included in complete-solve timing.

The [native capture audit](../results/validation/vtm/support-reuse-device-audit.json)
records 98,775 kernel activities, three graph launches and one final
synchronization. Its graph contains 10,312 kernel nodes, 3,497 memsets,
1,234 conditionals and 919 copies, all device-to-device. There are no
host transfers, callbacks, allocations or frees during replay. Nsight omits most
graph copy activities, so every DOT copy node is checked as well.

## Independent routines

Standalone paired gates use nine Delaunay repeats and fifteen diagram repeats:

| Operation | Before (ms) | After (ms) |
| --- | ---: | ---: |
| Delaunay, 1,000 random sites | 9.194 | 9.178 |
| Delaunay, 100,000 random sites | 181.423 | 182.658 |
| Delaunay, 8³ grid | 3.943 | 3.953 |
| Delaunay, 256 fitted-hand sites | 13.421 | 13.406 |
| Diagram, 100,000 uniform sites | 18.167 | 18.172 |

Standalone recovery graphs contain none of the private proof kernels and keep
identical facet launch dimensions. These small timing differences are treated as
variation; no standalone gain is attributed to this change. The full CPU
outperformance target remains unmet, and README retains its separately earned
Paragram diagram result. The [manifest](../results/validation/vtm/support-reuse-provenance.json)
pins source versions and every published artifact hash.
