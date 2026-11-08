# Certified maximum-corner hull bounds

Private VTM completion uses the existing certified plane normal to select the
corner that maximizes an oriented plane over a BVH box. The existing bounded
binary32 orientation filter can then decide whether to skip that box or traverse
it. Uncertain results retain the original Ball64 box test and exact site checks.
This optimization is tracked in [issue #216](https://github.com/alecjacobson/warpower/issues/216).

## Proof and device execution

For original binary32 face vertices `a,b,c`, the existing `halfspace` function
encloses each component of the exact real normal `n=(b-a)×(c-a)` in a Ball64
interval `[v-e,v+e]`. Finite bounds with `e>=0` and `v>=e` prove the component is
nonnegative; `v<=-e` proves it is nonpositive. If any component is uncertain, the
shortcut is unavailable. An exactly zero component may use either endpoint.

For an ordered box `[lo,hi]`, select `q[i]=hi[i]` for a certified nonnegative
normal component and `q[i]=lo[i]` otherwise. Then every real point `x` in the box
satisfies `n·(x-a)<=n·(q-a)`. The corner consists of stored binary32 endpoints;
there is no rounded box-center or extent calculation.

The existing bounded orientation filter encloses the exact value `n·(q-a)`.
Its strict signs give three possible actions:

| Filter result | Box action | Face-support effect |
|---|---|---|
| Strictly negative | Skip the node | Every contained site lies inside this halfspace. |
| Strictly positive | Traverse its children or sites without the old box calculation | The box cannot be wholly inside; this does **not** reject face support. |
| Uncertain or unsupported | Use the original interval box test | Original traversal and exact fallback remain. |

A positive corner need not be an actual site. It only proves that the sound
interval containment predicate cannot accept the whole box. Only an existing
exact exterior-site witness may clear a face-support flag. Exact coplanarity
retains the old box predicate's behavior, including non-strict inclusion.

Both box endpoints must satisfy the existing finite coordinate-magnitude limit
of 16, and their componentwise order is checked. The orientation filter also
checks the face/corner range. Its existing roundoff and underflow/FTZ proof is
unchanged; `benchmarks/bounded_predicate_proof.py` verifies the same expression
tree, constants and error envelope. There is no new roundoff formula or
tolerance-based sign. All calling modules disable fast math and contraction.

The corner selector is a local per-face value. No persistent buffers, graph
nodes, host decisions or cross-frame proofs are added. The caller still uses
current original sites, a current valid hierarchy and freshly certified positive
cells. The [coordinate-plane shortcut](axis-hull-support.md) runs first.
Independently callable power-diagram and Delaunay paths retain their defaults.

## Investigation and retained attempts

A warmed complete torus4096 profile at baseline `c74e492` attributes 292.565 ms
to 552 private support calls, out of 1,786.208 ms of kernel activity. The 4k
workspace accounts for 410 calls and 236.551 ms. Full and retained surface
classification together account for 478.414 ms. These are diagnostic costs,
not a clean throughput comparison. The profiled positions, vertex roles,
canonical full/material/oriented-boundary connectivity, aligned radius ratios
and recorded work match the independently validated preceding final result.

Traversal instrumentation observes about 204.60 million box tests in the 4k
torus workspace across the first solve and one fresh replay. About 103.58 million
are accepted by the original box predicate. The first prototype proves virtually
all of those acceptances but still runs the old predicate for the remaining
boxes. Its complete torus pilot improves only 0.71%, and hand is neutral. All
four pilot outputs independently validate, but this prototype remains unadopted.
The second version adds the certified box-not-contained branch described above.

The first Armadillo observer accidentally used generic gallery settings. It
rejects initialization before any hull call, with error 1 and zero counts. That
attempt is retained; a separate rerun uses the established feature angle 120
and 1/1 quadrature and succeeds. All observed node coordinates lie in the
supported filter range, and no observed traversal overflows its stack.

The second prototype passes 3,065 captured cases against exact rational tests
of all eight corners: 1,259 inside certificates and 656 box-not-contained
certificates, with no incorrect decisions or normal-component signs. Forty-eight
nonfinite/reversed cases remain unsupported. Existing support, motion, proof,
reuse and wall reconstruction tests also pass. Duplicate-decision audits compare
3,659,458 face masks across 3,775 calls on hand, plate, torus and graded Armadillo,
including smaller prefix workspaces, with no support-mask or error differences.

The source audit verifies that the production helpers and support kernel have
the same AST as the measured second prototype; other baseline definitions are
unchanged. The complete production suite passes 1,103 tests.

## Pinned complete-solve and standalone results

Baseline `c74e492` and production `fec4c08` use separate persistent checkouts on
the NVIDIA L40 with Warp 1.19.0. All 36 jobs have clean first and timed CPU-quota
windows. Complete-solve values are medians of three additional fresh warmed
replays, including reconstruction, refinement, fitting and quality work. These
are changes from the preceding GPU version, not CPU speedups.

| Case | Before (ms) | After (ms) | Time change |
|---|---:|---:|---:|
| hand-seed0 | 1207.652 | 1204.506 | -0.26% |
| hand-seed1 | 1037.906 | 1036.915 | -0.10% |
| hand-seed2 | 1816.620 | 1830.839 | +0.78% |
| hand-seed7 | 1295.007 | 1308.516 | +1.04% |
| hand-seed42 | 875.675 | 873.027 | -0.30% |
| hand-seed2026 | 1329.409 | 1334.832 | +0.41% |
| plate | 968.822 | 968.501 | -0.03% |
| coarse-cavity | 460.434 | 460.076 | -0.08% |
| coarse-disconnected | 729.495 | 727.663 | -0.25% |
| torus4096 | 1787.575 | 1687.552 | -5.60% |
| graded-armadillo | 3473.336 | 3455.625 | -0.51% |
| hand-seed7-local4 | 1237.128 | 1233.448 | -0.30% |
| Standalone delaunay1000 | 9.219 | 9.178 | -0.44% |
| Standalone delaunay100000 | 182.442 | 183.137 | +0.38% |
| Standalone delaunay-grid | 2.205 | 2.204 | -0.07% |
| Standalone delaunay-fitted-hand | 13.426 | 13.391 | -0.26% |
| Standalone diagram100000 | 18.811 | 18.202 | -3.24% |

The torus improves by 5.60%; most other complete solves are approximately flat.
Hand seed 7 is 1.04% slower in this sample. The largest standalone slowdown is
0.38%; the faster unchanged diagram control is not attributable to this private
filter. These samples do not establish a universal no-regression guarantee.
Coarse hand is explicitly invalid in both versions (error 64, zero published
counts) and excluded from valid-mesh speed comparisons.

All 24 valid final meshes independently pass original-coordinate CGAL full-hull,
orientation and empty-sphere checks, material quality and boundary topology,
single-cycle vertex links, exact VTU round trips and zero boundary
self-intersection pairs. Dense hand and plate also retain feature-distance
samples. All twelve valid pairs have matching roles, material cells, oriented
material boundary and recorded counts/rounds/steps/moves/rejections. Six pairs
have bitwise-equal coordinates; the other six are hand seeds with changes up to
5.820766091346741e-11 (seed 42).

The first strict checker stops at hand seed 7 because its **exterior full-hull
connectivity differs**: eight cells are removed and eight added after coordinate
changes of at most 3.469446951953614e-18. Both own-frame hulls are independently
valid; swapping the before/after connectivity makes both fail the exact
empty-sphere check. The retained audit records every changed cell and exact
positive rational volume. Material connectivity, oriented boundary and work
remain equal; the largest aligned material radius-ratio difference is
1.1102230246251565e-16. Fit/volume/mean quality agree within the explicit audit
limits (`rtol=1e-11`, `atol=1e-14`). No changed full cell belongs to the material.
The original failed checker/report/log are preserved. A second checker reuses
its hash-bound successful eight-mesh prefix, applies this explicit geometric
assessment and validates the remaining outputs. It does not claim identical
full connectivity or trace the variation to a particular kernel.

## Memory and native execution

All eight isolated sanitizer jobs pass with zero errors: new/axis primitives,
support and motion cases, a full hand seed-7 solve with four local retries,
and a captured four-phase failure/recovery loop, each with padded allocation
and pool-disabled lifetime instrumentation. The existing diagnostic 131,072-byte
stack allowance is confined to these checks. Production timings and native
traces use normal configuration; no tests time out or are excluded.

The full native trace records 57,744 kernel activities and the four-phase loop
records 149,313. All 1,466 copy nodes in each DOT graph are device-to-device.
There are no HOST, allocation or free nodes, nor allocation/free/memcpy API calls
inside either recorded execution. Each has one external final synchronization;
first graph instantiation/upload is visible. Setup and final host readback are
outside the region. Nsight may omit graph copy activities, so DOT directions
are checked independently.

The loop records errors `[0,1,0,0]`, valid flags `[1,0,1,1]` and proof-enabled
flags `[1,0,0,1]`. Its deliberately invalid phase publishes zero counts, and
its final healthy phase restores certified exterior removal and accepted
representable midpoint moves. These are execution/capture checks, not clean
throughput measurements.

## Evidence and scope

The [evidence manifest](../results/validation/vtm/bounded-hull-box/manifest.json)
binds pinned source hashes, primitive rational oracles, original and candidate
kernels, traversal/decision observations, neutral first prototype, later pilots,
all final timings and exports, the failed strict comparison and follow-up audit,
memory logs and lossless native reports/DOT graphs. Reproduction instructions
record helper paths, shared input hashes, environment and diagnostic scope.
The [verification supplement](../results/validation/vtm/bounded-hull-box/verification-supplement-v2.json)
preserves the first verifier's archive-schema error and supplies version 2,
which checks both retained archive schemas against the unchanged manifest.


Independent distance and feature measurements use samples; they are not
Hausdorff or exact constrained-edge conformity certificates. Hand coordinate
variation is retained without claiming bitwise reproducibility or a traced
causal kernel; [issue #208](https://github.com/alecjacobson/warpower/issues/208)
remains open. Coarse hand and the wider feature-fitting work remain tracked
separately. The wide comparable-quality native CPU speedup target remains unmet,
and the README retains its earned Paragram flagship result.
