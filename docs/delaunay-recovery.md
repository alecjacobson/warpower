# Certified Delaunay construction

`DelaunayWorkspace` is an optional, independently callable ordinary Delaunay
constructor. It certifies connectivity against the original float32 sites and
recovers some configurations rejected by the existing floating-point builder,
including cube corners and regular lattices. Input positions are never moved.
The existing `delaunay_tetrahedra`, `regular_tetrahedra`, diagram functions, and
`DiagramWorkspace` retain their implementations and performance characteristics.

```python
import warp as wp
from warpower import DelaunayWorkspace

mesh = DelaunayWorkspace(sites, tetrahedra_cell_capacity=1024)
with wp.ScopedCapture() as capture:
    produce_sites(mesh.points)  # Your Warp launches, writing the shared buffer.
    mesh.run()
    consume_mesh(mesh.points, mesh.tetrahedra, mesh.tetrahedra_count, mesh.error)
wp.capture_launch(capture.graph)
```

Consumer kernels must check `error[0] == 0` and use only the prefix delimited by
`tetrahedra_count[0]`. These are device arrays. The example's producer/consumer
functions stand for application kernel launches, not host copies. Setup allocates,
compiles, captures, and synchronizes; `run(stream=None)` and captured replay use
fixed storage and device decisions. Establish stream dependencies when sharing
inputs with other processes. Warp scoped capture is required for conditional
nodes; see [graph execution](graphs.md).

Inputs are at least four sites in a contiguous CUDA Warp/PyTorch float32 `(N,3)`
buffer, or NumPy data uploaded once. The site count is fixed at setup. All active
positions must be finite. Coincident positions, including signed zeros, share
the lowest site index. A full-dimensional result uses every distinct site.
This API does not accept additive power weights; the independent regular-mesh
APIs remain available with their existing numerical contract.

## Construction and certification

The first replay starts with the existing unweighted full-hull constructor. Both its
successful outputs and every recovery candidate undergo the new certificate.
Recovery runs the regular builder on a normalized copy with small auxiliary
weights. The default starts at `1e-5 * h(i)`, for a fixed float32 hash direction
`h`, and divides the amplitude by ten on each failed attempt. There are six
attempts by default (`max_attempts` accepts 1 through 12). Neither normalization
roundoff nor finite auxiliary weights are assumed to preserve the desired mesh.
Only the original-coordinate certificate can accept a candidate.

If those attempts fail, the default `symbolic_recovery=True` enables an
original-coordinate symbolic cell constructor. It stores incident plane IDs and
classifies vertices with filtered/exact predicates. Conservative cached Cartesian
intervals accept only strict signs; uncertain cases and exact ties use the
original symbolic predicates. Rounded Cartesian intersections never define its
topology. Site bisectors and clipping-box planes use the same
symbolic lifting rule as the certificate. This resolves the structured torus
boundary case tracked in [#61](https://github.com/alecjacobson/warpower/issues/61).
The resulting candidate still must pass the complete original-input certificate.

The symbolic constructor starts with a box padded by the largest input extent.
If certification finds only missing coverage or unsupported boundary faces, the
padding grows by eight on device. `symbolic_max_expansions` bounds the number of
box attempts (default 16, allowed 1 through 32). A successful box factor is reused
with current bounds on later replays. Capacity, predicate, and other certification
errors fail explicitly. Exact affine-rank checks reject lower-dimensional input
before symbolic cell construction. `symbolic_recovery=False` omits this path for
comparisons.

After an auxiliary or symbolic candidate succeeds, later replays try that construction
strategy first. Each replay rebuilds the original-site hierarchy and coincident
ownership from current input, constructs new connectivity, and certifies it
again. The auxiliary strategy normalizes a fresh copy of current positions. The cache retains a strategy choice only;
it does not reuse connectivity as success. If auxiliary attempts are exhausted,
the original constructor gets another chance on the current input. Invalid
input clears the preference, and every failure still publishes a zero count.
Each construction strategy runs at most once per replay, with its own finite
attempt budget. A failed preferred symbolic strategy falls back to the original
and auxiliary builders on current input.
`cache_strategy=False` disables this preference for comparisons. All choices and
retry limits are device values or fixed setup parameters.

The certificate uses [exact predicates](exact-predicates.md) with conservative
float64 filters. Every positive-volume candidate tetrahedron must have an empty
sphere against all original sites. A [conservative circumsphere box](sphere-bounds.md)
first skips strictly disjoint BVH nodes using float32 comparisons. Surviving nodes
retain the original float64 interval test, which can prove a strictly negative
determinant even for large, nearly flat hull spheres. Uncertain/nonfinite boxes
disable the new pretest. Uncertain leaves use exact
expansion arithmetic. Exact zero-volume candidates are discarded; negative
original orientations reject the candidate.

Cospherical ties use **one consistent symbolic lifting**. First evaluate the
ordinary determinant. At exact zero, evaluate its coefficient for infinitesimal
weights `epsilon * h(i)`. The determinant is affine in the weights, so this is
another exact weighted determinant. Remaining exact ties use successively smaller
positive infinitesimal weights in site-index order, evaluated using orientation
cofactors. This chooses connectivity without perturbing the output coordinates.
Connectivity is deterministic for fixed indexed inputs; output row order is not
specified. A different site ordering or a different library's symbolic rule can
produce another valid Delaunay triangulation of the same cospherical sites.

Each accepted tetrahedron is a strict lower facet of that common symbolic lifted
point set, so these tetrahedra form a nonoverlapping subcomplex. A face hash
requires opposite orientations at each paired interior face. Every unpaired face
must exactly support the original convex hull. Any missing volume would expose
an unsupported interior face, so a nonempty accepted subcomplex covers the full
hull. Duplicate tetrahedra fail incidence/orientation checks. Canonical-site usage
is checked explicitly. This argument uses global lower-facet checks, rather than
assuming that a topologically plausible candidate is geometrically embedded.
For the lifting interpretation, see the
[CGAL triangulations manual](https://doc.cgal.org/6.0/Triangulation_3/index.html).

## Errors and limits

`error[0]` is zero on success, one for invalid original input, and two for
exhausted construction/certification attempts. Failure always publishes a zero
output count. Corrected input can succeed on the next replay. **Finite retries
are not a universal exact constructor**: small perturbations can still fail to
resolve construction, capacity, or expansion problems. Entirely coplanar input
has no volumetric result and reports an error.

`attempts` counts normalized recovery attempts; `recovered` is one when auxiliary
or symbolic recovery supplied the output. `recovery_first` indicates that this replay tried
the cached strategy first. `base_attempted` indicates whether the original
constructor ran; its error is zero when skipped. `base_error` and `candidate_error` expose the
underlying builders' codes. A zero candidate code with zero attempts means no
candidate construction was needed. `certificate_error` describes the last
certificate, using the following bit mask:

| Bit | Meaning |
|---|---|
| 1 | Invalid count, index, repeated canonical vertex, or nonfinite vertex |
| 2 | Exact predicate could not certify a sign |
| 4 | Negative original tetrahedron orientation |
| 8 | Not a strict lower facet under the symbolic lifting rule |
| 16 | Traversal stack capacity exceeded |
| 32 | Invalid face multiplicity or orientation |
| 64 | Exposed face does not support the convex hull |
| 128 | Empty mesh or missing distinct site |

`symbolic_attempted` and `symbolic_recovered` indicate whether the original-coordinate
fallback ran and supplied the accepted output. `symbolic_expansions` counts box
enlargements in this replay. All are device scalars and reset each replay.
`symbolic_error` is zero when skipped or successful, with these failure bits:

| Bit | Meaning |
|---|---|
| 1 | Original sites have affine dimension below three |
| 2 | Cell capacity, traversal, or clipping topology failure |
| 4 | Uncertified or rank-deficient vertex predicate |
| 8 | Candidate tetrahedron capacity exceeded |
| 16 | Certificate rejected a property that box expansion cannot repair |
| 32 | Box-attempt budget exhausted |
| 64 | Expanded box cannot be represented by finite float32 bounds |

The default tetrahedron capacity is `8*N`; the worst-case Delaunay size can be
quadratic, so applications may need `tetrahedra_capacity`. Cell capacity is 1024
by default here, while the older `DiagramWorkspace` default remains 512. Two
construction workspaces, an additional original-site hierarchy for cached
recovery, and certificate buffers are reserved at setup. This
robustness path has measurable extra cost. Recovery and certification optimization
is tracked in [#56](https://github.com/alecjacobson/warpower/issues/56); it is not
a claim of performance parity with the original constructor.

The symbolic path shares existing integer and floating-point cell scratch and
candidate buffers; it adds only fixed scalar state and two box corners.
[Conservative directional bounds and cached vertex intervals](symbolic-bounds.md)
prune provably irrelevant sites and avoid repeated exact predicates. The remaining
worst-case cost is high; this fallback does not claim ordinary-builder throughput.
Setup includes compiling this optional kernel; cold compilation can be substantial.

The developing internal `VTMWorkspace` uses this certified constructor by default.
Its `certified_delaunay=False` option selects the earlier floating-point builder
for explicit comparisons. Fixed-hull `ODTWorkspace` remains unchanged. Boundary
and extraction behavior is documented [separately](boundary.md).

## Validation

Tests compare nondegenerate results against SciPy, complete hull volume against
an independent convex hull, and small degenerate outputs against rational
in-sphere determinants. They exercise lattices, cospherical points, duplicates,
scale/translation, corrupt candidates, failure recovery, and a changing-input
producer/mesher/consumer capture. Different valid cospherical triangulations are
compared geometrically rather than requiring SciPy's choice of diagonals.

```bash
python -m pytest -q tests/test_delaunay_recovery.py tests/test_filtered_predicates.py
python -m benchmarks.delaunay_recovery --case random --n 1000 \
  --repeats 10 --output results/local/recovery.json
```

The [full suite](../results/validation/vtm/recovery-tests.txt) passes **189 tests**.
The [installed-wheel check](../results/validation/vtm/recovery-wheel.json) runs a
captured producer/recovery/consumer loop, with installed sources matching the
checkout. The [isolation record](../results/validation/vtm/recovery-standalone-isolation.json)
verifies unchanged standalone source modules and no recovery imports from fresh
ordinary Voronoi or Delaunay calls.

CUDA memcheck with 32-byte allocation padding reports zero memory errors in the
[initial 14-case batch](../results/validation/vtm/recovery-memcheck-initial.txt).
That run had one test assertion failure: it incorrectly required SciPy's choice
of boundary diagonals for a cospherical case. The corrected test instead checks
the alternative tetrahedra with rational predicates. That test, four additional
one-ULP/near-coplanar cases, and all six VTM integration tests then pass the
[11-case follow-up](../results/validation/vtm/recovery-memcheck-followup.txt),
also with zero memory errors and stream-ordered lifetime checking. Together the
runs cover all current recovery/filter/VTM tests. Both use a diagnostic 32 KiB
stack; the first uses normal pooled allocation and the follow-up uses the
diagnostic synchronous allocator described in [CUDA validation](cuda-validation.md).
No production allocator or stack setting is changed.

The [native trace](../results/validation/vtm/recovery-device-audit.json) records
three warmed recovery replays for the 200-site corner cloud: 624 GPU kernel
activities and device-to-device copies. The captured graph contains 164 kernel,
40 memset, 19 conditional, and ten device-copy nodes, with no host or
allocation/free nodes. The CPU submits three graphs and performs one final
synchronization outside geometry. Record with Nsight `--cuda-graph-trace=node`;
the audit now rejects traces that omit kernel activity.

The original recovery baseline below is commit `3537da1`, before strategy reuse
and the bounded-integer predicate shortcut. Stage timings use CUDA events, five warmed repetitions, an NVIDIA L40,
and no observed CPU quota throttling. Setup is excluded. Stages are measured
separately, so their medians need not sum to the combined median; the auxiliary
stage is not part of a successful base-only replay.

| Input | Complete certified replay | Base attempt | Auxiliary candidate | Certificate |
|---|---:|---:|---:|---:|
| [8 cube corners](../results/validation/vtm/recovery-cube.json) | 190.26 ms | 182.98 ms, fails | 0.30 ms | 6.97 ms |
| [64-site lattice](../results/validation/vtm/recovery-grid.json) | 1872.37 ms | 1822.31 ms, fails | 1.56 ms | 48.40 ms |
| [200-site corner cloud](../results/validation/vtm/recovery-corner-cloud.json) | 19.13 ms | 10.18 ms, fails | 5.33 ms | 2.64 ms |
| [1,000 random sites](../results/validation/vtm/recovery-random.json) | 14.62 ms | 8.83 ms | 8.86 ms, unused | 5.69 ms |

Repeated failed base construction dominates the small structured cases. These
measurements motivate [#56](https://github.com/alecjacobson/warpower/issues/56),
including reuse of a previously successful construction strategy with fresh
certification. They are not a CPU mesher comparison or a VTM speedup claim.

## Reusing successful strategies

Strategy reuse and the bounded-integer predicate shortcut reduce repeated
structured-input work. The current CUDA-event measurements use ten warmed
repetitions on the same L40, with no observed CPU quota throttling. The baseline
is the original certified implementation at `3537da1`.

| Input | Original certified replay | Current warmed replay | Observed change |
|---|---:|---:|---:|
| [8 cube corners](../results/validation/vtm/recovery-fast-cube.json) | 190.26 ms | 0.53 ms | 359× faster |
| [64-site lattice](../results/validation/vtm/recovery-fast-grid.json) | 1872.37 ms | 2.71 ms | 691× faster |
| [200-site corner cloud](../results/validation/vtm/recovery-fast-corner-cloud.json) | 19.13 ms | 9.13 ms | 2.09× faster |
| [1,000 random sites](../results/validation/vtm/recovery-fast-random.json) | 14.62 ms | 14.70 ms | Within 1% |

These are improvements over the earlier optional certified implementation, not
Paragram or CPU speedups. The first replay still tries the original constructor:
it took 204 ms for the cube and 1846 ms for the lattice in this run. Setup is
excluded, and each record includes its first-replay timing. Every warmed replay
still rebuilds and certifies current input; the captured tests change positions,
scale, translation, duplicate ownership, and connectivity, and exercise fallback
to the original builder when the cached candidate strategy is exhausted.

The same change improves the complete *current uniform VTM core* for the
corner-only cube initialization. Five repetitions include the changing-input
producer, boundary/interior optimization, and volume consumer. Domain extraction,
graded sizing, and refinement are outside these timings. All runs complete the
requested iterations without rejected steps or CPU quota throttling.

| Example | Before | After | Iterations | Final full-hull tetrahedra |
|---|---:|---:|---:|---:|
| [Torus, 1,000 sites](../results/validation/vtm/recovery-core-after-torus.json) | [519.44 ms](../results/validation/vtm/recovery-core-before-torus.json) | 521.19 ms | 20 | 5902 |
| [Cube with full prefit, 200 sites](../results/validation/vtm/recovery-core-after-cube.json) | [76.16 ms](../results/validation/vtm/recovery-core-before-cube.json) | 76.92 ms | 12 | 876 |
| [Cube with corners only, 200 sites](../results/validation/vtm/recovery-core-after-corners-only.json) | [348.23 ms](../results/validation/vtm/recovery-core-before-corners-only.json) | 129.15 ms | 12 | 1156 |

The corner-only optimizer improves **2.70×**; the other two cases stay within 1%.
Final counts match, and the recorded energy histories differ by at most
`6.4e-15` relative to `max(1, abs(energy))`.

A [100,000-site random case](../results/validation/vtm/recovery-fast-random-100k.json)
produces 671,908 tetrahedra, all matching an independent
[native CGAL reference](../results/validation/vtm/recovery-fast-100k-cgal.json).
Certified replay is 1117 ms: roughly 473 ms construction and 645 ms certification.
The latter visits 383.9 million BVH nodes and remains a substantial bottleneck,
tracked in [#58](https://github.com/alecjacobson/warpower/issues/58). The overall
VTM CPU speed target is still open in [#40](https://github.com/alecjacobson/warpower/issues/40).

The [current full suite](../results/validation/vtm/recovery-fast-tests.txt) passes
**194 tests**. The [source audit](../results/validation/vtm/recovery-fast-isolation.json)
confirms only the three optional recovery modules change runtime code relative
to `3537da1`; all other runtime modules remain identical. The
[installed wheel](../results/validation/vtm/recovery-fast-wheel.json) composes
captured producers and consumers, including the preferred-strategy replay.
The [native trace](../results/validation/vtm/recovery-fast-device-audit.json)
records 390 kernel activities over three preferred-strategy replays. Its graph
has only kernel, memset, device-copy, and conditional nodes; it contains no host
or allocation/free nodes. All 23 graph copy nodes are device-to-device, although
this Nsight trace omits their execution activity records.

The [current memory-check batch](../results/validation/vtm/recovery-fast-memcheck.txt)
passes all **29 recovery, predicate-filter, and VTM tests**, with padding enabled
and zero CUDA errors. It uses the normal pooled allocator and the documented
32 KiB diagnostic stack allowance. The
[targeted lifetime batch](../results/validation/vtm/recovery-fast-lifetime.txt)
passes three captured producer/consumer and cached-strategy fallback cases with
stream-ordered race tracking, synchronous allocation, and zero errors. These
instrumentation settings follow [CUDA validation](cuda-validation.md); production
allocator and stack settings are unchanged.

## Structured boundary recovery

The original-coordinate symbolic fallback recovers the 128 boundary vertices of
`tests.test_domain.torus(16, 8)`, for which both earlier builders exhaust their
attempts. It returns 519 positive tetrahedra with an exact certificate of zero,
without moving the sites. The independent
[CGAL comparison](../results/validation/vtm/symbolic-cgal.json) has 520 tetrahedra:
467 are common, and all 52 GPU-only tetrahedra are independently empty and have
additional exactly cospherical sites. These are valid alternative triangulations.
The regression suite checks every output tetrahedron with rational predicates.

The [full suite](../results/validation/vtm/symbolic-tests.txt) passes **228 tests**,
including changed-input symbolic replay, exact hash collisions requiring the
final site-order tie rule, duplicate ownership, affine-rank failures, finite
capacity/box budgets, and fallback from an exhausted cached strategy. The
[source audit](../results/validation/vtm/symbolic-isolation.json) verifies that
26 existing runtime modules are byte-identical to `ac7b847`; only the optional
certified recovery dispatcher changes existing runtime code.

The [installed wheel](../results/validation/vtm/symbolic-wheel.json) composes a
changing producer, symbolic mesher, consumer, and device phase advance; the first
two outputs contain 519 and 523 tetrahedra. The
[native trace](../results/validation/vtm/symbolic-device-audit.json) records 285
kernel activities over three warmed changed-input replays. There are no host,
allocation, or free graph nodes, and all 21 graph copy nodes are device-to-device.
The CPU submits three graphs and performs one final synchronization.

The [initial memory batch](../results/validation/vtm/symbolic-memcheck-torus.txt)
completed two predicate tests and the full torus recovery test, including two
replays, with zero CUDA errors. The remaining batch was deliberately interrupted
after these three completed tests; its log retains the interruption and nonzero
application exit. The
[final 14-case batch](../results/validation/vtm/symbolic-memcheck-followup.txt)
passes the new predicates, smaller symbolic geometries, failure paths, and
captured cube producer/consumer with zero errors. The
[two lifetime cases](../results/validation/vtm/symbolic-lifetime.txt) also pass with
zero errors and stream-ordered tracking. Normal memory checks retain pooled
allocation and use 32-byte padding; lifetime checks use synchronous allocation.
Both use the documented 32 KiB diagnostic stack, without changing production
settings. The full torus geometry kernels are unchanged by the subsequent scalar
reset/dispatch optimization covered by the final batches.

The [paired timings](../results/validation/vtm/symbolic-paired.json) use ten warmed
CUDA-event repetitions on the L40 with matching tetrahedron counts and no observed
CPU quota throttling. The baseline's certified recovery source is identical to
`ac7b847`. Setup is excluded. Consolidating scalar resets and skipping remaining
fallback checks after a cached success avoids dispatch overhead on existing cases.

| Input | Before | After |
|---|---:|---:|
| 8 cube corners | 0.530 ms | 0.525 ms |
| 64-site lattice | 2.712 ms | 2.702 ms |
| 200-site corner cloud | 9.104 ms | 9.135 ms |
| 1,000 random sites | 14.669 ms | 14.636 ms |

These preserve the existing certified path within 0.4%, with a roughly 1.1%
improvement for the smallest cube. The original diagram and standalone
Delaunay/regular-mesh implementations remain unchanged. The new symbolic torus
fallback is much slower and is tracked separately in
[#62](https://github.com/alecjacobson/warpower/issues/62); these results do not
establish the complete VTM CPU speed target.

The [structured torus timing](../results/validation/vtm/symbolic-torus.json) is
4.286 seconds for the first certified replay and 2.423 seconds warmed, excluding
setup. It still needs substantial acceleration. The existing
[1,000-site uniform VTM core](../results/validation/vtm/symbolic-core-after-torus.json)
remains stable: [519.50 ms before](../results/validation/vtm/symbolic-core-before-torus.json)
and 519.95 ms after for 20 iterations, five warmed repetitions, no rejected
steps, and 5,902 full-hull tetrahedra. Energy histories agree within `6.2e-15`
relative to `max(1, abs(energy))`. This core timing includes its producer and
consumer, but excludes domain extraction, initialization, and refinement.
