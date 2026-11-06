# Certified Delaunay fragment completion

`DelaunayWorkspace` can recover an incomplete candidate by filtering exact
Delaunay seed cells and extending their exposed faces to a complete convex hull.
The internal `_facet_completion.FacetCompletion` implements that extension. Input points remain the original
float32 values. A caller must supply a current hierarchy with canonical owners
for finite sites. Seed indices, counts, orientation, incidence and strict lower
facets are validated; a malformed or non-Delaunay seed is rejected.

The initial implementation repeats the complete global certificate after each
round. Only missing support/site coverage (bits 64/128) is eligible for extension.
Each unpaired oriented face searches sites on its positive side and selects an
opposite vertex with the same exact symbolic sphere predicate as reconstruction.
Thirty-two workers scan disjoint sets; a second kernel reduces their winners.
A fixed-capacity hash table deduplicates old and proposed tetrahedra before the
next certificate. No host counts, convergence checks, geometry predicates or
allocations occur inside the operation. Explicit round/capacity/predicate errors
publish zero output count, even when a partial mesh exists internally.

## Why the face search is ordered

This is an implementation-specific argument, not an assumption that arbitrary
seed tetrahedra already form a valid triangulation. The full initial global
certificate establishes that each positive seed is a strict lower facet of one
common symbolic lift. All sites use the same ordered infinitesimal weights.

Let `h(x)` be the exact oriented plane function of an exposed triangular face,
positive away from its seed tetrahedron. All affine lifted planes through its
three lifted vertices have the form `f_lambda(x) = f_0(x) + lambda*h(x)`.
The certified seed supplies a parameter `lambda_0` whose plane supports every
lifted site from below. For any positive-side site `q`, the plane through `q`
has parameter `t(q) = (lift(q)-f_0(q))/h(q)`.

The sphere comparison used in the scan selects the smallest `t(q)`, since a
positive sphere sign means the query's lift lies below the current plane. This
is a total order over eligible sites in the ordered infinitesimal field: a tie
would contradict the existing symbolic predicate's nonzero cofactor rule for a
positive defining tetrahedron. Partitioning and reducing minima is therefore
equivalent to a serial scan.

The selected `lambda_star` is at least `lambda_0`. Its plane remains below every
positive-side site by minimality. On the negative side, increasing `lambda`
lowers the plane, preserving the seed's strict support. On the face plane,
heights are unchanged. Thus the proposed positive tetrahedron is also a strict
lower facet of the same lift. With no positive-side site, the face supports the
convex hull and needs no neighbor.

Deduplicating these facets preserves the subset invariant. A proper nonempty
subset of the full three-dimensional lower-facet triangulation has a missing
neighbor across some internal triangle; otherwise it would be disconnected
from the rest of that triangulation. Each successful incomplete round adds a
facet. Finite output/round budgets bound execution; the final global certificate
remains a mandatory safety gate independent of this progress argument.

This adapts the ordered circle search and consistent tie-breaking discussed in
[Shewchuk's Delaunay notes, sections 3.7–3.8](https://people.eecs.berkeley.edu/~jrs/meshpapers/delnotes.pdf).
The planar explanation there is not itself a proof of this 3D implementation.

## Integration and bounded failure

`facet_recovery=None` (default) enables this path for buffers of 256–2,048 sites
when symbolic recovery is enabled. Explicit `True` or `False` overrides this
setup choice. Original/normalized builders retain their order; completion runs
only after all ordinary auxiliary attempts fail. A previously successful facet
strategy tries its last auxiliary amplitude and completion first on the next
replay. Every replay rebuilds current owners, filters fresh candidates, and
recertifies the resulting mesh. No connectivity is reused as a success result.
If the preferred strategy fails, the original builder still gets its chance.
The existing symbolic clipping path remains available after failed completion.
The initial 128-site automatic threshold regressed the structured torus boundary
case from 228 to 432 ms because its normalized seed construction was expensive.
The final 256-site threshold retains the existing small-cloud strategy; explicit
`True` still permits completion there. The rejected threshold's data are retained.
Successful publication also copies the matching certified face index and
candidate prefix. Automatic sizing can then extract the current hull and poles,
and subsequent independent certificate runs see the same accepted cells.

`facet_max_rounds` accepts 0–64 (default 64). An empty/invalid seed batch,
capacity/predicate failure, missing progress, or exhausted rounds cannot publish
partial output. The usual final public error and zero-count contract applies if
all remaining strategies fail. Five device diagnostics reset every replay:
`facet_attempted`, `facet_recovered`, `facet_error`, `facet_rounds`, `facet_seeds`.
The error combines the completion mask below with the seed-filter mask shifted
left eight bits. A failed strategy's diagnostics can remain nonzero when a later
strategy succeeds; inspect the public `error` for the final result.

| Completion bit | Meaning |
|---|---|
| 1 | Invalid/empty seed prefix, or seed certificate rejected a property other than missing hull/site coverage |
| 2 | A face predicate could not certify a sign |
| 4 | Output capacity exceeded |
| 8 | No new cell added to an incomplete mesh |
| 16 | Round budget exhausted |
| 32 | Final global certificate rejected completion |

Seed-filter bit 4 means invalid count/capacity; bit 2 means an unresolved predicate.
Malformed individual indices, zero-volume cells, and nonempty spheres are
filtered out. The remaining seeds must independently pass the completion
certificate. All scratch is allocated at setup. Completion forces global
certification, using 32 point workers per cell for buffers up to 2,048 sites;
ordinary global certificates retain their existing 128–512-site dispatch.

## Validation and performance

Initial tests cover one-cell and scattered seeds, missing hull/interior cells,
complete input, exact cube/octahedron ties, very thin cells, duplicate sites,
malformed indices/counts, inverted/duplicate/non-Delaunay seeds, and exhausted
round/capacity budgets. The saved 1,981-cell hand fragment reaches the same
2,164-cell complete mesh in three rounds. That early experiment excludes seed
construction and must not be compared with a full constructor.

The initial 16 completion tests pass padded memory checking. A four-phase
resident loop (one seed, invalid seed, translated current points, complete seed)
also passes full allocation/lifetime checking and a native replay audit. The
corrected trace records 707 GPU kernels, four launches, one final synchronization,
and only device-to-device copies, with no host or allocation/free graph nodes.
The initial benchmark violated SiteHierarchy's child-graph capture contract;
its top-level native sort added allocation/free nodes. Capturing the rebuild in
the required child graph fixes that audit. Public construction already follows
that contract. These records validate the initial isolated implementation;
integrated recovery has separate validation records.

`_facet_seeds.CertifiedSeeds` additionally filters uncertified candidate cells
into exact strict lower facets. It skips malformed individual cells, maps
canonical owners, rejects zero-volume cells, positively orients and deduplicates
the rest, and discards any cell with a nonempty symbolic sphere. Invalid input
counts and predicate failure reject the batch. The caller supplies finite sites
and current owners; downstream completion still validates the resulting subset.
Eight tests pass, including exhaustive cube/octahedron vertex combinations,
positive/negative/zero orientations, duplicate owners and cells, three scales,
malformed indices, nonempty spheres, invalid counts and safe replay recovery.
Integration additionally tests the difficult hand fixture, a structured torus,
failed budgets, a failed cached strategy followed by original construction, and
a captured producer/consumer loop that changes input, rejects NaNs, and recovers.

[Completion memory checks](../results/validation/vtm/facet-memcheck.txt),
[corrected lifetime check](../results/validation/vtm/facet-lifetime-v2.txt),
[native audit](../results/validation/vtm/facet-device-audit-v2.json), and
[seed tests](../results/validation/vtm/facet-seeds-focused.txt) retain the scope.
The complete 384-site hand constructor (including seed construction and all
attempted strategies) returns the same 2,164 canonical tetrahedra as main.
The first replay takes 3,955 ms versus 37,731 ms (9.54×); subsequent cached replays
take 179 ms versus 6,473 ms (36.1×) on the L40. These are device event timings, excluding
setup and output validation. The complete 1,000-site torus mesher remains about
3,202 ms versus 3,204 ms. These are GPU version comparisons, not CPU speedup claims.

The full thin-hand mesher still fails its growth budget: after the first batch
reaches 384 sites, seven one-site retries reach only 391 of the requested 1,000.
Public output is invalid and no partial material mesh is exported.
[Issue #108](https://github.com/alecjacobson/warpower/issues/108) tracks that separate
failure. The rejected-batch trace shows successful Delaunay certification and
boundary checks, followed by ODT volume evaluation error 256. A diagnostic GPU
kernel reproduces negative rounded determinants for two exactly positive cells.
[Issue #110](https://github.com/alecjacobson/warpower/issues/110) tracks robust
volume evaluation; [#109](https://github.com/alecjacobson/warpower/issues/109)
tracks roundoff-split insertion centers. Full VTM CPU performance and
comparable-quality validation remain open.

The final full suite passes **469 tests**. An earlier integration run had two
pole-extraction failures; publishing the matching certified face index fixes
them, and the failure/follow-up records are retained. Padded memory coverage
comprises 24 completed completion/seed tests from a partial run, then a completed
seven-case integrated run (823.84 s, zero errors). The partial run was manually
stopped during an expensive default-reserve hand test and does not certify that
unfinished test. The focused integrated tests use the supported 512-plane
clipping reserve; complete constructor timings use the default 1,024 reserve.
A separate captured-loop check passes full allocation/lifetime instrumentation
(390.93 s, zero errors). It executes four initial input phases; the padded run
also included a duplicated warm-up cycle.

The integrated native audit records **859 GPU kernels**, four graph launches,
and one final synchronization. Its graph has 398 kernel, 52 conditional,
106 memset and 41 device-to-device copy nodes, with no host/allocation/free nodes.
Nsight omitted individual copy activity records; all graph copy directions were
checked separately. Profiling uses four warm-up phases followed by four measured
phases. The final automatic-size threshold does not change these explicitly
facet-enabled audit paths.

| Complete operation | Main (ms) | Final (ms) |
|---|---:|---:|
| Random 1,000 sites | 9.191 | 9.184 |
| Random 10,000 sites | 40.648 | 40.659 |
| Random 100,000 sites | 181.519 | 181.999 |
| Cube, 8 sites | 0.526 | 0.526 |
| Lattice, 64 sites | 2.184 | 2.204 |
| Corner cloud, 200 sites | 6.067 | 6.082 |
| Torus boundary, 128 sites | 227.764 | 227.940 |
| Fitted hand, 256 sites | 13.394 | 13.423 |
| Refined hand, 384 sites, cached | 6,473.251 | 179.302 |
| Complete torus mesher, 1,000 sites | 3,204.364 | 3,202.189 |

Standalone cases use five serial CUDA-event samples per version. The refined-hand
main reference has two cached samples and the final version has five; torus
meshing uses three complete solves per version. No timing window was CPU-quota
throttled. Positions and oriented connectivity match on the saved refined-hand
and full-torus outputs. The 47 other existing runtime modules and README are
byte-identical to main; existing certificate kernels are unchanged. The changed
existing code is confined to certified recovery and its optional certificate
dispatch flag.

[Performance/provenance](../results/validation/vtm/facet-performance.json),
[output equivalence](../results/validation/vtm/facet-output-equivalence.json),
[final suite](../results/validation/vtm/facet-256-suite.txt),
[integrated memory](../results/validation/vtm/facet-integrated-memcheck.txt),
[partial memory scope](../results/validation/vtm/facet-integrated-memcheck-partial-default.txt),
[lifetime check](../results/validation/vtm/facet-integrated-lifetime.txt), and
[native audit](../results/validation/vtm/facet-integrated-device-audit.json)
retain the underlying records.

Progress is tracked in [issue #104](https://github.com/alecjacobson/warpower/issues/104).
