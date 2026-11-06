# Local Delaunay certificate with a global fallback

The optional `DelaunayWorkspace` can replace all-site sphere queries with
shared-face tests when it first establishes coverage through a strict hull
anchor. This file gives the sufficient geometric conditions and their argument.
An uncertain anchor search uses the existing global certificate. The original
power-diagram and `DiagramWorkspace` APIs are unchanged.

## Sufficient geometric conditions

Let P be the finite set of distinct canonical sites, H its convex hull, and T a
finite collection of tetrahedra with vertices in P. Require:

1. Every tetrahedron has strictly positive original-coordinate orientation.
2. Every indexed triangular face occurs once, or exactly twice with opposite
   induced orientations.
3. Every face occurring once has every site on its closed interior halfspace.
4. At least one such boundary triangle has every other distinct site strictly
   on its interior side. Its supporting plane contains exactly its three sites.
5. Every canonical site is used.

The implementation already establishes 1–3 and 5. For condition 4 a separate
bounded search examines up to sixteen boundary faces. It uses conservative
interval orientation bounds for both BVH nodes and individual sites. Only a
strictly negative finite bound can exclude a distinct non-face site from the
supporting plane. An uncertain sign, including an exact coplanarity, disqualifies
that candidate immediately; there is no expensive exact tie evaluation in this
optional search. The ordinary hull-support check remains unchanged and mandatory.
A traversal overflow also disqualifies the candidate. Failure to find an anchor
is not an input error: the global sphere check is used instead.

## Coverage argument

Every tetrahedron is contained in H. On the interior of H, sum their indicator
functions to obtain the nonnegative integer coverage multiplicity m. Each
positive tetrahedron contributes its outward-oriented face measure to the
distributional boundary. Condition 2 cancels all paired faces exactly, while
condition 3 places every remaining face on the boundary of H. Thus m has zero
distributional derivative on the connected interior of H and is constant almost
everywhere there.

Choose a point in the relative interior of the strict boundary triangle, away
from the finitely many other face edges and noncoplanar intersections. No other
boundary triangle overlaps an open patch around it: such a triangle would need
three sites on the same supporting plane; condition 4 leaves only the same three
indices, and condition 2 rules out another unpaired copy. Crossing this patch
from outside H increases m from zero by one. Hence m equals one almost everywhere
inside H. The closed tetrahedra cover H and have disjoint interiors.

The paired indexed faces supply matching affine traces of the lifted function.
A generic line segment through the interior crosses only face interiors, never
edges or vertices. It passes from one tetrahedron to the tetrahedron paired
across that face: another occupant would contradict multiplicity one. Along
such lines the piecewise affine function is continuous. Since there are finitely
many affine pieces with bounded gradients, continuity extends to nongeneric
segments and lower-dimensional intersections by limits. This avoids assuming
an independently proved simplicial-complex topology at singular intersections.

## Local lifting argument

Use the same formal lifted heights as the global certificate: squared norm,
then the negated hash weight, then successively smaller negated unit weights in
canonical-ID order. At each paired face, test the other tetrahedron's opposite
site against the defining positive tetrahedron using the existing exact symbolic
power-sphere predicate. A strictly negative result says the lifted affine pieces
bend strictly convexly across that face.

There are finitely many comparisons. The formal signs can be realized by finite,
sufficiently small positive perturbations chosen in their lexicographic order.
For any generic line segment, the derivative of the continuous piecewise affine
lift is nondecreasing at every crossing. It is therefore convex on that segment;
continuity extends convexity to all segments in H. Every tetrahedron's affine
extension supports the lifted function globally. Together with canonical-site
coverage, this yields the global lower-hull property. Strictness needs one more
observation. A query outside a defining tetrahedron crosses at least one strictly
convex face along a generic segment from its interior, giving a strict supporting
inequality. A distinct site on that tetrahedron's boundary would, by continuity
and site usage, equate its independently perturbed height with an affine
combination of the tetrahedron's other site heights. The first nonzero unit-weight
coefficient rules out that equality. Thus every other canonical site is strictly
above the tetrahedron's lifted plane. This also rules out a hidden hanging site
at a lower-dimensional intersection. The symbolic perturbation resolves the
same exact cospherical choices as the existing global certificate.

No coordinate is moved by this argument or implementation. All comparisons use original float32 sites. Orientation, hull support and local
power tests retain the existing filtered/exact arithmetic; optional anchor
discovery uses conservative strict interval bounds and falls back on uncertainty. A failed predicate remains failure.

## Implementation boundaries

Buffers below 128 sites retain the previous kernel sequence. Larger buffers add
sixteen disqualification flags and two scalar branch conditions, allocated at
setup. Face incidence and hull support precede branch selection. An anchor with
no preceding error selects local opposite-site checks; otherwise the original
global BVH sphere traversal runs. Canonical-site coverage remains mandatory
outside both branches. Neither strategy accepts the constructor's approximate
coordinates or reuses old connectivity without checking current input.

Statistics count actual power queries. Node visits are zero for a local
certificate; global statistics retain their previous interpretation. The choice
of the first sixteen faces is a performance heuristic only: any one verified
anchor suffices, and no anchor falls back. Face output order is unspecified, so
branch choice can vary without changing the acceptance condition.

## Relation to published results

The local-to-global lifting step is the standard Delaunay Lemma, including its
weighted form; see Shewchuk, *General-Dimensional Constrained Delaunay and
Constrained Regular Triangulations, I*, §3.2, Theorem 13
([author manuscript](https://people.eecs.berkeley.edu/~jrs/papers/cdtj1.pdf)).
Its premise is a triangulation, so it cannot by itself validate an arbitrary
candidate collection. The coverage argument above is the additional obligation
for this implementation.

Lipman's *Bijective Mappings of Meshes with Boundary and the Degree in Mesh
Processing* develops degree-based sufficient conditions for orientation-preserving
simplicial maps ([paper](https://arxiv.org/abs/1310.0955)). Its stated setting is a
manifold mesh with boundary. We do not assume that unverified candidates already
form such a mesh, or cite that theorem as a substitute for the chain-coverage
argument here.


## Validation and measurements

At runtime commit `fed60a8`, all 407 tests pass. Adversarial cases include holes,
repeated and inverted cells, both diagonals of a convex bipyramid, a missing
interior site, symbolic cospherical octahedra, and overlapping cube covers whose
boundary diagonals differ on every facet. That last case passes ordinary indexed
face incidence and hull-support checks despite covering the cube twice; it has
no strict anchor and the global fallback rejects it.

All 671,908 tetrahedra for 100,000 uniform random sites match saved native
CGAL 6.2.1 connectivity and the preceding constructor. Running the global
certificate again on that result also succeeds. The complete 1,000-site torus
retains identical positions, roles, input geometry and canonical full/material/
boundary connectivity. Its independently checked quality and topology are
unchanged.

The following L40 / Warp 1.19 measurements include complete certified Delaunay
reconstruction, except the last row, which includes complete VTM. Random cases
use five additional replays, other initial cases three. CPU quota throttling was
zero in every reported timing window.

| Case | Before complete solve | After complete solve | Certificate before → after |
| --- | ---: | ---: | ---: |
| Random 1,000 | 9.797 ms | 9.172 ms | 0.854 → 0.283 ms |
| Random 10,000 | 45.696 ms | 40.705 ms | 5.923 → 0.931 ms |
| Random 100,000 | 228.499 ms | 182.577 ms | 54.808 → 9.128 ms |
| Cube, 8 sites | 0.528 ms | 0.525 ms | 0.151 → 0.150 ms |
| Grid, 64 sites | 2.200 ms | 2.196 ms | 0.552 → 0.555 ms |
| Torus boundary, 128 sites | 1,220.684 ms | 1,144.326 ms | 90.717 → 13.162 ms |
| Fitted hand, 256 sites | 25.807 ms | 16.048 ms | 10.839 → 3.471 ms |
| Complete torus VTM, 1,000 sites | 3,632.697 ms | 3,203.738 ms | — |

Flat hulls pay a small fallback overhead. The 200-site corner-cloud case has
substantial timing variation from candidate ordering. Its initial three-sample
medians were 7.233 → 8.165 ms; both versions produced samples near 7.2 and 8.3 ms.
Two subsequent alternating-version batches of 31 samples each gave complete
medians 8.133 → 8.164 ms (+0.37%), and means 8.081 → 8.174 ms (+1.15%). On the
same fixed candidate buffer, alternating old/global-only and new/fallback graphs
for 30 samples at each of three shuffled orders isolates about 28 microseconds
of added certificate work (about 1.1% of that stage). This is below the documented
2% regression investigation threshold; it is not a claim of universal zero
overhead. All initial and follow-up samples are retained.

Eight focused tests pass padded Compute Sanitizer memcheck with zero errors.
A captured producer/constructor/consumer loop switches through random 200 sites
(local), a padded cube (global), invalid input, and restored random input. It
passes full stream-ordered allocation/lifetime checking with diagnostic
synchronous allocations. Both sanitizer runs explicitly set a 32-KiB diagnostic
CUDA stack allowance. A warmed native trace observes 514 GPU kernel activities,
four graph launches and one final synchronization; all 21 graph copy nodes are
device-to-device, with no host callback, allocation or free nodes. The existing
[larger conditional-graph instrumentation limits](cuda-validation.md) still apply.

Raw measurements, tests, native trace counters, connectivity hashes and source
isolation are recorded under
[`results/validation/vtm/local-*`](../results/validation/vtm/local-performance.json).
Only `_delaunay_certificate.py` changes among runtime modules; the other 47
runtime modules and the main README are byte-identical to the baseline.
The CPU mesher speed goal remains open.

Reproduce the targeted comparisons and captured branch audit with:

```bash
python -m benchmarks.delaunay_recovery --case random --n 100000 --repeats 5 --output results/local/delaunay100k.json
python -m benchmarks.certificate_fallback --output results/local/certificate-fallback.json
python -m benchmarks.local_certificate --output results/local/local-graph.json --dot results/local/local-graph.dot
```

The native trace requires `--profile` on the last command inside Nsight Systems;
setup, warmup and final host readback are excluded from that profiling range.
