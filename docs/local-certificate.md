# Experimental local Delaunay certificate

This branch investigates issue #97. The following argument and implementation
need adversarial tests, independent equivalence checks and performance validation
before adoption. The existing global sphere certificate remains the fallback.

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

The implementation already establishes 1–3 and 5. For condition 4 it examines
up to sixteen boundary faces during the existing hull-support traversal. Strict
negative interval bounds may prune subtrees. An exact zero at any other canonical
site disqualifies that face. After a lane finds such a witness, that lane can
use the original nonpositive bound for its remaining traversal. The existing
32 lanes OR-write a shared disqualification flag; they do not need to repeatedly
read one another's flags while traversing. Failure to find an anchor is not an
input error: the global sphere check is used instead.

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

No coordinate is moved by this argument or implementation. All orientation,
hull-support and local power comparisons use original float32 sites and the
existing filtered/exact arithmetic. A failed predicate remains failure.

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
