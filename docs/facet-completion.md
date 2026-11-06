# Certified fragment completion experiment

`_facet_completion.FacetCompletion` is an internal experiment, currently isolated
from public construction. It extends a nonempty subset of exact symbolic
Delaunay tetrahedra to a complete hull. Input points remain the original
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

## Current scope

Initial tests cover one-cell and scattered seeds, missing hull/interior cells,
complete input, exact cube/octahedron ties, very thin cells, duplicate sites,
malformed indices/counts, inverted/duplicate/non-Delaunay seeds, and exhausted
round/capacity budgets. The saved 1,981-cell hand fragment reaches the same
2,164-cell complete mesh in three rounds. This measures completion of an already
available fragment; it excludes the cost of constructing that fragment and is
not a full VTM or CPU-speed comparison. The 16 completion tests pass padded memory checking. A four-phase resident loop
(one seed, invalid seed, translated current points, complete seed) also passes
full allocation/lifetime checking and a native replay audit. The corrected
trace records 707 GPU kernels, four launches, one final synchronization, and
only device-to-device copies, with no host or allocation/free graph nodes.
The initial benchmark violated SiteHierarchy's child-graph capture contract;
its top-level native sort added allocation/free nodes. Capturing the rebuild in
the required child graph fixes that audit. Public construction already follows
that contract. No complete-constructor integration is claimed yet.

`_facet_seeds.CertifiedSeeds` additionally filters uncertified candidate cells
into exact strict lower facets. It skips malformed individual cells, maps
canonical owners, rejects zero-volume cells, positively orients and deduplicates
the rest, and discards any cell with a nonempty symbolic sphere. Invalid input
counts and predicate failure reject the batch. The caller supplies finite sites
and current owners; downstream completion still validates the resulting subset.
Eight tests pass, including exhaustive cube/octahedron vertex combinations,
positive/negative/zero orientations, duplicate owners and cells, three scales,
malformed indices, nonempty spheres, invalid counts and safe replay recovery.
Seed filtering is not yet connected to public Delaunay construction.

[Completion memory checks](../results/validation/vtm/facet-memcheck.txt),
[corrected lifetime check](../results/validation/vtm/facet-lifetime-v2.txt),
[native audit](../results/validation/vtm/facet-device-audit-v2.json), and
[seed tests](../results/validation/vtm/facet-seeds-focused.txt) retain the scope.
Full integration and performance validation remain required before adoption.

Progress is tracked in [issue #104](https://github.com/alecjacobson/warpower/issues/104).
