# Material constraints on unassigned VTM sites

Boundary quadrature ownership is an approximate classification. A site with
`role=0` can sit very near the input surface without owning a sample. Its
unconstrained ODT optimum can lie outside the material. The shortest incident
edge need not be smaller than the distance to the surface.

The thin-hand example exposed this in the first optimization step: sites 9, 35,
and 112 were strictly inside the initial hull, had no boundary assignments, and
crossed the input bounding box. Continued boundary-priority acceptance let the
volume energy grow from 0.115 to roughly six billion by the hundredth step,
while surface and crease objectives improved. Eventually 201 of 1,000 sites had
a coordinate magnitude above five; the hand's bounds are
`[-1.4,-1,-0.12]..[1,2,0.12]`. Certified Delaunay connectivity alone does not
prevent this optimization error.

The internal VTM core now constrains nonzero directions of unassigned sites
using the existing pure-Warp surface hierarchy:

- For an interior site, the nearest surface distance defines a ball within its
  current material component. The maximum damped displacement uses 90% of this
  distance after subtracting a float32 storage margin of
  `8e-7 * (norm(position) + shortest_edge)`. Backtracking only reduces the move.
- An exterior site that has lost its boundary assignment moves toward its
  closest surface point, subject to the existing shortest-edge cap and damping.
  This is a recovery direction, not an interior ODT update.
- An unassigned site exactly on the surface remains fixed for this step.
  Subsequent quadrature assignments may give it a boundary update.
- Boundary/crease/corner directions and zero directions remain unchanged.
  In particular, selective-update masks and inactive NaN padding stay intact.

This trust region is an implementation safeguard, additional to the sampled
boundary fitting in [Alliez et al., section 3.5](https://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Alliez05.pdf).
It is conservative: tangential interior motion can also be shortened near a
surface. It does not establish convergence or a manifold extracted boundary.
An exterior recovery step can remain outside while approaching the surface.
Input boundary centroids keep their existing approximation semantics.

Invalid closest-point queries or ambiguous material classification set internal
core error bit `64`. The core stops, restores the last accepted sites and
connectivity, and publishes a zero full-tetrahedron count. This bit belongs to
the core error namespace; the parent mesher reports child errors through its
existing stage diagnostics.

The guard runs once per core optimization step, before reconstruction and
backtracking. For an interior site, it bounds the closest-point search to the
radius that could actually restrict the proposed displacement. A missing hit
inside that expanded cutoff leaves the direction unchanged.

One int32 per site records a known interior classification. A constrained move
stays in the material, so later unassigned steps can reuse that classification.
Any boundary role invalidates it, including a masked boundary step. Every core
call clears the cache, so new input positions, domain refits, external quality
moves, and changing active prefixes cannot inherit an old classification.
Backtracking restores an interior position and therefore preserves the fact
being cached. There are no host decisions. The guard applies to
uniform, graded, active-prefix and selective VTM paths. Independently callable
power diagrams, Delaunay reconstruction, boundary fitting, and fixed-hull ODT
do not call it.

Tests cover translated/scaled boxes, multiple damping factors, cavities,
disconnected material, exterior recovery, unchanged boundary roles and masks,
inactive NaNs, a captured changing-input loop, and rollback on query failure.
The real hand regression checks both uniform and graded updates and verifies
physical tetrahedron orientations using independent integer arithmetic.

The [full suite](../results/validation/vtm/interior-domain-full-suite.txt) passes
494 tests. All 12 new cases pass
[padded memory checking](../results/validation/vtm/interior-domain-memcheck.txt)
with zero errors, and the changing-input loop passes
[allocation/lifetime checking](../results/validation/vtm/interior-domain-lifetime.txt).
The [native audit](../results/validation/vtm/interior-domain-device-audit.json)
records 12 kernel activities and eight memory clears across four graph launches,
with one final synchronization and no CPU transfers, host callbacks, allocations,
or frees during replay. Instrumented checks use the
[documented diagnostic stack settings](cuda-validation.md).

Fresh paired L40 measurements include the complete pipeline and domain refit:

| 1,000-site case | Before | With guard | Change |
|---|---:|---:|---:|
| Uniform torus | 3,203.27 ms | 3,259.05 ms | +1.74% |
| Graded torus | 5,894.21 ms | 5,947.09 ms | +0.90% |

Positions, roles, and oriented connectivity match bitwise after sorting rows.
Graded quality arrays also match after aligning tetrahedron rows. Workspace
storage grows by 4,000 bytes at this capacity. These are correctness costs, not
a speedup claim. The initial unbounded-query implementation added 11% / 7.5%;
its [rejected measurements](../results/validation/vtm/interior-domain-unbounded-validation-summary.json)
remain available. The [final report](../results/validation/vtm/interior-domain-validation-summary.json)
includes raw paired results and confirms that 52 other existing runtime modules
and the README are unchanged.

Independent Delaunay medians are 9.175 → 9.181 ms at 1,000 sites and
182.969 → 182.629 ms at 100,000 sites. Short eager-diagram runs were noisy, so
an [alternating-order follow-up](../results/validation/vtm/interior-domain-diagram-repeat-summary.json)
uses two processes per revision and 15 measured calls per process. The per-run
medians span 18.14–18.19 ms before and 18.13–18.31 ms after; the data show no
consistent diagram regression. Both the short and longer records are retained.
All reported measurement windows have zero recorded CPU throttling.

The complete guarded hand reaches 1,000 sites with no rejected growth batches
and no sites outside the input bounding box. Its 5,851-cell full Delaunay mesh
is certified. The diagnostic material candidate contains 3,615 cells and 1,608
boundary faces, with 13 independently counted nonmanifold edges. Public output
therefore remains invalid with zero published counts. See the
[saved candidate](../results/validation/vtm/interior-domain-final-hand-final-candidate.npz)
and [edge-incidence diagnostic](../results/validation/vtm/interior-domain-final-hand-extraction-diagnostic.json).

The complete hand still requires the extraction work in
[#112](https://github.com/alecjacobson/warpower/issues/112). Its failure gate
remains enabled; these constraints alone do not constitute a usable hand mesh
or a CPU performance result.
