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
backtracking. It adds no workspace arrays or host decisions. It applies to
uniform, graded, active-prefix and selective VTM paths. Independently callable
power diagrams, Delaunay reconstruction, boundary fitting, and fixed-hull ODT
do not call it.

Tests cover translated/scaled boxes, multiple damping factors, cavities,
disconnected material, exterior recovery, unchanged boundary roles and masks,
inactive NaNs, a captured changing-input loop, and rollback on query failure.
The real hand regression checks both uniform and graded updates and verifies
physical tetrahedron orientations using independent integer arithmetic.

The complete hand still requires the extraction work in
[#112](https://github.com/alecjacobson/warpower/issues/112). Its failure gate
remains enabled; these constraints alone do not constitute a usable hand mesh
or a CPU performance result.
