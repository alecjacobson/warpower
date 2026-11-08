# Prefix equivalence and captured ownership

Scope: private unweighted VTM reconstruction after prepare_mesh has copied all
current active coordinates and filled every inactive scratch entry with an exact
copy of scratch site zero. This is not a permission to pass arbitrary unprepared
buffers to the private router.

Let N be reserved capacity, a the current valid active count (4 <= a <= N), and
m the selected constructor capacity (a <= m <= N). All sites with index >= a
are copies of site zero. Every distinct original active position therefore occurs
in [0,m). Removing [m,N) removes only duplicates of an earlier site. Canonical
first-owner indices are identical, and the remaining original site IDs are not
renumbered. The symbolic lifting tie order on canonical sites is unchanged.
This includes signed zero: padding copies the entire original point-zero value.

The selected constructor computes and certifies a complete current-coordinate
Delaunay mesh for the prefix, including current hull support. Since no distinct
active site has been removed, this proves the full active hull. The proof does
not follow from retained connectivity alone: every reuse rebuilds current owners
and hierarchy and re-establishes the original-coordinate certificate. Nonfinite
active input remains in every fitting prefix and must fail its existing guards.
Invalid device counts select no constructor and publish zero tetrahedra/error8.

This change adds capacities4096 and8192 only as explicit VTM-owned setup choices.
It does not change point predicates, geometry filters, canonicalization, sphere
certification, topology, existing fragment proof provenance or fallback rules.
A child output/strategy failure still invokes the full constructor; failed full
construction publishes no partial mesh. Prefix views begin at offsetzero.

The full constructor owns public output buffers. Connectivity and scalar results
from an accepted child are copied DtoD into those buffers. The full and every
child retain separate private previous-fragment storage; publishing one child
cannot mutate another child's saved fragment. Each fresh solve clears retained
fragments and strategy preferences for all routes. Re-entering any route after
a move, growth, shrinkage, rejection or unrelated input requires fresh proof.

Constructors, array slices, condition views and graphs are allocated before
capture. The count-dependent selection, execution, result copies, rejection and
fallback are device graph nodes. No host count read, new device view/allocation,
callback or host transfer is needed in execution. Native trace qualification is
still required; setup and any first-launch graph upload are outside this claim.

Exact symbolic geometry equivalence does not imply identical row order from
existing parallel construction, nor identical floating reduction trajectories
for every conceivable VTM solve. The experiment and production checks compare
coordinates, canonical cells, oriented boundary, roles and solver work strictly
for the measured inputs, and validate geometry independently. No future failed
strict comparison may be silently omitted or described as an identical solve.
