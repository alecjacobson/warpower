# Exterior-band boundary recovery

The thickened-domain rule in Alliez et al. can retain exterior-center cells
that make a material boundary nonmanifold. The graded Armadillo regression has
two edges with four incident boundary triangles. Its complete Delaunay hull is
valid; each bad edge has a retained exterior-center cell separated from an
interior-center cell around the edge. Containment additions cannot resolve it.

`ExtractionWorkspace(..., exterior_recovery=True)` enables a bounded device
transaction for this case. The independent extractor defaults to `False`.
`VariationalMeshingWorkspace` enables it by default as a fallback after its
existing containment and additional-global-fitting recovery budget. Previously
successful fitting trajectories retain their selection and quality. Pass both `containment_recovery=False`
and `exterior_recovery=False` to use only the original circumcenter extraction
and additional-fitting policy. Diagram and Delaunay functions remain independent.

## Selection and safeguards

Recovery starts only with no input/query error and topology diagnostic exactly
2 (bad boundary edges). Each pass marks selected cells incident to such an edge
whose signed circumcenter distance is strictly positive. Interior centers and
centers on the input surface remain selected. Optional `feature_roles` is a
contiguous int32 array with one value per site: roles at least two protect every
incident cell. The full mesher supplies its current crease/corner roles; standalone
callers must supply roles when they need this protection.

The transaction preserves all vertices referenced by the starting selection,
including ordinary surface and interior sites. It may reduce selected volume
by at most one percent of the starting selected volume. It runs at most four
passes and stops when no cell changes or an unsupported diagnostic appears.
Every changed selection undergoes the complete face, edge and vertex-link
topology check. An orphaned vertex, excess volume change, unresolved topology,
or error rejects the transaction and restores the entire starting selection.

This is explicitly a boundary-approximation policy. A circumcenter outside the
domain does **not** prove that its tetrahedron is wholly exterior. The policy
does not establish a distance bound, exact facet conformity, preservation of
every input feature, or a quality threshold. Input cells must already form a
nonoverlapping positively oriented complex. The full mesher additionally requires
the output's component, shell, cavity and aggregate Euler counts to match its
input. Unresolved failures retain the existing zero-public-count behavior.

## Captured state and quality guards

All allocations and conditional graphs are created during setup. Replay uses
device classification, edge-table lookups, reference flags, compaction and
topology checks. It leaves the sites and complete input tetrahedralization
unchanged. `selected` and `source_indices` describe the final material selection.
If containment recovery also runs, that selection can include both additions
and removals relative to the original circumcenter rule.

Optional `exterior_enabled` is a device int32 scalar. Zero disables attempts on
that replay without changing ordinary extraction; any nonzero value enables
them. Omitting it enables attempts whenever the topology prerequisite holds.
The full mesher owns a shared scalar, resets it for each fresh solve, and enables
it only if the existing initial-extraction recovery budget is exhausted. The
same scalar controls extraction inside all subsequent quality guards.

Device `removed_count`, `exterior_rounds` and `exterior_rejected` reset every
replay. `removed_count` is zero on rejection; rounds count attempted passes;
rejected distinguishes an attempted unsuccessful transaction from no attempt.
The original classification measurements remain available in the circumcenter,
circumradius and signed-distance arrays.

After either recovery policy activates, selective optimization, boundary jitter
and interior moves assess the recovered subcomplex. The full mesher's checkpoint
and minimum-quality guards remain active. A fresh solve resets this trajectory
state; healthy trajectories retain their cheaper ordinary quality evaluator.

## Regression and independent validation

The reduced eleven-cell edge star tests preservation of interior centers,
surface centers, crease/corner cells and referenced vertices. It also tests
rollback for excess volume change and a second unresolved bad edge. Captured
device producers and consumers cover success, protected rejection, empty input,
upstream failure and restoration. A saved real-input regression requires exactly
the independently identified two cell removals and checks that selective quality
uses the same recovered cells and an independent NumPy minimum radius ratio.

The full graded scanned example uses the same input, sizing field and budgets
as the preserved failure. Its native full-hull, self-intersection, topology,
quality and boundary-distance checks are separate from resident GPU timing.
See the [scanned example](scanned-armadillo.md) for published measurements.
