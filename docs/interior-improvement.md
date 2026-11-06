# Interior teleportation and insertion

The private `warpower._teleport.InteriorImprovementWorkspace` implements a
bounded interior-improvement step based on §3.6 of [Alliez et al., *Variational
Tetrahedral Meshing*](https://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Alliez05.pdf).
It moves one eligible dense interior site to a poor interior tetrahedron's
centroid, or inserts a site there if a target budget permits growth.

```python
from warpower._teleport import InteriorImprovementWorkspace

# core must have current valid geometry, statistics, and boundary assignments.
move = InteriorImprovementWorkspace(core)  # Preserve the current site count.
move.run()

# With a core configured with active_count and spare point capacity:
grow = InteriorImprovementWorkspace(core, target_count=device_target)
grow.run()
```

`run(stream=...)` composes with producer/core/consumer kernels in a larger Warp
capture. Setup allocates buffers and captures a graph; proposal selection,
optional dual-volume reduction, reconstruction, certification, acceptance,
and rollback run entirely on device. The independent diagram and Delaunay
functions and default global VTM path do not execute this stage.

## Selection and transaction

The destination is the worst retained tetrahedron below `threshold=0.3` whose
four vertices are currently interior sites (role 0). Its rounded float32
centroid must be strictly inside the input material. Equal computed qualities
use sorted vertex IDs; the centroid uses that canonical summation order.

When `target_count > active_count`, the proposal appends one site at the active
prefix's end. Otherwise, it moves the eligible donor minimizing
`Voronoi_volume / sizing(site)**3`. Donors must be represented interior sites
with bounded cells and must not be among the destination's four vertices.
Equal computed scores use the lowest site ID. Omitting `target_count` preserves
the current active count. An explicit target requires a core with
`active_count`; it can be a setup integer or contiguous same-device int32 `[1]`
array and must satisfy `active_count <= target_count <= capacity` at replay.

One run attempts at most one candidate. It shares the [jitter transaction's
quality, fitting, topology, volume, and RMS guards](jitter.md#search-and-safeguards).
In particular, retained minimum quality cannot decrease, and the poor-element
deficit must fall. The donor and target choice is a heuristic, not a guarantee
that a trial will be accepted. Rounded duplicate/no-op proposals are rejected.

Acceptance refreshes statistics and restarts core history. Rejection restores
all original points (including unused capacity), the active count, complete
tetrahedra, statistics, and boundary assignments, preserving optimization
history. Corner, crease, and surface sites present before the operation stay
fixed. No eligible destination or donor returns the original valid mesh.

## Bounded Voronoi volumes

`warpower._voronoi_volume.BoundedVoronoiVolumes` is a private, read-only helper
requiring a current certified full-hull Delaunay mesh. It rebuilds its own face
table from the published connectivity, so stale certificate scratch after an
optimizer rollback cannot affect hull classification. Hull sites, unrepresented
duplicate sites, and the inactive prefix tail receive zero volume and validity.

For a positive tetrahedron and each permutation `(i,j,k,l)`, the helper adds
the signed circumcentric flag volume

`parity(i,j,k,l) * det(midpoint(i,j)-p_i, face_center(i,j,k)-p_i, tet_center-p_i) / 6`

to site `i`. Signed pieces handle obtuse faces and tetrahedra. Local float64
coordinates avoid forming large translated determinants. The sum is the bounded
Voronoi-cell volume, rather than a lumped incident-tetrahedron volume. The
normalized value uses the current sizing field at the original site, or unity
for uniform sizing.

The helper exposes `volume`, `normalized`, `valid`, `hull`, `incidence`,
`count`, `invalid_cells`, and `error`, all device arrays. Its errors are invalid
upstream/count/field (1), invalid indexed geometry (2), face multiplicity above
two (4), and invalid numerical volume/normalization (8). A nonpositive or
nonfinite volume, or a volume at most `1e-12` times the sum of absolute signed
contributions, fails explicitly. This cancellation diagnostic is not a rigorous
interval bound. Any error clears all published volumes, normalized values,
validity flags, and the valid-cell count.

Independent halfspace-intersection/convex-hull tests compare bounded cells,
including translated/scaled point sets, obtuse cells, a cospherical lattice,
and graded normalization. Floating-point reductions are not promised bitwise
identical across hardware; donor tie breaks apply to computed scores.

## Output and failure contracts

The shared transaction exposes `points`, `full_tetrahedra`, retained
`tetrahedra`, their device counts, before/after quality metrics, topology
diagnostics, and last reconstruction errors. Its [status and rejection bits](jitter.md#outputs)
are unchanged. Additional fatal error bits are **16** for a failed dual-volume
calculation and **32** for an invalid target budget. On any error, all three
published counts are zero even if the source core still contains a valid mesh.

`mode[0]` is 0 for teleportation and 1 for insertion. `inserted[0]` becomes 1
only after a committed insertion; an unsuccessful insertion leaves it zero.
`source_tetrahedron` refers to a pre-trial retained row;
`canonical_tetrahedron` identifies its vertices. `site` and `proposed_position`
describe the attempted change and are authoritative only with the corresponding
success/attempt diagnostics. The helper `cells` contains **pre-trial** donor
volumes only when a donor search ran. An insertion or no-op can leave that
scratch from a previous replay; call its `run()` separately for current volumes.

Material extraction must already be valid. A valid full-hull mesh alone is
insufficient: a coarse graded cube can have a nonmanifold retained boundary.
This stage refuses it before moving sites. [Issue #76](https://github.com/alecjacobson/warpower/issues/76)
tracks bounded device recovery/refinement in the full scheduler.

## Example and validation

On a sharp cube with 128 sites after five global iterations, either one donor
teleport or one insertion raises the worst fully interior tetrahedron's radius
ratio from **0.05584 to 0.31539**. Poor-element deficit falls from **0.39972 to
0.34011**. The global minimum remains 0.02592 at an unaffected boundary element;
no global minimum improvement is claimed. The extracted complex stays valid,
and independent Delaunay and topology checks agree (exact ties may use different
valid diagonals). Median radius ratio changes from 0.75876 to 0.75596
(teleport) or 0.75618 (insert). This coarse example retains volume 6.60 of the
box's volume 8; it demonstrates the local improvement, not adequate boundary
resolution for a finished mesh.

Median L40 times over five unthrottled replays are **19.22 ms** for teleportation
and **18.41 ms** for insertion. They include resident point/count reset,
reconstruction/statistics refresh, the transaction, and extraction/diagnostics.
They exclude initial fitting, sampling, sizing construction, and other mesher
stages. These are stage results, not complete-mesher CPU speedups.

The [1,000-site cube after 20 global iterations](../results/validation/vtm/interior-teleport-cube1000.json)
also accepts a teleport: minimum interior radius ratio rises from 0.14046 to
0.23893 in **59.85 ms**. Its volume is 7.92734 of 8 and directed surface RMS is
0.0136805. The global minimum is still a poor boundary element at `3.41e-5`;
the interior stage does not fix it. Median quality stays 0.87577, while the
fifth percentile changes from 0.57538 to 0.57463. This result also passes
independent Delaunay/topology checks and has no measured CPU throttling.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python -m benchmarks.interior \
  --mode teleport --reference --output results/local/teleport.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python -m benchmarks.interior \
  --mode insert --reference --output results/local/insert.json
```

Raw [teleport](../results/validation/vtm/interior-teleport-cube128.json) and
[insertion](../results/validation/vtm/interior-insert-cube128.json) measurements
include the full quality distributions and boundary errors.
[Focused tests](../results/validation/vtm/interior-tests.txt) cover volume
oracles, hull/duplicate/inactive sites, graded scores, stale scratch, accepted
and rejected modes, target changes, failed geometry, restoration, and capture.
[Source isolation](../results/validation/vtm/interior-isolation.json) records
38 unchanged runtime modules and all 20 unchanged transaction kernel/function
ASTs. The only preexisting Python change extracts the same point-update launch
into an overridable method; jitter records the same default launch.

The [full suite](../results/validation/vtm/interior-full-tests.txt) passes 330
tests. Three [full-memory/allocation-lifetime cases](../results/validation/vtm/interior-lifetime.txt)
and three [pooled full-memory cases](../results/validation/vtm/interior-pooled-memcheck.txt)
report zero CUDA sanitizer errors, using the documented
[diagnostic stack allowance](cuda-validation.md). The
[native audit](../results/validation/vtm/interior-device-audit.json) records
1,812 kernel activities over three teleportation replays; all observed and graph
copies are device-to-device, with no host/allocation/free graph nodes or CPU
transfers. Captured insertion, rejected geometry, changing device targets, and
downstream consumption are exercised by the tests.
