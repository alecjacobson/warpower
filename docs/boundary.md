# Sampled boundary fitting

`BoundaryWorkspace` implements the sampled centroid update used for boundary and
feature fitting in Alliez et al. It is independently callable, and does not
construct tetrahedra. Current surface and crease samples are assigned to their
nearest active sites; weighted moments determine targets. Site roles can change
on every pass.

```python
import warp as wp
from warpower import SurfaceDomain, BoundaryWorkspace

domain = SurfaceDomain(vertices, triangles, surface_subdivisions=4)
fit = BoundaryWorkspace(domain, sites)
with wp.ScopedCapture() as capture:
    for _ in range(10):
        fit.run()                 # Compute targets and roles from current sites.
        fit.apply(damping=0.5)    # Move boundary sites; keep interior sites unchanged.
wp.capture_launch(capture.graph)
```

`run()` preserves the input positions. `apply()` uses the latest targets and
updates shared CUDA site storage in place. Recompute targets after changing the
sites. NumPy inputs are uploaded once; contiguous CUDA Warp arrays and PyTorch
tensors share their position buffers. Setup allocates/compiles and synchronizes.
Replays use device counts, conditionals, reductions, and fixed buffers, with no
host geometry decisions or allocation nodes. Use Warp `ScopedCapture` for the
conditional graphs. Both methods accept an optional Warp stream; callers must
establish dependencies with any other producers/consumers.

## Targets and priorities

| `roles[i]` | Site update |
|---|---|
| 0 | Interior/unassigned; unchanged by boundary fitting |
| 1 | Weighted centroid of assigned surface samples |
| 2 | Weighted centroid of assigned crease samples, overriding surface moments |
| 3 | Exact prescribed corner, overriding other targets |

`targets` stores the position for each active site, and device `role_counts`
contains the four population counts. Surface weights are `area / mu^4`; crease
weights are `length / mu^3`. Optional `surface_sizing` and `crease_sizing` arrays
supply positive float32 `mu` values on the device, one per reserved quadrature
sample. The default is uniform `mu=1`. These are sizing values, not power weights.
Inactive crease quadrature entries have zero base weight and are ignored.

Centroids approximate the restricted Voronoi cells. They are not projected onto
the input triangles or curves. In particular, an evolving crease site can be off
a feature, and a centroid whose samples span a junction can lie between its
branches. This is boundary approximation rather than exact facet/edge conformity.
Corners are preserved exactly: they receive distinct sites through parallel
nearest-available proposals and deterministic corner-index conflict resolution.
This matching is not a global minimum-cost assignment. A device loop resolves
collisions, bounded by the number of corners. The active site budget must cover
all corners. The compact `corner_sites[:domain.corner_count[0]]` maps corners to
sites, and `corner_owners` supplies the inverse map (or -1).

`apply(damping=1.0, max_step=1e30)` moves surface and crease sites toward their
targets, limiting displacement by `max_step`. Damping must be in `(0, 1]`.
Corner anchors are exact and bypass these two limits. Subsequent fits normally
retain an already occupied corner, but coincident-site ties can change the site
index used to represent it. Nearest-sample ties choose the lowest active index.
Float64 atomic sums need not be bitwise identical across executions.

## Counts, statistics, and errors

An optional contiguous CUDA int32 `active_count` scalar selects a prefix of the
fixed site capacity. Producers can change this count before replay. Inactive
storage is excluded from assignment and updates; the hierarchy still reserves
and processes its full capacity. `active_count` must be between one and capacity.

`surface_owners` and `crease_owners` map each sample to a site, with -1 for inactive
samples. `surface_mass` and `crease_mass` store weighted sample masses per site.
The two entries of `energy`, `total_weight`, and `max_distance` correspond to
surface and crease samples. Energy is the weighted sum of squared sample-to-site
distances. `sqrt(energy / total_weight)` gives weighted RMS distance when the
weight is nonzero. These are point-fitting diagnostics, not a symmetric distance
between the input and final mesh surfaces.

The device `error[0]` is a bit mask. Consume targets and statistics only when it
is zero; on failure, targets revert to the current input, roles/counts are cleared,
and `apply()` leaves positions untouched.

| Bit | Meaning |
|---|---|
| 1 | Invalid surface domain |
| 2 | Invalid active count or nonfinite active site |
| 4 | Insufficient distinct sites for corners or unresolved matching |
| 8 | Nonpositive/nonfinite active sample sizing |
| 16 | Invalid weighted reduction or target |

Corrected inputs can recover on a subsequent `run()`. Inactive site storage and
inactive quadrature sizing need not contain valid values. Domain topology remains
subject to the [surface-domain contract](domains.md).

## Boundary/interior integration under development

The internal `warpower._vtm.VTMWorkspace` combines these evolving boundary roles
with uniform interior ODT displacements, reconstruction, device backtracking,
and complete point/connectivity rollback. It is not yet exported as the public
full-mesher API: initial sampling, graded interior energy, refinement, and
the full public API are still under development. The independent
[domain extractor](extraction.md) can now consume its full-hull output in the
same captured loop.

Combined proposals use a shortest-edge displacement limit and damping. Accepted
reconstructions prioritize decreasing crease error, then surface error, then ODT
energy, with a small reduction tolerance. Moving boundary constraints changes the
current convex hull, so ODT energy alone can increase during accepted boundary
fitting. This acceptance rule is a practical heuristic, not a theorem about a
single global objective. Fixed-hull `ODTWorkspace` retains its separate behavior.

The internal core currently returns `full_tetrahedra` and a device
`full_tetrahedra_count`, covering the convex hull. This includes the torus hole
and exterior of other nonconvex domains. Do not use those outputs as a finished
domain mesh before applying `ExtractionWorkspace`. `termination=2` means exhausted backtracking and
restoration of the last valid state; it is not convergence.

The core now uses [exact-certified Delaunay recovery](delaunay-recovery.md) by
default. It can recover the cospherical cube-face configuration produced by
snapping only the eight corners. The cube integration example also demonstrates
a complete boundary fitting pass before optimization, supplying surface/crease
sites as well as corners. Finite construction recovery can still exhaust its
budget; failures publish a zero tetrahedron count. The explicit
`certified_delaunay=False` option selects the earlier floating-point constructor
for comparisons. Existing boundary benchmark records predate this change.
