# Tangential boundary jitter

`warpower._jitter.BoundaryJitterWorkspace` is an optional private VTM stage for
the boundary-jitter heuristic in §3.6 of [Alliez et al., *Variational
Tetrahedral Meshing*](https://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Alliez05.pdf).
It searches for a small tangential move that improves poor tetrahedra. It does
not run during ordinary diagram, Delaunay, or global VTM calls.

```python
import warp as wp
from warpower._vtm import VTMWorkspace
from warpower._jitter import BoundaryJitterWorkspace

core = VTMWorkspace(domain, points, max_iterations=20)
jitter = BoundaryJitterWorkspace(core, max_attempts=8, seed=2026)
with wp.ScopedCapture() as capture:
    core.run()
    jitter.run()
    # Consume jitter.tetrahedra[:jitter.tetrahedra_count[0]] on device.
wp.capture_launch(capture.graph)
```

Construction allocates fixed buffers and captures a graph. A call requires
current valid core geometry, statistics, boundary assignments, domain, and
sizing. Run the core after changing its inputs; refit mutable domain/sizing
inputs before that. Calls neither initialize sites nor do further ODT steps.
All decisions, proposals, reconstruction, exact Delaunay certification,
acceptance, and rollback execute on device. `run(stream=...)` can be recorded
inside a larger Warp capture. Workspace/array lifetimes must cover its replay.

## Search and safeguards

The search chooses the worst retained tetrahedron below `threshold=0.3` that
has a surface site (role 1). Equal qualities use sorted vertex IDs as the tie
break. It fixes this tetrahedron for the search, choosing a surface vertex and
a displacement uniformly in its tangent disk on each trial. The tangent plane
passes through the original site and uses the normal of the closest input
triangle. Crease and corner sites remain fixed. This is an approximate boundary
fit, not projection or exact facet conformity.

The disk radius is `step_fraction=0.1` times that site's shortest incident edge.
Float32 proposals must remain finite, distinct from every active site, and
within the displacement bound after rounding. The inactive capacity tail is
preserved bit for bit. A signed int32 setup seed or contiguous same-device
int32 `[1]` array controls reproducible SplitMix64 samples; the caller owns seed
advancement. Each attempt uses a separate stream. Same seed and initial state
give the same proposals; floating-point parallel reductions are not promised
bitwise across hardware.

Every trial reconstructs the complete convex-hull Delaunay mesh. Acceptance
requires all of the following:

- A valid extracted complex using `extraction_tolerance=0.4`.
- No decrease in minimum retained radius ratio and a strict decrease in
  `sum(max(0, threshold-q)**2)`, with relative numerical tolerance `1e-10`.
- Surface and crease sampled squared fitting errors within
  `boundary_tolerance=0.01` of the original values.
- Unchanged component, shell, cavity, and aggregate shell Euler counts.
- Extracted volume within `volume_tolerance=0.01` of the original.
- Directed surface RMS distance within `boundary_tolerance` of the original,
  plus `1e-7` times the current domain bounding-box diagonal for roundoff.

Retention-only quality checks discard unpromising trials before the more
expensive full extraction. Topology counts and volume guards are diagnostics,
not proof of equivalent material coverage. Jitter is a bounded heuristic:
it can exhaust its trials without improving the mesh.

A successful move refreshes core statistics and resets its optimization
history to one initial row. Exhausted trials restore the original positions,
tetrahedra, count, statistics, and boundary assignments; core iteration history
is preserved. Only the optional stage pays for these snapshots and checks.

## Outputs

All counts and diagnostics are device arrays. Check `error[0]` before consuming
outputs. `points` and `full_tetrahedra` alias the core; `tetrahedra` is the
compact extracted complex owned by this stage. `point_count`,
`full_tetrahedra_count`, and `tetrahedra_count` are zero on error, including a
failure to restore the original valid mesh. A failed proposal is a search
rejection rather than a fatal error when restoration succeeds.

| `status[0]` | Meaning |
|---|---|
| 0 | One move accepted |
| 1 | No eligible poor tetrahedron/surface site |
| 2 | Attempts exhausted; original valid mesh restored |
| 3 | Error; published counts are zero |

| Error bit | Meaning |
|---|---|
| 1 | Invalid upstream state, active count, or source tetrahedron count |
| 2 | Initial extraction invalid or empty |
| 4 | Invalid tangent or edge geometry during proposal preparation |
| 8 | Restoration failed |

`accepted`, `attempts`, and `invalid_proposals` summarize the bounded search.
`metrics` and `previous_metrics` contain minimum radius ratio, poor-quality
deficit, volume, directed surface RMS, sampled surface error, and sampled crease
error. `topology` and `previous_topology` contain component, shell, cavity, and
aggregate shell Euler counts. These arrays are meaningful only on success
(`error == 0`, including status 1 or 2).

`source_tetrahedron` is a **pre-move retained row**, while
`canonical_tetrahedron` holds its sorted vertex IDs. `site`,
`proposed_position`, `proposed_normal`, `proposed_distance`, and `proposed_limit`
describe the last attempted proposal; it is committed only if `accepted == 1`.
`last_errors` contains the last attempted mesh, statistics, boundary, query or
extraction, and topology errors. Invalid rounded proposals do not overwrite
those reconstruction diagnostics. Initial errors can leave proposal scratch
from a previous run; the published counts and error are authoritative.

`rejection_reason` bits are geometry/query/empty/nonfinite (1), worse minimum
quality (2), no deficit improvement (4), fitting error (8), topology counts
(16), volume (32), directed RMS (64), and invalid/no-op/coincident proposal
(128). This is the last trial's reason, not a union over the search.

## Measured example

On an L40, a 1,000-site torus after 30 global iterations accepted its second
jitter trial. Minimum radius ratio increased from **0.03429 to 0.08142**, and
the poor-element deficit fell from **0.54576 to 0.47516**. Median radius ratio
changed from 0.882305 to 0.882286. The one-component genus-one boundary remained
valid; directed RMS changed from 0.0180764 to 0.0180810, within the configured
guard. Independent Delaunay and topology checks matched.

Median GPU time was **133.41 ms** over five unthrottled replays. This includes
resetting resident fitted points, certified reconstruction/statistics refresh,
and the complete jitter search/extraction. It excludes initial fitting,
sampling, sizing, and other mesher stages; it is not a complete-mesher or CPU
speedup result. See [raw measurements](../results/validation/vtm/jitter-torus1000.json).

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python -m benchmarks.jitter \
  --n 1000 --warmstart-iterations 30 --attempts 8 --reference \
  --output results/local/jitter-torus1000.json
```

Eleven focused tests cover accepted movement, independent geometry, tangential and
step bounds, canonical selection, rejected quality and geometry restoration,
duplicate proposals, invalid and empty inputs, active-prefix graded sizing,
sharp features, and device-seed/capture composition. Native graph and CUDA
memory-check evidence is stored alongside the measurements:

- [Full suite](../results/validation/vtm/jitter-full-tests.txt): 311 passed.
- [Allocation-lifetime and full memory checks](../results/validation/vtm/jitter-lifetime.txt):
  three capture/rollback/duplicate cases, zero sanitizer errors.
- [Pooled full memory checks](../results/validation/vtm/jitter-pooled-memcheck.txt):
  two graded-prefix/quality-rollback cases, zero sanitizer errors.
- [Native audit](../results/validation/vtm/jitter-device-audit.json): 1,995 kernel
  activities over three replays; only device-to-device copies, no host callbacks,
  allocating/freeing graph nodes, or CPU transfers. The final synchronization
  is outside geometry. Memory checks use the documented diagnostic stack
  allowance in [CUDA validation](cuda-validation.md).
- [Source audit](../results/validation/vtm/jitter-isolation.json): all 37
  preexisting runtime modules remain byte identical to `c72f152`. Ordinary
  diagram, Delaunay, and VTM paths import neither new module.
