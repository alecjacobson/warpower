# Progressive VTM refinement

The private `warpower._refinement.RefinementWorkspace` performs one transactional
Voronoi-center insertion batch. It implements the progressive growth stage in
[Alliez et al., section 3.6](https://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Alliez05.pdf).
The [public mesher](mesher.md) composes it with initialization, fitting, quality
improvement, and final extraction.

```python
import warp as wp

from warpower._vtm import VTMWorkspace
from warpower._refinement import RefinementWorkspace

# points has fixed capacity; active_count and target_count are CUDA int32[1].
core = VTMWorkspace(domain, points, active_count=active_count, sizing=field)
growth = RefinementWorkspace(core, target_count, fraction=0.5)
with wp.ScopedCapture() as capture:
    core.run()
    growth.run()
    core.run()
```

Run the core with current points, domain, sizing, and active count before calling
the growth stage. It consumes the core's current valid connectivity, statistics,
and site hierarchy. Changing an input without refreshing the core violates this
precondition. Calling growth repeatedly is supported, but interleaving positional
optimization normally produces better meshes. Each call grows by at most
`min(target - active, max(1, ceil(fraction * active)))`. Target counts below the
current active count are invalid; this operation does not remove sites.

The default fraction is 0.5. `target_count` can be a setup-time integer or a
contiguous device int32 scalar, and defaults to the point capacity. A device
producer can change the target between replays. All buffers, including sorting,
deduplication, and rollback storage, are allocated at setup. Replays use Warp
kernels, GPU radix sorting/scans, and device conditional graphs; no host geometry,
readback, allocation, or synchronization occurs during replay.

## Candidate selection

Each current tetrahedron contributes a circumcenter and float64 circumradius.
The center's float32 rounding is certified, with an exact integer fallback when
the ordinary arithmetic is uncertain; see [consistent centers](stable-centers.md).
The actual float32 insertion position is retained only
when it is finite, classified inside the material, and distinct from existing
sites. Ambiguous material queries are excluded. Priority is descending `R / mu`,
where sizing is evaluated at the rounded insertion position. Uniform sizing uses
`mu=1`. Ranking and material queries remain geometric heuristics. Both the center's
rounding and the subsequently reconstructed Delaunay mesh are certified.

Canonical vertex order makes the calculation independent of tetrahedron row and
positive-orientation permutation. Stable device sorting breaks equal priorities
by the canonical four site IDs. Exact float32 coincident centers, including signed
zero, collapse to the first priority rank. In particular, co-spherical tetrahedra
do not insert repeated copies of the same center. Overlapping insertion cavities
are allowed because the complete mesh is reconstructed and certified globally.

Nonfinite or degenerate geometric candidates are skipped and counted in
`invalid_candidates`. Exterior, ambiguous, duplicate, and already occupied
centers are excluded from `eligible`. If no eligible center remains, the stage
reports stagnation even when the target has not been reached. Thin or sparsely
sampled domains can need an additional sampling strategy. This stage makes no
minimum quality or target completion guarantee.

## Transaction and outputs

The stage appends selected positions in priority order, updates the device active
count, reconstructs connectivity, and refreshes core statistics and boundary
roles. A successful insertion resets the core's iteration/rejection history and
records the new initial energy. Existing active positions and unused tail storage
are preserved. Refinement itself performs no positional optimization, and energy
is not expected to decrease when the site count or covered hull changes.

A failed reconstruction restores all positions, active count, connectivity, and
core statistics on device. The restored core can retain its previous valid state,
but this stage publishes zero output counts and a nonzero error. Consumers must
use this stage's error/counts when consuming its result. Internal certificate
scratch may describe a rejected trial and is not published connectivity.

`points` and `tetrahedra` alias the core arrays. `point_count` and
`tetrahedra_count` delimit this stage's successful output. All counts and status
values below are device arrays. `selected_tetrahedra[:inserted[0]]` records source
row IDs in the **pre-insertion** mesh. They are diagnostics, not indices into the
new connectivity. Outside that successful prefix the entries have no output
meaning, including after a rollback.

| Output | Meaning |
|---|---|
| `requested` | Number desired in this batch |
| `eligible` | Distinct eligible candidate centers |
| `attempted` | Number selected for the reconstruction trial |
| `inserted` | Number committed; zero on failure |
| `status=0` | Progress; target not yet reached |
| `status=1` | Target reached, including a successful no-op |
| `status=2` | No eligible centers; target not reached |
| `status=3` | Error; output counts are zero |

`error` is a bit mask:

| Bit | Meaning |
|---|---|
| 1 | Invalid upstream state, domain, sizing, count, or target |
| 2 | Invalid source tetrahedron index or nonfinite referenced site |
| 4 | Reconstruction, statistics, or boundary fitting rejected the trial |
| 8 | Restoring the prior core statistics/boundary failed |

`last_errors` stores reconstruction, statistics, and boundary error values from
the trial before rollback. Corrected inputs can recover on a later run.

The implementation is in a separate private module. Existing diagram, Delaunay,
ODT, and fixed-budget VTM code paths do not import or execute refinement work.

## Validation and example

The complete suite passes **280 tests**, including 11 refinement cases with
independent linear-system circumcenters, all-triangle winding, sizing/ranking,
co-spherical deduplication, a thin plate, repeated torus growth, invalid counts,
rollback, and captured producer/consumer recovery
([suite output](../results/validation/vtm/refinement-full-tests.txt)).
Two uniform/graded ranking cases pass fully instrumented pooled CUDA memcheck
with 32-byte padding and zero errors
([output](../results/validation/vtm/refinement-ranked-memcheck.txt)).
Two rollback/captured-loop cases pass full memory and allocation-order checking
with the [documented diagnostic settings](cuda-validation.md), also with zero
errors ([output](../results/validation/vtm/refinement-lifetime.txt)). A broader
instrumented run was stopped after six passing cases because of excessive
instrumentation overhead; it is not counted as a completed validation run.

The [native graph audit](../results/validation/vtm/refinement-device-audit.json)
covers three complete replays of a resident-input reset, optimization, growth
from 64 to 96 sites, optimization, and extraction. It records 2,868 GPU kernel
activities and only device-to-device copies. The graph contains no host callback,
allocation/free, or host transfer nodes. Its fixed nodes include both successful
and rollback branches; separate tests execute the failure branches.
All **35 existing runtime modules** are byte-identical to `dbecbff`
([source audit](../results/validation/vtm/refinement-standalone-isolation.json)).

The L40 torus example grows **256 → 384 → 576 → 864 → 1,000** sites with five
optimization steps per stage. Four additional five-step relaxation rounds
produce 4,206 retained tetrahedra, one genus-one component, median radius ratio
**0.880**, worst ratio **0.0664**, and directed sampled boundary RMS **0.0182**.
The retained volume is 4.82% below the input. Independent Delaunay and topology
checks pass. Median resident GPU time is **1.264 s** over five samples with no
CPU throttling ([record](../results/validation/vtm/refinement-torus-1000.json)).
Without the final relaxation, worst/median ratios are 0.00065/0.808 and RMS is
0.0315 ([record](../results/validation/vtm/refinement-torus-1000-before-polish.json)).
These are intermediate pipeline measurements: initialization, sizing, additional
quality stages, and a native CPU comparison are excluded.

```bash
python -m benchmarks.refinement --n 1000 --initial 256 --stages 4 \
  --iterations 5 --polish-rounds 4 --repeats 5 --reference \
  --output results/local/refinement-torus.json
nsys profile --trace=cuda --cuda-graph-trace=node --cuda-event-trace=false \
  --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi \
  --capture-range-end=stop --output results/local/refinement-loop \
  python -m benchmarks.refinement --n 96 --initial 64 --stages 1 \
  --iterations 1 --polish-rounds 0 --repeats 1 --profile \
  --dot results/local/refinement-loop.dot
```

The complete-pipeline [schedule comparison](quality-frontier.md) measures
progressive insertion against fixed-site optimization at the same site budget,
including quality and fit tradeoffs across seeds and domains.
