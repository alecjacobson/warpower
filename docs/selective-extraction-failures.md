# Repeated selective material-boundary failures

The uniform 8K Armadillo's repeated selective rejection is a material manifold
failure, not an unsuccessful full-hull Delaunay construction. A read-only device
observer records the original recovered-extraction status before it becomes
selective error 16. All fresh extraction failures have original error zero,
validity zero and topology diagnostic 2 (nonmanifold boundary edges).

The observer allocates separate bounded event and snapshot buffers, records
fresh rejection counters, and marks stale or ordinary-path extraction fields
unavailable. It preserves all original trial and acceptance code. No observed
duration is used as a performance result. Production does not import this module.

## Observed trials

These fresh 8K uniform solves use the Armadillo input with 43,243 vertices and
86,482 triangles, seed 2026, initial count 256, feature angle 120 degrees,
corner angle 45 degrees, one surface/crease subdivision and four quality rounds.
The input is treated as valid. The experiment is based on runtime `f3b35c9`, before
selective step history; source `ce68f92` and all loaded runtime hashes are pinned
in the evidence. It is not a new performance comparison of surface backends.

| Surface / nearest-site backend | Selective trials | Accepted | Fresh extraction failures | Saved failure meshes |
|---|---:|---:|---:|---:|
| Reference / FP64 | 59 | 2 | 57 | 4 |
| Native / mixed | 59 | 3 | 56 | 4 |

There is no event or snapshot overflow. The first failed recovered selection in
each of core calls 15–18 is saved, at iteration zero and the original step scale.
Each of those eight saved full Delaunay meshes passes independent CGAL exact
orientation, global empty-sphere, face pairing and hull-support checks.

All eight selected material meshes have the same three nonmanifold edge IDs:

| Site pair | Site roles | Incident full-hull tets | Selected tets in the edge star | Incident material boundary faces |
|---|---|---:|---:|---:|
| 2131, 3406 | Surface, surface | 4 | 2 | 4 |
| 2388, 3020 | Surface, surface | 7 | 2 | 4 |
| 2593, 6840 | Surface, surface | 5 | 2 | 4 |

Each selected boundary also has six invalid vertex links. Selected cell counts
are 39,382, 39,383, 39,377 and 39,376 across the four calls, so repeated edge IDs
do not imply unchanged geometry. The evidence includes coordinates, the complete
local edge stars and selected/unselected flags. Only the first failure of each
call is saved; these witnesses do not establish that every one of the 57 or 56
failed proposals has identical geometry or obstructions.

Both final before/observed meshes pass independent full-hull, material incidence,
boundary-link, surface-fit and exact boundary self-intersection checks. Within
each backend pair, sites are bitwise equal, and roles, canonical full/material
connectivity and oriented boundary faces match. Raw boundary row order differs;
that is not treated as a geometry change. The one-trial acceptance difference
between backends is retained in the report.

## Qualification and next experiment

The observer's bounded-event, stale-status and snapshot-tail contracts pass on
CUDA; the first two also pass on CPU. All three pass isolated padded and
pool-disabled allocation-lifetime memory checks with zero errors. This qualifies
the observer kernels, not an additional full 8K native-mesh memory audit.
The first offline analysis stopped at a missing module path after completing
four witness checks. Its partial output is retained; the corrected analysis
reuses those hashed proofs and finishes all eight witnesses and four final meshes.

The next useful experiment is a local, transactional response to these specific
edge-star obstructions. Repeated diagnostic bits alone cannot justify skipping
selective optimization or caching a rejection across changed sites. Any repair
must preserve the current full Delaunay certificate, containment and feature
conditions, manifold boundary, referenced sites and quality/fit acceptance; an
inconclusive repair must fall back to the existing bounded search. No recovery
policy or speedup is adopted by this report. Issue #258 remains open.

Reproduce the observation with `python -m benchmarks.selective_failures --input
armadillo.off --output results/local/unique-run`. Add `--query-backend native
--boundary-query-precision mixed` for the second configuration. The raw outputs
must not be committed to ordinary Git. Source pins, counters, exact witnesses,
quality reports and file hashes are in
[experiments/selective-extraction-failures](../experiments/selective-extraction-failures/).
