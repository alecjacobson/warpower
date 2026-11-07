# Prepared seed-query hierarchy experiments

Four private traversal variants preserve the tested meshes but make complete VTM
solves slower. None is adopted. The subsequent [flat Morton-group design](grouped-seed-queries.md)
uses a different layout and retains the [prepared point expression](prepared-point-queries.md).

The current 4k trace attributes 554.292 ms across 208 launches to prepared seed
queries, about 19.5% of its 2,839.922 ms kernel activity. That motivates testing
fewer queries; these diagnostic durations are separate from complete-solve times.

## Variants and measurements

The first variant partitions the current site hierarchy into 32 disjoint
subtrees. Conservative sphere boxes prune strictly excluded nodes. Unresolved
leaf points use the existing point-specific Ball64 expression, then the original
complete symbolic predicate. The second variant also applies the general
polynomial interval enclosure to each surviving node. Prepared geometry remains
fixed during traversal, and no arrays or launches are added.

Two further box-only variants use one or eight workers per candidate. Their
partition depths are zero and three; the 32-worker variants use five path bits.
An early leaf belongs only to the worker whose unused path bits are zero. All
versions continue querying after rejection, and stack overflow fails closed.
They do not incorporate the separate rejected-cell-stop experiment in #169.

Fresh paired controls use main `ff5dff2` on NVIDIA L40 / Warp 1.19.0. Every replay
includes domain refit, fresh sampling, growth, fitting, quality stages, extraction
and diagnostics. The 1k cases use five measured replays and the 4k cases three.
All initial and measured CPU quota windows are unthrottled. Private kernels are
compiled before timing; no setup-speed claim is made.

| Query variant | Sites | Control (ms) | Prototype (ms) | Change |
| --- | ---: | ---: | ---: | ---: |
| 32 workers, box pruning | 1,000 | 715.675 | 744.117 | +4.0% |
| 32 workers, box pruning | 4,096 | 2,644.589 | 2,858.197 | +8.1% |
| 32 workers, box and polynomial pruning | 1,000 | 718.273 | 856.926 | +19.3% |
| 32 workers, box and polynomial pruning | 4,096 | 2,638.561 | 3,824.350 | +44.9% |
| 1 worker, box pruning | 1,000 | 716.302 | 996.537 | +39.1% |
| 1 worker, box pruning | 4,096 | 2,642.350 | 5,128.257 | +94.1% |
| 8 workers, box pruning | 1,000 | 715.778 | 799.954 | +11.8% |
| 8 workers, box pruning | 4,096 | 2,637.259 | 3,514.494 | +33.3% |

The experiments establish a regression for these layouts. They do not isolate
the individual contributions of traversal, divergence, local storage and bound
evaluation, or prove that every hierarchy-based approach must be slower.

## Validation and scope

Each of the four variants passes the same 69 existing reconstruction tests.
Each pair of variants also passes 36 direct captured-mask tests against both
the previous general-box checker and the current point-specific checker.
Those cases change geometry, duplicate ownership and candidate prefixes across
six device-produced frames, including empty and restored prefixes and poisoned
prior live flags. Cube, octahedron, thin, random and fitted 4k clouds cover scales
2^-60 through 2^60 and block sizes 32/256. Small cases additionally match independent
rational physical and symbolic masks. The narrow-worker tests launch the actual
one/eight-worker grids, rather than retaining the reference grid size.

All eight complete-solve pairs preserve bitwise coordinates and roles,
full/material tetrahedron sets, oriented boundaries, input arrays, parameters
and iteration work. Every prototype output passes independent CGAL global
full-hull certification and NumPy/AABB material-quality and single-cycle
boundary-link checks. Persistent pool allocations match the controls.

No broader-domain adoption benchmark, new memory/capture adoption audit, or
comprehensive extreme-box validation is claimed for these rejected prototypes.
Runtime code, public tests, standalone APIs and the README are unchanged.

The [manifest](../results/validation/vtm/prepared-hierarchy-probes/manifest.json)
records source hashes, private kernels, commands, raw samples and test logs.
[32-worker comparisons](../results/validation/vtm/prepared-hierarchy-probes/prepared-hierarchy-pilot-independent.json)
and [narrow-worker comparisons](../results/validation/vtm/prepared-hierarchy-probes/prepared-narrow-tree-pilot-independent.json)
contain the independent evidence. Scripts target the recorded baseline revision
and require the existing offline reference tools.

Tracked in [#172](https://github.com/alecjacobson/warpower/issues/172). The
[native CPU performance target](cpu-mesher-baseline.md) remains open.
