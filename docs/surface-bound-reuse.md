# Exact reuse of nearest-surface box distances

Reusing already computed child box distances reduces complete uniform and graded
Armadillo4096 solve time by **3.43% and 5.16%** in fresh-process measurements.
The independent power-diagram and Delaunay routines remain unchanged, with no
measured regression. This is an improvement between Warp revisions; the
comparable-quality CPU outperformance goal remains unmet.

## Query behavior

The original nearest traversal computes both child distances to choose traversal
order, then recomputes a child's distance when popping it. The new traversal
stores those rounded float64 scalars beside the original node stack. It tests
each stored value against the **current** best distance at pop time. The query
point and current tree are fixed throughout a call, so the value has the same
meaning at both locations.

The original root evaluation, child ordering (including strict ties), triangle
arithmetic, strict nearest update and `>=` pruning comparison are retained.
Nonfinite comparisons follow the same branches. An induction over the stack
gives the same node order and query decisions under the original arithmetic:
each pushed entry holds its node's original box result, and subsequent triangle
updates affect only the best-distance threshold. There is no approximate bound,
persisted query result or stale value across refits.

Only `closest_point` changes in runtime source. No public arguments, defaults,
persistent device arrays or independent geometry implementation change. The
extra local stack has 128 float64 elements. Its actual compiled local-memory
cost is reported below. The separate
[work-attribution and bounded-arithmetic investigation](surface-bound-investigation.md)
records the motivation and prototypes that were not adopted.

## Complete solves

Pairs compare merged baseline `18ecb03c10cb10e42a5089e8253ed92a24c3bae1`
with runtime `576ac5193125ae980da2314ed7502bd6938aeb79` on NVIDIA L40,
Warp 1.19. Each replay executes a fresh complete solve, including refits,
sampling, growth, fitting, quality work and extraction. The 1k torus uses five
warmed samples; the other seven examples use three. Setup/compilation,
serialization and offline validation are excluded. Setup and first-replay times
are retained separately. All first and timed quota windows are clean.

| Example | Before ms | After ms | Time reduction |
|---|---:|---:|---:|
| torus1000 | 617.027 | 606.032 | 1.78% |
| torus4096 | 1814.942 | 1783.559 | 1.73% |
| graded-torus | 2085.384 | 2050.441 | 1.68% |
| thin-hand | 1336.097 | 1331.454 | 0.35% |
| cavity | 442.270 | 449.028 | -1.53% |
| disconnected | 497.206 | 501.070 | -0.78% |
| armadillo-uniform | 2953.910 | 2852.595 | 3.43% |
| armadillo-graded | 3705.450 | 3514.188 | 5.16% |

This is not a universal speedup: cavity is 1.53% slower and disconnected solids
0.78% slower in this batch. The larger scanned cases improve more substantially.
No aggregate score hides the small-case results.

All input arrays, parameters, roles, complete/material tetrahedron sets, oriented
boundary faces and solver work match. Seven position arrays are bitwise equal.
Thin-hand differs in two near-zero coordinates, with maximum absolute difference
`7.453889935837843e-19`; both values are recorded in the paired checks.

Every final mesh passes independent CGAL global empty-sphere, original-coordinate
orientation, face and convex-hull checks, NumPy quality, CGAL AABB sampled
distances and boundary vertex-link checks. All six gallery VTUs round-trip
exactly, and their exported boundaries have zero checked self-intersection
pairs. Sampled distance checks are not a Hausdorff certificate.

Every public quality field agrees within absolute/relative tolerance `1e-12`;
the largest observed floating reduction difference is `1.315e-13`. Discrete
fields match, and public RMS remains available and nonzero. The initial offline
checker required bitwise positions in every case and stopped at thin-hand. Its
source, failure log and partial report are retained. The revised checker keeps
the earlier comparison policy for these previously sensitive examples, bounds
coordinate differences by `1e-8`, and requires matching topology/work and the
independent quality checks above. No runtime or benchmark output was changed to
satisfy that check.

## Independently callable routines

| Routine/input | Before ms | After ms | Change |
|---|---:|---:|---:|
| delaunay1000 | 9.185 | 9.179 | -0.06% |
| delaunay100000 | 182.969 | 182.900 | -0.04% |
| delaunay-grid | 2.205 | 2.193 | -0.54% |
| delaunay-fitted-hand | 13.442 | 13.425 | -0.12% |
| diagram100000 | 18.190 | 18.099 | -0.50% |

All recorded hashes, counts, errors and diagnostic fields match. These small
timing differences do not establish a standalone speed gain. Source and defaults
are unchanged. README retains installation and the earned Paragram flagship.

## Local resources and graph execution

CUDA function attributes queried for the compiled block-size-32 variants report:

| Kernel | Registers/thread before → after | Local bytes/thread before → after |
|---|---:|---:|
| `classify` | 255 → 255 | 33,800 → 35,960 |
| `classify_selection` | 255 → 255 | 33,848 → 36,008 |
| `surface_error` | 102 → 96 | 1,136 → 3,296 |
| `selection_surface_status` | 104 → 98 | 1,136 → 3,296 |
| `classify_retained` | 255 → 255 | 33,808 → 35,968 |

The added local footprint is 2,160 bytes per thread in these compiled kernels.
These attributes do not measure occupied memory, traffic or achieved occupancy.
Nsight's local-memory fields in this capture returned zero per-thread values
and some negative totals, so the retained function-attribute queries provide
the resource comparison. Measured persistent workspace pool sizes are identical
for both torus capacities; no persistent device arrays are added.

The complete graded-scan native audit records 73,845 kernel activities and a
graph with 38,112 kernel and 3,654 device-copy nodes, matching the preceding
runtime's structure. A separate eight-phase producer/refit/selection/full-query/
consumer loop records 1,130 kernel activities and three device-copy nodes. Its
device comparisons pass across ordinary/recovered selection, scaled inputs,
nonfinite sites/domain, source failure and recovery. Public actual distances
and private sign-only results keep their respective contracts.

Both traces include first native instantiation/upload, one graph launch and
one final audit synchronization. Neither contains host transfers/callbacks,
graph allocation/free nodes or explicit allocation/free API calls. DOT copy
nodes and observed copy activities are checked separately because Nsight can
omit graph-copy activities. These instrumented executions validate structure;
the clean timing table above supplies performance evidence.

## Tests and records

The full suite passes **1,027 tests**; the focused suite passes **236 tests**.
The 46 new cases compare against a copy of the original nearest traversal whose
AST matches the baseline after renaming. They cover six shapes, three scales,
FMA enabled/disabled, face/edge/corner queries, adjacent floating values, strict
search-limit ties, extreme and nonfinite inputs/bounds and empty leaf tables.
Existing tests separately compare independent geometry and captured changing
domains, including failure and recovery.

Four focused Compute Sanitizer jobs pass with zero reported errors: the 46 new
cases and 20 surface/selection lifetime cases, each with padded allocations and
again with the pool disabled plus API-access/use-after-free checks. The
diagnostic stack is 128 KiB. This does not claim the entire suite ran under
sanitizer; the broader investigations #147/#155 remain open.

The [artifact manifest](../results/validation/vtm/surface-bound-reuse/manifest.json)
contains raw paired samples, exact commands, complete meshes, source/binary
hashes, independent checks, failed checker records, memory logs, resource
attributes and losslessly compressed authoritative Nsight/DOT archives.
SQLite exports are regenerable. Earlier probe files retain their baseline
source identity; they are not current-runtime tests.

This completes [#200](https://github.com/alecjacobson/warpower/issues/200).
The float32-filter experiment remains separate in
[#198](https://github.com/alecjacobson/warpower/issues/198), and the overall CPU
performance target remains open in [#40](https://github.com/alecjacobson/warpower/issues/40).

[Exact push-time rejection](surface-push-pruning.md) subsequently skips children
already rejected by the existing bound comparison, preserving the pop-time check
and nearest-hit arithmetic.
