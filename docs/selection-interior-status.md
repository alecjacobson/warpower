# Interior classification for private material recovery

The private selection extractor needs retained cells, physical geometry and
quality, and the sign of each circumcenter's surface distance. Its exterior-band
recovery tests only whether that distance is nonpositive; containment recovery
does not consume it. Public extraction, quality moves and final mesher output
continue to use the original full-distance classifier.

The private classifier now omits nearest-surface traversal for a proved inside
center. Its `signed_distance` buffer is explicitly a nonpositive/positive sign
marker: shortcut interiors store `-1`, whose magnitude is unavailable. The
previous private RMS/max/integral fields remain zero and unavailable. Public
`ExtractionWorkspace` retains actual signed distances and all diagnostics.

The shared recording schedule has a private classification hook. Geometry,
physical volume/quality, compaction, topology, containment/exterior transactions
and final validity run as before. The private kernel preserves the original
geometry arithmetic; the original public kernel is unchanged. There are no new
device arrays, public controls or runtime kernel substitutions.

## Success witness and fallback

A center can use the shortcut only when every absolute component is below
`1e140`, and the existing `closest_triangle` computation on the first current
sorted triangle returns a finite squared distance below the original query's
initial squared bound, approximately `1e300`.

For the current validated tree, all box coordinates are finite float32 values.
Every initial squared box distance from such a center is below `4e280`, so the
original full traversal cannot prune the witnessed leaf before finding a
finite hit. It either reaches that leaf and performs the identical computation,
or finds another finite hit first. A strict distance comparison cannot replace
a finite hit with a nonfinite one. The existing bounded traversal stack is
unchanged. This establishes query success, not a nearest-distance estimate.

The shortcut additionally requires the existing `material_side` query to return
exactly one. The original full classifier would then either find exact zero
and force inside, or find a positive distance and obtain the same inside result.
Both keep the cell for any supported tolerance and give a nonpositive distance
to exterior-band recovery.

All outside, ambiguous, unproved and unsupported centers execute the complete
original nearest query. Exact zero still forces inside; positive distances use
the original strict `distance / radius < tolerance` test. An already-computed
ray result is reused only when valid for that same center and current tree.
There is no shortened outside search or rounded threshold approximation.
Domain producers must refit after changes, as before. Invalid domain/source
inputs still fail before classification.

## Caller and query attribution

The preceding [surface-status audit](selection-surface-status.md) found no
private recovery extractions on seven of the eight complete examples. The
graded Armadillo trajectory has 71: four current-selection and 67 quality-guard
extractions. Its remaining 39 full extractions need physical distances and
retain them.

An observer on merged baseline `f338eb5` kept every original classification
query and ran the complete graded trajectory twice. Both replays matched
positions bitwise, topology, roles and solver work. Each replay recorded
1,751,758 private classification queries: 1,328,296 inside and 423,462 outside.
Every center passed the bound and finite witness; there were no missing hits,
exact-zero distances or ambiguous rays on this particular trajectory. Query
counts do not imply an equal fraction of saved runtime.

The private pilot improved the complete graded scan from 4392.469 to 3720.182 ms
with identical positions/topology/work and independent mesh validation. The
candidate module was preloaded outside setup timing, so this does not support
a setup-speed claim. Production measurements follow separately.

## Complete solves

Fresh-process production pairs compare merged `f338eb5c5d90` with runtime
`57c42cb5d196` on NVIDIA L40 and Warp 1.19. Each replay starts a fresh complete
solve, including refits, sampling, growth, optimization, quality work and
extraction. Torus 1k uses five warmed samples; the other examples use three.
Setup/compilation and offline validation are excluded. Setup and first replay
measurements are retained separately. All first and timed CPU-quota windows
are clean.

| Example | Before ms | After ms | Time reduction |
|---|---:|---:|---:|
| torus1000 | 616.252 | 616.776 | -0.09% |
| torus4096 | 1813.102 | 1817.810 | -0.26% |
| graded-torus | 2069.575 | 2083.423 | -0.67% |
| thin-hand | 1332.713 | 1336.195 | -0.26% |
| cavity | 441.404 | 441.380 | 0.01% |
| disconnected | 498.584 | 500.421 | -0.37% |
| armadillo-uniform | 2964.701 | 2956.736 | 0.27% |
| armadillo-graded | 4389.586 | 3718.738 | 15.28% |

The graded scan improves **1.180x (15.28% less time)**. The seven examples without
private recovery calls remain within 0.67% timing variation; their changes are
not attributed to this optimization. Inputs, parameters, roles, complete and
material tetrahedron sets, oriented boundary faces and solver work match in all
eight pairs. Six position arrays are bitwise identical. Thin-hand changes one
coordinate by 2.711e-20; disconnected changes one by 4.657e-10. The exact values
are retained in the
[paired independent checks](../results/validation/vtm/recovery-classification/recovery-classification-final-independent.json).

All final outputs pass independent CGAL global empty-sphere, orientation, face
and convex-hull checks, NumPy quality, CGAL AABB sampled distances and boundary
vertex-link checks. The six gallery VTUs round-trip exactly, with zero checked
boundary self-intersection pairs. Sampled distances are not a Hausdorff
certificate. A separate
[public-quality comparison](../results/validation/vtm/recovery-classification/recovery-classification-public-quality.json)
checks every reported field across all eight pairs: discrete values and quality
percentiles match exactly; the largest floating reduction difference is
6.395e-14. Public RMS remains available and nonzero. Measured torus workspace
pool sizes are unchanged; no persistent device arrays are added.

## Independent APIs

| Routine/input | Before ms | After ms | Change |
|---|---:|---:|---:|
| delaunay1000 | 9.174 | 9.178 | +0.04% |
| delaunay100000 | 182.670 | 183.687 | +0.56% |
| delaunay-grid | 2.199 | 2.208 | +0.43% |
| delaunay-fitted-hand | 13.451 | 13.439 | -0.09% |
| diagram100000 | 18.211 | 18.208 | -0.02% |

These routines' source and storage remain unchanged. All recorded input hashes,
counts, errors and diagnostic fields match. The largest observed slowdown is
0.56%, within the existing nonregression threshold; no standalone speed gain is
claimed. README retains installation and the earned Paragram flagship result.

## Tests and capture

The full suite passes **981 tests**; the focused classification, selection,
exterior, control and capture suite passes **157 tests**. The 87 new cases compare
private decisions with original full classification across scales 1e-30 through
1e30, exact face/edge/corner centers, strict threshold ties, needle/flat geometry,
padded inactive entries and invalid inputs. Direct query cases cover centers
around the 1e140 proof boundary and 1e150 original search limit, overflow/nonfinite
centers, unavailable witnesses, empty leaf tables, all-three-ray ambiguity and
non-binary winding. Captured changing-input tests keep stable buffers and public
diagnostics; a dedicated test distinguishes the private sign marker from an
actual public signed distance.

The initial production fixture assumed two nested outward-positive shells
would be accepted as a domain. The existing domain orientation check correctly
rejects them with error32; that run had 12 failed and 145 passed tests. The
corrected fixture explicitly checks domain rejection and both workspaces'
early failure, then uses direct tree queries to test the non-binary winding
error. Its original source/log are retained. The earlier private probe also
retains a failed beyond-float32 fixture assertion: scale 2^100 was insufficient,
and the corrected fixture uses 2^120. Neither fixture correction changed runtime
code. These failed runs are not counted as successful checks.

Four focused Compute Sanitizer jobs pass with **zero reported errors**: all 87
new classification cases and seven capture/public-contract cases, each with
padded allocations and again with the pool disabled plus API-access/use-after-
free checking. The diagnostic stack is 128 KiB. This is focused coverage, not a
claim that all 981 tests ran under the sanitizer; #147/#155 remain open.

The native complete graded-scan capture records 73,845 kernel activities,
38,112 kernel nodes and 3,654 device-to-device copy nodes, unchanged from the
baseline graph's counts. A separate eight-phase producer/domain-refit/selection/
full-reference/consumer loop records 1,130 kernel activities and three copy
nodes. Its device consumer compares nonpositive signs and exact positive
distances with the public extractor, along with errors, masks, counts and
quality. It covers ordinary/recovered selection, scaled geometry, nonfinite
sites/domain, source failure and recovery; all comparisons pass.

Both traces include first native instantiation/upload, one graph launch and one
final audit synchronization. Neither contains host copies/callbacks, graph
allocation/free nodes or explicit allocation/free API calls. All DOT copy nodes
and observed copy activities are device-to-device; both are checked because
Nsight may omit graph copy activities. The loop includes public reference
queries and is structural validation, not a performance measurement.

Separate first-launch native activity explains the graded-scan gain:

| Classifier | Baseline calls | New calls | Baseline activity ms | New activity ms |
|---|---:|---:|---:|---:|
| Full `classify` | 110 | 39 | 2447.363 | 871.587 |
| Private `classify_selection` | 0 | 71 | 0 | 929.830 |

Total classification activity decreases from 2447.363 to 1801.417 ms, about
646 ms. The 39 full boundary-distance calls remain, with 503.434 ms of activity;
71 boundary-status calls cost 0.659 ms. Both classifiers still use 255 registers
per thread at block size 32. These profiles are separate from the clean warmed
performance pairs above. Further physical-geometry/query separation remains
tracked in [#159](https://github.com/alecjacobson/warpower/issues/159).

The [artifact manifest](../results/validation/vtm/recovery-classification/manifest.json)
records exact commands, source hashes, meshes, independent-tool hashes, failed
probes, memory logs and authoritative Nsight/DOT archives. SQLite exports are
regenerable. The private pilot meshes remain retained with hashes; complete
production meshes are published. Private prototype/observer launchers require
their recorded baseline, not the new production hook.

The overall comparable-quality CPU outperformance goal remains unmet (#40).
This work measures an improvement between Warp revisions.
