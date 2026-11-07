# Shared certified wall reconstruction

The VTM reconstruction routes now try bounded Local DeWall candidates after an
absent or unsuccessful temporal fragment. Current exact seed filtering and full
facet completion must certify those candidates before publication. On the L40,
the complete thin-hand and cavity examples improve substantially; the 4k torus
improves modestly. The quality-matched wide CPU speedup goal remains unmet.

## Construction and proof lifetime

This adopts the [earlier candidate experiment](local-dewall-candidates.md) with a
smaller integration. The approximate float32 wall kernels remain untrusted.
Finite artificial startup vertices, bounded traversals and failed local closure
can omit or propose invalid tetrahedra. They never establish mesh acceptance.
The existing original-coordinate symbolic seed filter, exact predicates and full
certificate determine validity. Overflow, empty proposals, invalid candidates or
exhausted completion select ordinary on-device construction.

Temporal reuse runs first. Once it has consumed its preceding fragment, the wall
builder borrows that fragment's array and count as bounded candidate storage
(up to eight candidate cells per reserved site). It also shares the existing
seed filter and completion workspace. Only new normalized positions and per-site
wall state are allocated. Every successful output is remembered as a complete
certified fragment after publication; rejected proposals cannot become history.

The wall completion has an eight-round limit. Its current proof publishes cells,
counts, the face index, face occurrences and balance, hull boundary, vertex usage,
and certificate statistics together. This lets downstream poles and future reuse
consume matching metadata, and removes the outer duplicate certificate. The
ordinary constructor retains its own complete checks when the wall fails. A
proof established for preceding positions or canonical owners is never accepted
for the next input without checking it again.

The dispatch changes only private VTM/routed reconstruction: 256–2,048-site VTM
workspaces, preallocated 1,024/2,048 prefix routes, and full routes through 4,096
sites. Smaller private reuse retains its old storage. Full routes above 4,096
retain ordinary construction. The independently callable power-diagram, regular
triangulation and Delaunay APIs retain their source and defaults.

## Complete solves

Fresh-process pairs compare runtime `6eb1882d8af1` with `cc0a020f5f7c` on an
NVIDIA L40, Warp 1.19. Each median covers a complete fresh captured solve, including
sampling, progressive growth, fitting, refinement, quality work and extraction.
Torus 1k uses five warmed samples; the other seven cases use three. Compilation,
setup and offline checks are excluded; first-call and setup measurements are
retained separately. All first and timed CPU-quota windows are clean.

| Example | Before ms | After ms | Reduction |
|---|---:|---:|---:|
| torus1000 | 640.993 | 622.803 | 2.84% |
| torus4096 | 1854.409 | 1833.974 | 1.10% |
| graded-torus | 2118.251 | 2104.104 | 0.67% |
| thin-hand | 2267.283 | 1342.333 | 40.80% |
| cavity | 866.913 | 442.557 | 48.95% |
| disconnected | 600.784 | 497.206 | 17.24% |
| armadillo-uniform | 3101.604 | 2999.924 | 3.28% |
| armadillo-graded | 6675.084 | 6583.771 | 1.37% |

The [paired results and independent validation](../results/validation/vtm/shared-wall/shared-wall-final-independent.json)
retain every coordinate difference, topology/work comparison and quality metric.
All full/material cell sets, oriented boundaries, roles, parameters, input arrays,
counts and solver work match. Seven outputs have bitwise-identical positions; thin-hand differs in one
coordinate by 1.6263e-19. Every difference is retained in that record. All eight
outputs pass independent CGAL global empty-sphere, positive-orientation, face and
hull checks, independent NumPy quality measurements, CGAL AABB surface-distance
measurements and boundary vertex-link checks. Six gallery VTU files round-trip
exactly and their boundary surfaces have zero checked self-intersection pairs.
The sampled surface distances are not a Hausdorff certificate.

Torus workspace storage increases from 273,863,500 to 273,947,644 bytes at 1k and
from 1,743,416,484 to 1,744,019,028 bytes at 4k: additions of 84,144 and 602,544
bytes. The original experimental wrapper added 8.8 and 62.8 MB respectively.
An intermediate wrapper removed duplicate certification but still duplicated
completion storage; its negative memory comparison is retained with the pilots.

## Independent APIs

| Routine/input | Before ms | After ms | Change |
|---|---:|---:|---:|
| delaunay1000 | 9.198 | 9.187 | -0.12% |
| delaunay100000 | 182.527 | 182.440 | -0.05% |
| delaunay-grid | 2.199 | 2.192 | -0.29% |
| delaunay-fitted-hand | 13.416 | 13.430 | +0.10% |
| diagram100000 | 18.155 | 18.160 | +0.03% |

These unchanged routines have matching output counts, status and diagnostic
records in the [fresh comparison](../results/validation/vtm/shared-wall/shared-wall-final/commands.json).
The small timing differences measure run-to-run variation, not an attributed
algorithmic gain. Repeating identical inputs is separate from VTM's changing-site
reconstruction workload.

## Validation and capture

All 844 ordinary tests pass, including 75 focused reconstruction tests. The new
13 cases cover changing producers/consumers, canonical-owner promotion, invalid
input and recovery, extreme finite scales, malformed/overwritten counts, empty
and overflowing proposals, filter/completion failure, insufficient final output
capacity, stable addresses and publication of the current face/hull index.
A poisoned outer certificate establishes that an accepted wall proof takes the
intended direct-publication branch; an independent face-index comparison checks
that its metadata matches the output cells.

Six fresh-process Compute Sanitizer jobs cover all 13 new cases plus the routed
2,050-site changing-input test, once with padded pooled allocations and once
with pool-disabled API/use-after-free tracking. Every job reports zero errors
with a 128 KiB diagnostic stack. The first combined padded run was interrupted
after 97 seconds with nine pytest passes while spending substantial time in
setup synchronization. It is retained as **incomplete**, without a final sanitizer
summary. Smaller fresh groups then completed the entire selected coverage. The
unresolved setup-delay investigation remains in #155; these observations do not
establish its cause.

The [full-solve native audit](../results/validation/vtm/shared-wall/shared-wall-full-native-audit.json)
and [changing-input audit](../results/validation/vtm/shared-wall/shared-wall-loop-native-audit.json)
each show one graph launch and one final audit synchronization. Neither graph
contains host transfer, host callback, allocation or free nodes; no explicit CUDA
allocation/free API is observed. All 3,652 full-graph and 288 loop-graph copy nodes
are device-to-device. This first execution also includes native graph
instantiation/upload. Nsight can omit graph-copy activity records, so the audit
checks both the observed activity and every DOT copy node.

The complete solve executes eight wall initializations/walks and returns the
expected valid 4,096-site mesh. The second trace contains eight phases in one
device-controlled loop: full input, a small move, a 512-site prefix, unrelated
input, invalid active count, recovery, NaN input, and recovery again. All six
valid outputs match independent SciPy connectivity and mesh checks; both invalid
phases publish zero tetrahedra. Fresh full inputs accept the wall path, the small
move accepts temporal reuse, and recovery accepts freshly certified walls.
The trace records 62,280 and 1,285 kernel activities respectively. Profiling is
structural/cost evidence, separate from the warmed timing comparison above.

All candidate kernels are pure Warp. Setup creates fixed storage and captures
the conditional branches; replay makes retry, completion and publication decisions
from device values. Setup synchronization and offline verification are outside
the device-loop contract. The existing full allocation-order sanitizer limitation
for native CUB/routed setup remains documented in #147/#155; passing use-after-free
checks does not claim that stronger mode passed.

The wall traversal is derived from Gao and Chen's MIT implementation, pinned and
attributed in [the earlier source audit](local-dewall-candidates.md),
[NOTICE](../NOTICE) and [LICENSE-Local-DeWall](../LICENSE-Local-DeWall).
The full Local DeWall paper was not obtained; no performance claim about the
authors' CUDA binary is made. The built wheel includes the new modules and license.

The [artifact manifest](../results/validation/vtm/shared-wall/manifest.json)
records source hashes, commands, complete production mesh pairs, pilot summaries
and native traces. Earlier pilot raw meshes remain private; their hashes are
listed in `retained_private_pilot_meshes`. The initial publication manifest
included a still-open stdout log; its corrected closed snapshot and the original
manifest are retained. The final manifest is authoritative for published paths. Large text
artifacts have verified lossless gzip copies. To rerun the drivers, restore the
listed scripts/helpers under `.work/vtm`, configure the baseline worktree noted
in the commands, and use new output directories. The original private validator
ran out of disk while expanding derived query CSV; its resumed validation checks
input hashes and uses regenerable scratch outside the persistent results volume.
The failed script/log and corrected resume are retained. These are validation
harness failures, not accepted meshing failures.
