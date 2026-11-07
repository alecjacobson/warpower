# Surface-query validity for private material selection

Selective VTM recovery needs the retained cells, their quality, and extraction
validity. It does not use the recovered boundary's RMS or maximum distance.
The private `_SelectionExtractionWorkspace` now checks whether those surface
queries can succeed without computing an unused nearest distance at every
sample. It shares the public extractor's classification, compaction, topology,
containment/exterior recovery, volume and final validity schedule.

Only the private surface-metric hook changes. Its three unused distance fields
(`_surface_integral`, `surface_rms_distance`, and `surface_max_sample_distance`)
are zero and explicitly unavailable. The public `ExtractionWorkspace`, quality
moves and final mesher outputs still compute full diagnostics. Circumcenter
classification distances and exterior-band behavior are unchanged. There are
no additional device arrays, public options, or runtime Python substitutions.

## Error-preserving witness

For each of the original three quadrature samples, the private kernel evaluates
the existing `closest_triangle` arithmetic on the first current sorted triangle.
A finite squared distance strictly below the original search limit proves that
the full traversal must find a finite hit. It does not estimate the nearest
surface or infer that the witness triangle is geometrically closest.

The proof uses the current validated BVH. Its boxes are min/max bounds of finite
float32 coordinates. Boundary samples are the original double-precision convex
combinations of finite float32 site coordinates. Their magnitudes are below
3.5e38, so every initial squared box distance is below 1.5e78, far below the
original squared limit of approximately 1e300. Before a hit, the full traversal
cannot prune the witness leaf. It therefore either finds another finite hit
first, or evaluates this same finite leaf computation. Once a finite hit is
found, the strict distance comparison cannot replace it with NaN or infinity.
The existing Morton/index radix-depth bound keeps the traversal stack bounded.

When the witness is unavailable, the kernel executes the original complete
`closest_point` query and applies its original error predicate. Invalid domain,
source and geometry checks still run, as do all recovery and topology checks.
Producers must refit the domain after changing its inputs, as before; arbitrary
externally corrupted tree topology is not a new supported interface. All tests
and branches execute on device using fixed setup storage and captured kernels.

## Measured scope

An observer that retained every diagnostic counted all extraction calls on eight
complete examples, twice each. Only the graded Armadillo trajectory used full
recovery extraction inside selection: four current-selection calls and 67
quality-guard calls out of 110 total extractions. The other calls were quality
initial/trial/restore (8/10/4) and mesher extraction (17). The other seven examples
had no selective recovery extractions. Quality moves consume RMS distance and
the mesher publishes it, so those callers keep the full metric path.

The private pilot improved the complete graded scan from 5293.261 to 4375.192 ms
with bitwise-identical positions, matching topology/work and independent
geometry validation. Its kernel was preloaded outside the setup timer, so that
pilot does not support a setup-speed comparison. Final production measurements
are reported separately below.

## Complete solves

Fresh-process production pairs compare merged baseline `6f1601e2f960f` with
runtime `987568a054a0` on the NVIDIA L40, Warp 1.19. Each timed replay includes
refits, sampling, progressive growth, optimization, quality work and final
extraction. Torus 1k uses five warmed samples; the other examples use three.
Setup/compilation and offline validation are excluded, with setup and first
replay measurements retained separately. All first and timed CPU-quota windows
are clean.

| Example | Before ms | After ms | Time reduction |
|---|---:|---:|---:|
| torus1000 | 617.103 | 616.497 | 0.10% |
| torus4096 | 1814.934 | 1814.349 | 0.03% |
| graded-torus | 2076.286 | 2072.132 | 0.20% |
| thin-hand | 1335.679 | 1334.453 | 0.09% |
| cavity | 441.900 | 441.600 | 0.07% |
| disconnected | 498.260 | 498.456 | -0.04% |
| armadillo-uniform | 2946.325 | 2950.636 | -0.15% |
| armadillo-graded | 5299.126 | 4369.422 | 17.54% |

The graded scan improves **1.213x (17.54% less time)**. The other seven examples
have no selective recovery calls in the preceding observer; their small changes
are ordinary run-to-run variation. All eight pairs preserve input arrays,
parameters, iteration/refinement work, roles, full/material tetrahedron sets and
oriented boundaries. Five point arrays are bitwise identical. Thin-hand changes
three coordinate values, cavity and disconnected one each, with a maximum
absolute change of 4.657e-10. Exact values are recorded in the
[paired independent checks](../results/validation/vtm/recovery-diagnostics/recovery-diagnostics-final-independent.json).

Every final mesh passes independent CGAL global empty-sphere, orientation, face
and convex-hull checks; NumPy quality measurements; CGAL AABB sampled surface
distances; and boundary vertex-link checks. All six gallery VTU files round-trip
exactly, and their boundaries have zero checked self-intersection pairs. These
sampled distances are not a Hausdorff certificate. A separate
[public-quality comparison](../results/validation/vtm/recovery-diagnostics/recovery-diagnostics-public-quality.json)
checks every reported field in all eight pairs: integer/topological fields and
quality percentiles match exactly; the largest floating reduction difference is
7.461e-14. Public surface RMS remains available and nonzero in every example.
The private workspace adds no arrays; measured torus pool sizes are unchanged.

## Independent APIs

| Routine/input | Before ms | After ms | Change |
|---|---:|---:|---:|
| delaunay1000 | 9.189 | 9.180 | -0.10% |
| delaunay100000 | 182.432 | 182.858 | +0.23% |
| delaunay-grid | 2.208 | 2.197 | -0.50% |
| delaunay-fitted-hand | 13.404 | 13.410 | +0.05% |
| diagram100000 | 18.795 | 18.157 | -3.39% |

These routines' code and storage are unchanged, as are all recorded inputs,
counts, errors and diagnostic fields. The largest observed slowdown is 0.24%;
the faster diagram measurement is also variation and is not attributed to the
selection optimization. The main README retains its existing Paragram result.

## Validation and capture

The full suite passes **894 tests**; the focused selection/extraction/control
suite passes **82 tests**. The 43 new tests cover ordinary and extreme samples,
scales 1e-30 through 1e30, missing/nonfinite witness leaves, original-query
fallback, preexisting errors and inactive padded entries. Captured changing
site/domain/control phases compare the private extractor with a separately
constructed public full extractor, preserve buffer addresses, and forbid host
readback, allocation and synchronization during capture. A dedicated test
verifies that private distance fields reset to zero while public RMS/max remain
available.

The first production fixture mutated `domain.points` after construction rather
than the separate `domain.tree.points` snapshot actually queried by the kernel.
It therefore failed to force the intended missing-leaf case: six tests failed
and 70 passed. The corrected fixture mutates and verifies the queried snapshot;
the initial source/log and corrected run are retained. Earlier private
missing-witness cases had the same limitation; their nonfinite-query cases did
exercise fallback. These failed/incomplete probes are not counted as validation
of a missing witness.

Four focused Compute Sanitizer jobs pass with **zero reported errors**: 36
primitive cases and seven capture/public-contract cases, each with padded
allocations and again with the pool disabled plus API-access/use-after-free
checks. The diagnostic stack is 128 KiB. This is focused memory coverage, not a
claim that the entire suite ran under the sanitizer; #147/#155 remain open.

The native complete graded-scan capture has 73,845 kernel activities, 38,112
kernel nodes and 3,654 device-to-device copy nodes. A separate eight-phase
producer/domain-refit/selection/full-reference/consumer loop has 1,130 kernel
activities and three copy nodes. It covers ordinary/recovered selection,
changed scale, invalid sites/domain, source failure and recovery. All selection
errors, masks, counts and quality agree with the separately allocated public
reference; recovered valid phases retain nonzero public RMS and zero private
unavailable RMS.

Both traces include first native graph instantiation/upload, one graph launch
and one final audit synchronization. Neither contains host transfers/callbacks,
graph allocation/free nodes or explicit allocation/free API calls. All DOT copy
nodes and observed copy activities are device-to-device; both are inspected
because Nsight may omit graph copy activities.

The graded-scan native trace attributes the saved work directly:

| Kernel | Baseline calls | New calls | Baseline activity ms | New activity ms |
|---|---:|---:|---:|---:|
| classify | 110 | 110 | 2450.768 | 2447.363 |
| surface_error | 110 | 39 | 1424.468 | 502.444 |
| selection_surface_status | 0 | 71 | 0 | 0.660 |

The baseline is the retained graded-scan profile from #193. These first-launch
profiles explain the optimization and are separate from the clean warmed
performance pairs. Full nearest-distance work remains at the 39 consumers that
need it. Classification is now the largest remaining cost on this trajectory;
[#194](https://github.com/alecjacobson/warpower/issues/194) tracks whether its
private distance requirements can also be reduced with error equivalence.
The bounded boundary-sample witness proof above does not automatically apply to
potentially unbounded circumcenters.

The [artifact manifest](../results/validation/vtm/recovery-diagnostics/manifest.json)
records commands, source hashes, final meshes, independent reference hashes,
failed probes, memory logs and authoritative Nsight/DOT archives. SQLite exports
are regenerable from those archives. Earlier observer/pilot meshes remain
privately retained with hashes; final production counterparts are published.
Private launchers target their recorded baseline, not the production subclass.

**The overall CPU outperformance goal remains unmet (#40).** These are speedups
between Warp revisions. Comparable-quality native CPU measurements remain
required; the 4k torus still takes about 1.814 seconds and the earlier CPU
baselines have better lower-percentile/surface quality.
