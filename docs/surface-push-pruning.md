# Exact nearest-surface child rejection

Skipping already-rejected child pushes reduces complete uniform and graded 8K
scanned Armadillo replay time by **5.62% and 4.94%** in the final paired
measurements. Independently callable power-diagram and Delaunay
implementations and APIs are unchanged. This improves the Warp implementation;
the comparable-quality native CPU outperformance goal remains open.

## Traversal change

The existing nearest-surface traversal computes both child box distances, orders
the children, and saves their rounded distances on a stack. Previously it pushed
both children unconditionally and rejected unsuitable entries when popped.
It now pushes a child only when `not (distance >= result.distance2)`, retaining
the original pop-time check and far-then-near insertion order.

The current best squared distance changes only under a strict smaller-distance
update. A child already rejected by the same comparison cannot become acceptable
later. Removing that entry preserves the visit order and triangle arithmetic of
all surviving leaves. Keeping the negated `>=` comparison matters: substituting
`<` would change unordered NaN behavior. Pending entries still need the pop check
because an intervening leaf can improve the best distance.

There are no new or approximate bounds, new geometry terms, shorter stacks, device
arrays, host decisions or launches. The 128-entry stacks, root handling, strict
nearest-hit update, cutoff semantics and tie behavior remain. This differs from
the earlier unadopted tail traversal, which unconditionally stacked its deferred
child. See [the preceding investigation](surface-bound-investigation.md) and
[cached bound reuse](surface-bound-reuse.md).

The production function AST matches the private measured candidate. Reversing
the single push-block replacement restores the original module byte for byte.
The source audit binds 267 files; all 266 other source, test, benchmark, example
and installation files are unchanged. Existing nearest
tests retain their independent uncached reference. No global floating-point
atomic reduction is promised bitwise deterministic.

## Complete measurements

NVIDIA L40, Warp 1.19.0+cu12; baseline
`55f356c7aa8cf91d5143b1f392e3b96038c6462d` and runtime
`7bd8542d1292512b0594a6ac5e6c75451b77aa50`.
Warmed graph replay starts a fresh complete solve, including
refit, sampling, growth, fitting, quality work, recovery and extraction. Setup,
uploads and offline verification are outside replay and remain separately
reported. All 32 final complete/standalone jobs have clean first and timed
CPU-quota windows. These are GPU revision comparisons, not CPU speedup ratios.

| Complete example | Before, ms | After, ms | Time reduction |
|---|---:|---:|---:|
| torus16384-fixed32 | 4401.260 | 4383.826 | 0.40% |
| torus1000 | 495.604 | 488.537 | 1.43% |
| torus8192-fixed32 | 1362.481 | 1355.314 | 0.53% |
| torus8192-progressive | 2646.519 | 2630.720 | 0.60% |
| thin-hand-seed7 | 1226.490 | 1223.580 | 0.24% |
| cavity | 350.018 | 352.038 | -0.58% |
| disconnected | 479.084 | 478.567 | 0.11% |
| plate | 877.293 | 867.939 | 1.07% |
| armadillo-graded4096 | 2312.476 | 2227.191 | 3.69% |
| armadillo-graded8192-recovery | 3862.305 | 3671.334 | 4.94% |
| armadillo-uniform8192-recovery | 3672.135 | 3465.875 | 5.62% |

All pairs retain input arrays, roles, oriented material/boundary connectivity and
recorded solver work. Workspace pool usage is unchanged. The independently
measured minimum radius ratios agree within 1e-12.

Three pairs have coordinate differences; oriented full-hull connectivity remains
identical for every pair. Both meshes have valid current-geometry proofs.

| Case | Changed scalar coordinates | Maximum absolute difference |
|---|---:|---:|
| thin-hand-seed7 | 2 | 1.0408340855860843e-17 |
| cavity | 1 | 2.3283064365386963e-10 |
| disconnected | 1 | 4.6566128730773926e-10 |

These differences are observed; this study does not identify their causal kernel.

The mean radius ratio differs by at most 7.77e-13 across all pairs.
The retained variation review records coordinate witnesses and all other checked
quality deltas.

All 22 final meshes pass independent full-hull orientation, empty-sphere and
coverage checks, positive material geometry, closed oriented boundary edges,
single-cycle boundary vertex links and exact boundary self-intersection checks.
19 proofs transfer from prior independently checked meshes only after exact
array/oriented-connectivity/aligned-quality comparisons;
the others receive fresh checks. All fourteen gallery VTU/OFF exports round-trip
exactly. Surface distances use sampled quadrature, not a Hausdorff certificate.
The scan retains its [original Stanford/libigl provenance](scanned-armadillo.md).

| Independent routine/input | Before, ms | After, ms | Change |
|---|---:|---:|---:|
| delaunay1000 | 9.181 | 9.175 | -0.06% |
| delaunay100000 | 182.211 | 183.838 | +0.89% |
| delaunay-grid | 2.190 | 2.203 | +0.62% |
| delaunay-fitted-hand | 13.392 | 13.407 | +0.11% |
| diagram100000 | 18.853 | 18.151 | -3.72% |

Recorded standalone inputs, counts, errors and diagnostics match. These timing
differences do not establish a standalone speed gain. README retains installation
instructions and the earned Paragram flagship.

## Static probe and limits

All 64 static rows use twenty warmed samples and clean quota windows. Classifiers
include existing ordering/sorting, cache rebuilding and cold reset when requested;
they launch the complete `8*N` tet capacity with the real live count. Quadrature
includes resets and `32*N` reserved faces. Warm-cache rows reuse unchanged input.
Nearest-only rows query live cell centers and are not complete classifier timings.

| Static final mesh | Full cold, before/after ms | Selection cold, before/after ms | Quadrature, before/after ms |
|---|---:|---:|---:|
| torus1000 | 3.862 / 3.483 | 3.795 / 3.463 | 0.357 / 0.322 |
| torus16384 | 15.543 / 14.896 | 12.558 / 11.529 | 0.599 / 0.577 |
| armadillo-uniform8192 | 38.693 / 31.583 | 18.428 / 17.227 | 6.071 / 5.229 |
| armadillo-graded8192 | 35.551 / 28.816 | 18.183 / 15.612 | 6.563 / 5.417 |

The graded 8K retained cold classifier is 5.03% slower in the static probe;
that result is retained. Warm cache costs are nearly unchanged. Compiled nearest
and quadrature kernels report 80 registers and 3,296 local bytes per thread.
Those properties alone do not measure occupancy or establish complete-solve gains.
The four complete pilot pairs preserve all coordinates, oriented connectivity,
roles and work exactly, with zero additional pool bytes. The first candidate
constructor includes extra JIT compilation; no setup speedup is claimed.

## Qualification and reproduction

The private candidate passes 145 nearest/quadrature/extraction/selection cases
and 29 cache/ordering cases. Strong nearest fixtures compare full hit point,
face and squared distance across six shapes, three binary scales, both caller
contraction settings, strict limits, nonfinite data and empty trees. The cache
tests preserve unmodified ordinary callers as independent references and include
refits, changed counts/cutoffs, failure/recovery, epochs and nondefault streams.

The committed production suite passes all 1,387 tests. The unchanged complete
8K five-phase loop covers bounded topology-recovery exhaustion, success, invalid
target, collapsed input and recovery with stable pointers and cleared public
failure counts. Ordinary, native and sanitizer frames match independently
certified geometry and consumed diagnostics.

Native traces contain 93,800 full-solve and 158,566 loop kernel activities.
Activity and graph-DOT audits show only device-to-device copies, no host callbacks and no
replay allocation/free nodes. Each has one outer graph launch and final sync;
any initial graph instantiation/upload remains explicit in the API records.
Profiled timings are diagnostic, separate from clean performance measurements.

| Native full-solve kernel attribution | Calls in each run | Before, ms | After, ms |
|---|---:|---:|---:|
| selection_cached_ordered | 63 | 683.586 | 602.941 |
| classify_cached_ordered | 60 | 400.839 | 333.458 |
| surface_error | 60 | 371.542 | 313.034 |
| accumulate_samples | 534 | 495.627 | 500.669 |
| search_unproved_faces | 309 | 239.937 | 240.053 |

The preceding native runtime and current runtime differ only in the nearest-traversal
source within the recorded source scope. Equal calls accompany lower query-kernel
time; this attribution is consistent with the separate complete replay results.

Four isolated sanitizer jobs pass with zero errors: 116 nearest/cache/ordering/
quadrature cases and the complete loop, each with padding and with pool-disabled
allocation-lifetime checks. The full-loop audits take 298.5 s with padding and
491.0 s with lifetime tracking.
The diagnostic stack allowance is recorded in the isolated runner; production
timings do not use that override.

The [manifest](../results/validation/vtm/surface-push-pruning/manifest.json)
binds candidate and production source, raw samples, meshes, exact commands,
native traces, memory logs and shared proof dependencies. Restore the retained
`.work/vtm` paths at the pinned revisions, use the documented Warp environment,
and run private contracts/static/pilot, production tests, paired timings, then
native/memory/geometry/frame checks. Run one GPU job at a time and keep heavy CPU
work outside clean timing windows. Existing output paths are never overwritten.

This completes [#253](https://github.com/alecjacobson/warpower/issues/253).
Broader query performance and comparable-quality native CPU comparisons remain
tracked in [#197](https://github.com/alecjacobson/warpower/issues/197) and
[#40](https://github.com/alecjacobson/warpower/issues/40).
