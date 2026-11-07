# Exact exterior witnesses in hull-support traversal

Private facet completion stops testing a face as soon as an exact exterior
witness disproves its hull support. Other tree partitions can observe the same
negative result and stop work on that face too. Every thread still processes
its later strided face slots. Ordinary standalone Delaunay certification and
power-diagram kernels, allocations and launch defaults are unchanged.

## Proof and error handling

Support of an oriented face requires every canonical site's exact orientation
to be nonpositive. One valid positive witness refutes that statement. The private
support mask starts at one for each current boundary ordering and is only
atomically cleared to zero. Reads are atomic because another partition can clear
the same slot concurrently. Sites, canonical owners, tree bounds and boundary
ordering remain fixed throughout the certificate.

A positive support result still requires every live partition to finish without
a witness or fatal error. The completion gate rejects encountered arithmetic,
topology and stack errors. The final original-coordinate certificate still
checks orientation, spheres, face pairing, coverage and hull support. No mask
survives a call to `run()`; the next certificate resets it before reuse.

This shortcut does not exhaustively diagnose skipped malformed inputs. After a
conclusive negative support proof, unvisited sites or tree nodes cannot supply
additional diagnostic bits. The private caller's finite-site/current-valid-BVH
contract still applies. An unresolved predicate cannot establish positive
support. A fatal stack overflow invalidates the whole candidate, including any
later strided slots the failing thread did not reach. The ordinary standalone
certifier retains its preceding traversal and diagnostics.

The only changed runtime function is `record_face_support` in
[`_proved_completion.py`](../warpower/_proved_completion.py). Host allocations,
launch dimensions and graph structure are unchanged. This builds on the
[preceding support-mask reuse](hull-support-reuse.md) and
[same-input cell proofs](completion-proofs.md).

## Complete-solve measurements

The following comparisons use baseline `7c459bb` and runtime `32ee3c6` on L40 /
Warp 1.19.0. Each median contains three warm, complete fresh resident solves,
including domain refit, initial sampling, progressive fitting/refinement, four
quality rounds and extraction. Setup, compilation, transfer, validation and
export are outside replay timing. Inputs, seeds and budgets match in each pair.

| Complete resident solve | Before (ms) | After (ms) | Speedup |
| --- | ---: | ---: | ---: |
| Torus, 1,000 sites | 1,192.054 | 1,086.928 | 1.097× |
| Torus, 4,096 sites | 7,793.917 | 7,629.272 | 1.022× |
| Graded torus, 1,000 sites | 1,968.800 | 1,750.878 | 1.124× |
| Thin hand, quadrature 20/20 | 2,616.242 | 2,472.700 | 1.058× |
| Cavity, quadrature 20/20 | 1,132.295 | 1,021.415 | 1.109× |
| Disconnected, quadrature 20/20 | 896.919 | 725.503 | 1.236× |

All six pairs retain full/material cell sets, oriented boundaries, roles and
iteration counts. Five have bitwise-identical positions; the hand's largest
coordinate difference is `1.93e-18`. Independent CGAL global full-hull checks and
NumPy material checks pass for all six; gallery VTU round trips and boundary
self-intersection checks pass. These measured speedups are against the previous
GPU implementation, not a CPU mesher.

The runtime allocates no additional workspace. The measured pool deltas remain
272,938,828 bytes at capacity 1,000 and 1,706,590,013 bytes at capacity 4,096.
The first new 1,000-site setup includes fresh Warp compilation and takes 61.092 s;
that record is preserved. A separate warm-cache setup takes 1.488 s, versus
1.466 s for the baseline. The 4,096-site setup measures 2.733 / 2.747 s. Setup
conditions are separate from the warm replay comparison.

Standalone Delaunay before/after medians are 9.185856 / 9.174240 ms at 1,000
random sites, 181.552383 / 182.895325 ms at 100,000, 2.188768 / 2.199296 ms for
the 64-site cospherical grid, and 13.404768 / 13.408448 ms for the fitted hand.
The 100,000-site diagram measures 18.160831 / 18.198352 ms. These changes are
within 0.75% on unchanged paths and are treated as timing variation. All six
paired solve windows and the standalone timing windows are unthrottled.

The initial local-only prototype (`03ec6a3`) measured 1,097.942 ms for the
1,000-site torus and 7,641.411 ms for 4,096 sites. An initial shared-witness
prototype measured 1,084.990 ms at 1,000 sites. Those pilot records are retained
separately from the final paired measurements.

## Validation status

The full suite passes 577 tests. Thirteen support tests include an independent
rational oracle, coplanarity, duplicates, scales from `2^-60` to `2^60`, every
tree partition, early leaves and more face slots than launch workers. New replay
cases move the exterior witness between faces, use an empty count, then restore
the input. Every strided slot is compared with rational signs and the preceding
exhaustive kernel. Other tests retain failed-proof gates, nonfinite rejection,
malformed cyclic-tree overflow and independent final Delaunay certification.

All 47 focused cases pass unfiltered padded Compute Sanitizer checks: 13 support,
24 cell-proof, seven reuse and three refinement cases, each group in its own
process. No kernel is filtered or error suppressed. A separate six-phase device
producer/core/consumer lifetime audit disables the allocation pool and enables
API memory-access and stream-ordered race checks; it reports zero errors. The
existing five-phase complete mesher loop also passes changing inputs, invalid
inputs and restoration. Diagnostic processes use a 128 KiB CUDA stack allowance;
the runtime sets no such allowance. See the separately tracked
[instrumentation observation](https://github.com/alecjacobson/warpower/issues/147).

The native trace covers three complete fresh torus1k solves, with 98,982 kernel
activities, three graph launches and one final synchronization. Before and after
have identical counts and graph node types: 11,756 kernels, 4,358 memsets, 1,337
conditionals and 994 device-to-device copies. No host callback, allocation/free,
or host/device transfer occurs during replay. All DOT copy nodes and observed
native copy directions are checked together, since Nsight can omit graph copy
activity records.

| Three-solve profile | Before (ms) | After (ms) |
| --- | ---: | ---: |
| Private hull-support checks, 951 calls | 575.100 | 254.413 |
| All 98,982 kernel activities, summed | 3,574.729 | 3,253.929 |

These are instrumented kernel sums, separate from the clean complete-solve
medians. The reduced support cost accounts for almost all of the profile's gain.

Both scanned Armadillo meshes are bitwise identical to the published uniform
and graded results and pass independent full-hull, material, self-intersection
and VTU checks. Their complete resident medians are 20.544 s and 20.599 s. The
initial uniform batch had four quota-throttled periods during warm replay and
is excluded from final timing. A complete fresh batch was repeated with zero
throttling; both original and replacement records are retained. These scanned
runs are additional regression checks, not fresh before/after speedup pairs.

The coarse quadrature-4 cavity and disconnected cases remain valid; the coarse
hand retains its existing error 64 and zero public counts. Across the six paired
cases, two scans and two valid coarse cases, ten full hulls receive independent
CGAL checks and NumPy material validation. Every gallery VTU reproduces its saved
arrays exactly, and every gallery material boundary has zero self-intersections.

The [comparison record](../results/validation/vtm/hull-stop-summary.json) retains
raw timing references and independent measurements. The
[native audit](../results/validation/vtm/hull-stop-native-audit.json),
[before profile](../results/validation/vtm/hull-stop-before-kernels.json),
[after profile](../results/validation/vtm/hull-stop-after-kernels.json), and
[artifact manifest](../results/validation/vtm/hull-stop-provenance.json) pin graph
structure, source hashes and validation evidence. The source audit confirms that
only the private support function changes; host allocations and standalone
paths are identical. No CPU outperformance is established, and the main README
is unchanged.
