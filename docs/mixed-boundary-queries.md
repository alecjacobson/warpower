# Mixed boundary nearest-site queries

`BoundaryWorkspace(..., query_precision="mixed")` uses conservative FP32 lower
bounds during the site-hierarchy search, then evaluates possible winners in FP64
with the reference index tie rule. `VTMWorkspace` and
`VariationalMeshingWorkspace` accept `boundary_query_precision="mixed"`; the gallery
accepts `--boundary-query-precision mixed`. The default remains `float64`.
Masses, moments, energy, corner matching and topology arithmetic remain unchanged.
This option searches moving sites for fixed surface/crease samples, independently
of the input-triangle surface-query backend.

## Conservative pruning

Input points, samples and hierarchy bounds are binary32. Let `u=2^-24` be the
binary32 unit roundoff. For finite FP32 squared-distance result `d`, the lower
bound is `max(0, d*(1-64*u)-1e-37)`. Nonfinite results return zero, retaining the
node or candidate for FP64 evaluation. Normal-range subtraction, squaring and
three-term addition consume fewer than eight unit roundoffs of relative error;
the 64-unit allowance also covers rounding in the slack calculation. The absolute
term exceeds eight minimum-normal binary32 values and covers underflow and
flush-to-zero contributions. The query uses no host precision fallback.

Children can be visited in a different order, but a pruned candidate must already
be farther than the current FP64 winner. Equal distances retain the reference's
smallest-index rule. This is the reference nearest-site decision on the stored
coordinates, not a replacement for exact Delaunay/topology certification.
FP64 atomic accumulation order can still vary, so identical optimization
trajectories are measured controls, not a universal bitwise guarantee.

## Measured complete solves

Independent checks
find bitwise-identical sites, roles, full/material connectivity and oriented
boundary faces on all five native/trusted-input controls below, as well as the
default reference-backend torus pair. All pass exact full-hull checks, material
incidence, boundary vertex links, exact self-intersections and VTU round trips.

On NVIDIA L40 / Warp 1.19.0, median five additional fresh captured solves:

| Case | FP64 queries | Mixed queries |
|---|---:|---:|
| Uniform Armadillo 8K | 2585.30 ms | 2320.63 ms |
| Graded Armadillo 8K | 2756.59 ms | 2515.34 ms |
| Torus 1K | 484.61 ms | 467.83 ms |
| Counterbored plate 1K | 744.66 ms | 669.78 ms |
| Cavity 1K | 338.52 ms | 323.62 ms |

Timing includes refit, sampling, growth, fitting, quality stages and extraction;
setup/transfers/export/independent validation are separate. These are paired
GPU configurations, not CPU comparisons. Native surface-query limitations remain
as described in [its report](native-surface-queries.md).

The cavity retains minimum radius ratio 0.236810 and the reference connectivity.
Pure FP32 assignment had produced a different valid mesh with minimum 0.094167.
The mixed near-tie unit test resolves the same owner as FP64. Query tests also
cover coordinate scales from 2^-120 to 2^120, duplicate ties, changed active
prefixes, and 98,304 lower-bound cases spanning cancellation, nearby floats,
underflow and overflow. The five scale tests compare owners to independent NumPy
nearest-site distances, not only to the existing device implementation.

The private pilot is retained at `experiments/mixed-boundary-query` (runtime
prototype commit `e6f25be`). Its four sampled 8K boundary passes take about
0.97 ms versus 1.74–1.82 ms for FP64, with identical owners and target coordinates.
The first pilot stopped at a test-kernel compilation error; the corrected run
and the failed attempt are both retained. Pilot measurements are separate from
the production table above.

## Qualification and limits

The full ordinary suite passes 1,434 tests. All nine selected mixed-query cases
pass isolated padded and pool-disabled allocation-lifetime memory checks
(18 runs, zero errors), including two fresh complete captured mesher replays
with the reference surface backend. The separate native surface backend's
conditional/refit limitations are unchanged.

The paired public power/regular-diagram matrix passes its 2% regression gate.
Standalone Delaunay changes +0.29% at 1K and +0.22% at 100K; the default
reference torus changes 488.56 to 488.64 ms (+0.02%). All complete timing windows
are CPU-quota clean. Existing device definitions in `_boundary.py` are unchanged;
all other existing runtime files are byte-identical. The new module is loaded
only for the opt-in mode, and adds no per-workspace query arrays.

Source pins, commands, full sample timings, independent numerical reports and
raw-file hashes are in
[experiments/mixed-boundary-queries](../experiments/mixed-boundary-queries/).
Raw meshes stay outside ordinary Git. This is query-precision qualification,
not a solution for FP32 accumulation (#274), general default-input recovery
(#263), or a quality-matched CPU speedup (#40).
