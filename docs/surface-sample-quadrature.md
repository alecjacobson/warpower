# Parallel boundary surface-error quadrature

One query per thread reduces complete graded 4K Armadillo replay time by **10.43%**.
Uniform and graded 8K replays improve by **7.19% and 5.29%** in the final paired
measurements. The independent power-diagram and Delaunay routines retain
their implementations and callable APIs. This is an improvement between Warp
revisions; the comparable-quality native CPU outperformance goal remains open.

## Implementation

The extractor previously ran three nearest-surface queries sequentially in each
boundary-face thread. The current kernel maps thread `j` to face `j // 3` and
quadrature sample `j % 3`. One launch covers three times the reserved face capacity;
the current device count and error flag guard every access to inactive faces.
The three-point quadrature, area weights, double-precision closest-point
traversal, error bits, integral addition and maximum reduction are unchanged.

Only this kernel and its launch dimension change in production. The selection
extractor's status-only path is unchanged. There are no new persistent arrays,
host count reads, replay allocations, or extra launches. The arithmetic AST
matches the private measured kernel, while the test reference matches the old
sequential loop. The source audit binds 266 files, with 263 unchanged and changes
confined to `_extraction.py` and two test files.

Atomic addition order changes, so the integral is not promised bitwise identical.
The initial instrumented comparisons preserve each nearest hit's point, face,
squared distance and weighted contribution exactly across the tested caller
contraction settings and scales. Static integral differences are a few parts in
10^15; maximum distances are bitwise equal. Full trajectories and independent
mesh quality receive separate checks below.

## Complete solves

NVIDIA L40, Warp 1.19.0+cu12; baseline `c4dcae4bb764ad6640a81308ffdace1c3f27fd8f` and runtime
`e6033866f87775b3959d8780e019fd1bf5bf9639`. Each warmed replay starts a fresh complete solve, including
refit, sampling, growth, fitting, quality work, recovery and extraction. Setup,
upload and offline validation are outside replay; setup and first-call results
remain in the raw records. Mesher-only constructor observations and benchmark
setup observations have distinct scopes, identified in the comparison report.
All 32 final complete/standalone jobs have clean first and timed quota windows.

| Complete example | Before, ms | After, ms | Time reduction |
|---|---:|---:|---:|
| torus16384-fixed32 | 4417.123 | 4409.845 | 0.16% |
| torus1000 | 512.930 | 494.766 | 3.54% |
| torus8192-fixed32 | 1372.304 | 1362.608 | 0.71% |
| torus8192-progressive | 2658.037 | 2648.703 | 0.35% |
| thin-hand-seed7 | 1244.078 | 1221.403 | 1.82% |
| cavity | 358.231 | 350.809 | 2.07% |
| disconnected | 485.109 | 478.246 | 1.41% |
| plate | 921.456 | 877.503 | 4.77% |
| armadillo-graded4096 | 2578.452 | 2309.608 | 10.43% |
| armadillo-graded8192-recovery | 4075.479 | 3859.995 | 5.29% |
| armadillo-uniform8192-recovery | 3957.132 | 3672.809 | 7.19% |

The larger torus cases change by less than 1%. Input arrays, site roles, material connectivity, oriented
boundaries and recorded solver work match for all pairs. The independently
measured minimum radius ratios agree within 1e-12.

One scalar coordinate differs in each hand and cavity pair: by at most
`4.404571739312668e-18` and `2.3283064365386963e-10`, respectively. Both pairs retain
identical oriented full-hull connectivity; both current geometries validate.

All 22 final meshes have valid independent full-hull empty-sphere/orientation/
coverage checks, positive material geometry, closed edge topology, single-cycle
boundary vertex links and zero checked boundary self-intersections. Of these,
19 reuse a previous independent proof only after exact comparison of every array,
oriented connectivity and aligned stored quality.
Other meshes receive fresh native and NumPy checks. All fourteen current gallery
VTUs and boundary OFF exports round-trip exactly. Surface-distance measurements
use quadrature and are not Hausdorff certificates. The scanned domain keeps its
[original provenance](scanned-armadillo.md); it is not claimed as an original
Alliez-paper input or relicensed under MIT.

## Independently callable routines

| Routine/input | Before, ms | After, ms | Change |
|---|---:|---:|---:|
| delaunay1000 | 9.179 | 9.176 | -0.04% |
| delaunay100000 | 182.066 | 181.829 | -0.13% |
| delaunay-grid | 2.205 | 2.213 | +0.35% |
| delaunay-fitted-hand | 13.393 | 13.400 | +0.05% |
| diagram100000 | 18.238 | 18.226 | -0.06% |

All recorded inputs, counts, errors and diagnostic fields match. These small
differences do not establish a standalone speed gain. The README retains
installation instructions and the earned Paragram flagship.

## Static experiment and qualification

The static batches use independently validated final meshes and the full reserved
boundary capacity (32 times the site budget). Resets and padded launch work are
included; setup and host checks are excluded. Each row has twenty warmed GPU
samples. The 32-thread candidate was selected before the complete pilot.

| Final mesh | Original, ms | Samples/32, ms | Samples/64, ms | Samples/128, ms |
|---|---:|---:|---:|---:|
| torus1000 | 1.011 | 0.368 | 0.618 | 1.169 |
| torus16384 | 0.948 | 0.596 | 0.575 | 1.025 |
| armadillo-uniform8192 | 11.252 | 6.000 | 6.114 | 10.185 |
| armadillo-graded8192 | 12.067 | 6.465 | 7.115 | 11.869 |

The private compiled kernel reports 78 registers/thread versus 96 for the original,
with the same 3,296 local bytes/thread. Those function properties do not measure
occupancy or explain a full-solve gain by themselves. The four pilot pairs retain
all coordinates, oriented cells, roles and work counts, with zero additional
workspace pool bytes.

The production suite passes all 1,387 tests. The 41 new cases cover both caller
contraction settings, six shapes at three scales, poisoned inactive capacity,
empty/failed queries, live counts and capture on another stream. Tests restore
module options after each case and synchronize setup before changing streams.
The unchanged full 8K five-phase loop covers bounded recovery exhaustion, success,
invalid target, collapsed domain and recovery, with stable buffer addresses and
public failure counts cleared.

Native captures observe 93,800 full-solve and 158,566 loop kernel activities. The retained
CUDA activity and graph-DOT audits contain only device-to-device copies and no
host callbacks or replay allocation/free nodes. Each run has one outer graph
launch and a final synchronization; first loop instantiation/upload, where
present, is reported in the raw API list. These profiled times are diagnostic,
not substitutes for clean timing samples.

Four isolated sanitizer jobs pass with zero errors: quadrature contracts and the
full recovery loop, each with padding and with pool-disabled allocation-lifetime
checking. The loop jobs took 302.7 s and 514.6 s.
The existing diagnostic stack allowance is recorded by the runner; production
benchmarks do not use that override. Ordinary, native and sanitizer loop frames
match independently certified reference geometry and consumed diagnostics.

The record retains an interrupted pilot driver with an export-loop naming bug,
two offline checker setup errors, and an interrupted first suite run before
test-option cleanup was corrected. None is counted as passing or as a numerical
failure. Original scripts, logs, partial outputs and corrections remain intact.

The [manifest](../results/validation/vtm/surface-sample-quadrature/manifest.json)
binds raw timings, complete meshes, source versions, failures, native traces,
memory logs and independent proof dependencies. This completes
[#252](https://github.com/alecjacobson/warpower/issues/252).
[#253](https://github.com/alecjacobson/warpower/issues/253) tracks a separate
nearest-traversal hypothesis; it is not part of this implementation.
