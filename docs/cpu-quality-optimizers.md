# CGAL quality-optimizer controls

The native baseline now supports perturbation and exudation, after optional ODT.
All remain separately timed and explicitly enabled. The production implementation
is the pure-Warp library; these host programs are independent benchmark tools.
See the [CLI/build instructions](../benchmarks/cpu/README.md).

This study does **not** establish the requested wide GPU speedup. The current
uniform 8K mixed-query GPU solve takes 2.321 seconds. The valid roughly 6K-site
CPU controls below are faster but have different element-quality and boundary-fit
tradeoffs. The historical near-8K optimized CPU controls fail material validation;
those failures remain in the evidence, not in successful timing medians.

## Current measurements

Input is the same normalized FP32 Armadillo surface used by the GPU: 43,243
vertices and 86,482 triangles, without repair/simplification. CGAL 6.2.1 uses
GNU C++ 11.4, `-O3 -DNDEBUG -std=c++17`, sequentially or TBB with four threads.
The CPU is a Xeon Platinum 8362 under a four-core quota; the GPU is an L40.
Size 0.045, facet distance 0.01, feature angle 120, manifold refinement, seed 2026,
facet angle 25 and cell radius-edge bound 2 are fixed. Matching feature angles
does not establish identical CPU/GPU feature constraints.

Each row has three fresh native runs. Refine skips all optimizers. ODT has a
32-iteration cap, convergence 0.02 and freeze bound 0.01. Quality skips ODT and
runs perturbation then exudation, each with a five-second limit and ten-degree
target. These successful quality runs report bound reached; ODT reports convergence.
Iteration counts are limits, not executed counts. Quality controls supplement
rather than replace the ODT comparison.

| CPU configuration | Valid / clean samples | Referenced sites | Complete seconds | Prepared-domain seconds |
|---|---|---:|---:|---:|
| Sequential refine | 3/3 valid, 3 clean | 6,006 | 0.486 | 0.351 |
| Sequential ODT | 3/3 valid, 3 clean | 6,006 | 2.127 | 1.992 |
| Sequential quality | 3/3 valid, 3 clean | 6,006 | 0.627 | 0.491 |
| Four-thread refine | 3/3 valid, 3 clean | 6,220–6,263 | 0.290 | 0.157 |
| Four-thread ODT | 2/3 valid, 1 clean | 6,191–6,267 | 1.393* | 1.261* |
| Four-thread quality | 3/3 valid, 3 clean | 6,249–6,279 | 0.424 | 0.269 |

Times are medians of valid unthrottled samples. `*` is a single qualifying sample,
not a clean repeated estimate: one output fails material validation and one valid
run is CPU-quota throttled. The entire four-thread ODT batch is marked invalid.
Every native execution and rejected outcome is retained.

Complete CPU time includes domain preparation, meshing and optimization;
prepared-domain time excludes preparation. Input, native triangulation checking,
output and independent validation are separate. GPU resident replay includes
refit, fresh sampling, growth, fitting, quality work and extraction; setup,
transfers/export and independent checking are excluded. CPU and GPU timing
windows do not overlap. The CPU comparison is not made slower by including I/O.

| Configuration (valid outputs) | Minimum radius ratio | Mean radius ratio | Boundary RMS / bbox diagonal | Signed volume error |
|---|---:|---:|---:|---:|
| Warp uniform 8K, mixed queries | 0.03102 | 0.87457 | 0.14799% | −2.012% |
| Sequential refine | 0.00858 | 0.76838 | 0.10524% | −1.520% |
| Sequential ODT | 0.00858 | 0.85296 | 0.12017% | −2.102% |
| Sequential quality | 0.14698 | 0.77597 | 0.10571% | −1.530% |
| Four-thread refine | 0.00690–0.01474 | 0.76681–0.77177 | 0.10290–0.10312% | −1.514% to −1.451% |
| Four-thread ODT | 0.01595–0.05597 | 0.85558–0.85705 | 0.11941–0.12097% | −2.057% to −2.055% |
| Four-thread quality | 0.14682–0.16627 | 0.77366–0.77674 | 0.10266–0.10391% | −1.506% to −1.467% |

The CPU quality controls improve worst-element quality considerably but retain
lower mean quality than Warp. They use fewer vertices and fit the boundary more
closely here. These differences preclude a single quality-matched speedup ratio.
RMS is area-weighted, sampled in both directions with three quadrature points per
face; it is not a Hausdorff certificate. Raw reports retain full quality percentiles.

## Geometry, compatibility and retained failures

All 17 initially valid current outputs pass independent positive-volume/material
incidence checks, single-cycle boundary vertex links and exact boundary
self-intersection checks. Each has one component and boundary Euler characteristic
2. The eighteenth output is retained as rejected. Native regular-triangulation
validity alone is not sufficient for this material review.

New binaries export every material-referenced vertex weight alongside coordinates
and cells; the runner validates and hashes the weights. Feature protection can
already give nonzero weights before exudation. Exudation changes weights and runs
last, as required by the [CGAL reference](https://doc.cgal.org/latest/Mesh_3/group__PkgMesh3Functions.html).
No CPU output is mislabeled an ordinary unweighted full-hull Delaunay mesh.

Seven fresh native functional outputs, four runner cases and five invalid native
controls pass. Disabled sequential geometry matches the historical binary exactly;
explicit disabled flags match too. The cube perturbation and exudation controls
change geometry/weights and pass independent topology/self-intersection checks.
Four automated Python tests check options, requested-vs-reported optimizer policy,
legacy compatibility and preservation of partial output on timeout. Original
binaries are unchanged. No Warpower runtime source or public GPU routine changes.

The earlier private pilot used the same optimizer ordering with explicit source
pins. Its coarse sequential/four-thread quality medians were 0.616/0.427 seconds,
with all six coarse outputs valid and unthrottled. All six finer size-0.04,
10-ODT outputs failed material validation. That pilot is separate from the current
production measurements and is available in the
[verified historical evidence release](https://github.com/alecjacobson/warpower/releases/tag/evidence-selective-step-history-b343f70).

Compact build dependencies, commands, all current/historical numerical reports,
independent reviews and raw-file hashes are in
[experiments/cpu-quality-optimizers](../experiments/cpu-quality-optimizers/).
Native binaries and raw meshes remain outside ordinary Git. The wide CPU
outperformance goal (#40/#29) remains open.
