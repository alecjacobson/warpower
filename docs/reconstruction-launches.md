# VTM reconstruction and projection launches

VTM-owned facet completion, seed filtering, boundary projection and refinement
candidate evaluation now use 32-thread launch blocks. The five affected kernels
are `search_faces`, `extend_faces`, `check_seeds`, `constrain` and `candidates`.
Their geometric arithmetic, exact predicates, query limits, iteration budgets and
device array allocations are unchanged.

The private VTM reconstruction class selects its launch settings before its base
constructor captures any graph, so initial construction, previous-cell recovery
and fallback use the same policy. The independent `DelaunayWorkspace` keeps its
256-thread facet launches. Public constructor signatures are unchanged. Module
variants are loaded during setup, before application graph capture.

This follows the [surface-query tuning](query-launches.md). On the NVIDIA L40,
the five kernels accounted for 31.1% of the preceding complete torus profile's
summed kernel time. Most used 255 registers per thread. Smaller blocks can spread
short active prefixes across more multiprocessors; the selected size is measured
for this GPU and Warp 1.19, not a universal optimum.

## Complete-solve launch sweep

The prototype changed all five launch settings together. Each result is the
median of three warm, complete fresh resident solves, including domain refit,
sampling, growth, fitting, four quality rounds and final extraction. Setup,
compilation, uploads, export and independent checks are outside replay timing.
The torus has 48 by 24 surface samples, seed 2026, 256 initial sites and
quadrature 4; the other mesher benchmark budgets retain their defaults.

| Threads per block | 1,000 sites (ms) | 4,096 sites (ms) |
| ---: | ---: | ---: |
| 256, preceding default | 1,520.013 | 7,994.083 |
| 128 | 1,410.729 | 7,911.882 |
| 64 | 1,416.163 | 7,905.250 |
| 32, selected | 1,358.073 | 7,857.445 |
| 16 | 1,498.483 | 7,983.615 |

The 32-thread combined variant was fastest at both sizes. Additional individual
kernel probes and their raw samples are retained in the comparison record.

All ten combined sweeps have bitwise-identical positions and matching full and
material tetrahedron sets at each site count. Every run passes independent
Delaunay/topology checks with zero observed CPU quota throttling. Additional
individual 16-thread probes differed only slightly from individual 32-thread
results; the combined 32-thread setting was selected without adding a mixed
configuration. The production results match the selected prototype's geometry.

The [comparison record](../results/validation/vtm/recon-launch-summary.json) and
[artifact manifest](../results/validation/vtm/recon-launch-provenance.json) retain
all raw measurements, independent checks and hashes. The archived sweep wrapper
changes only the selected launches. To reproduce its original baseline context:

```bash
git worktree add --detach ../warpower-before-launches 7a50f58
WARPOWER_RECON_BLOCK_DIM=32 WARPOWER_RECON_BLOCK_TARGET=all \
  PYTHONPATH=../warpower-before-launches python \
  results/validation/vtm/recon-launch-sweep-probe.py \
  --case torus --n 1000 --repeats 3 --reference --output recon-block32.json
```

Repeat with block sizes 16, 64, 128 and 256, and with `--n 4096`. Individual
probe targets are `extend`, `search`, `seeds`, `constrain` and `candidates`.
Keep their imported checkout at the baseline so the other launches remain 256.
Run GPU experiments separately, without concurrent CPU work during timing.

## Paired production measurements

These measurements compare baseline `7a50f58` with runtime `53ced73`. Each row
uses three warm complete fresh resident solves, with the same timing scope as
the sweep. Every first/warm replay window reports zero CPU quota throttling.

| Complete resident solve | Before (ms) | After (ms) | Speedup |
| --- | ---: | ---: | ---: |
| Torus, 1,000 sites | 1,521.119 | 1,358.290 | 1.120× |
| Torus, 4,096 sites | 8,001.466 | 7,857.397 | 1.018× |
| Graded torus, 1,000 sites | 2,530.202 | 2,297.076 | 1.101× |
| Thin hand, 1,000 sites, quadrature 20/20 | 3,192.729 | 3,050.018 | 1.047× |
| Cavity, 1,000 sites, quadrature 20/20 | 1,384.773 | 1,277.181 | 1.084× |
| Disconnected, 1,000 sites, quadrature 20/20 | 1,214.109 | 1,119.754 | 1.084× |

All six preserve full and material tetrahedron sets, oriented boundaries, vertex
roles, optimizer steps and quality-move counts. Five have bitwise-identical
positions; the hand's maximum coordinate difference is `9.22e-19`. Floating-point
aggregate reductions can differ in their last bits when scheduling changes.
No iteration budget or quality work was removed.

The two coarse quadrature-4 recovery cases also reproduce the preceding repaired
meshes, taking 2.858 s for the cavity and 2.951 s for the disconnected components.
All eight saved full hulls pass independent CGAL 6.2.1 exact orientation, global
empty-sphere, face-pairing and hull-support checks. Those independent sphere
checks allow cospherical alternatives. NumPy checks independently verify material
quality and topology; all six gallery VTU round trips reproduce the saved arrays.

The workspace pool deltas remain 272.603 MB at capacity 1,000 and 1,705.410 MB
at capacity 4,096. Warm-cache setup is 1.354 → 1.331 s and 2.575 → 2.618 s,
respectively. These are allocator deltas, not total process-memory peaks. Cold
compilation can take additional time for block-size variants. Source AST checks
confirm unchanged module-level kernel/function bodies and device array allocation
calls in all six edited runtime modules.

## Independent routine gates

The ordinary diagram and Delaunay APIs remain independently callable. Standalone
facet recovery retains its preceding launch configuration, including conditional
branches that an ordinary random input might not execute. Before/after captured
graphs confirm 256-thread `check_seeds`, `search_faces` and `extend_faces` blocks,
with identical grids and shared-memory sizes.

| Standalone operation | Before (ms) | After (ms) |
| --- | ---: | ---: |
| Delaunay, 1,000 random sites | 9.174 | 9.187 |
| Delaunay, 100,000 random sites | 181.327 | 182.511 |
| Delaunay, 8³ structured grid | 3.956 | 3.957 |
| Delaunay, 256 fitted-hand sites | 13.414 | 13.437 |
| Diagram, 100,000 uniform sites | 18.251 | 18.058 |

Delaunay uses nine warm repeats per row; the diagram uses fifteen. All complete
successfully without observed quota throttling. The largest Delaunay shift is
+0.65%; these small differences are treated as timing variation. No standalone
speedup is attributed to this VTM launch change.

## Native profile and validation

The [before](../results/validation/vtm/recon-launch-before-kernels.json) and
[after](../results/validation/vtm/recon-launch-kernels.json) native profiles each
contain three complete fresh torus1000 solves. Profiling overhead applies;
the clean measurements above are separate.

| Kernel | Calls | Before (ms) | After (ms) |
| --- | ---: | ---: | ---: |
| Facet extension | 531 | 454.901 | 190.411 |
| Facet candidate search | 531 | 454.503 | 388.311 |
| Seed filtering | 420 | 288.687 | 275.102 |
| Boundary projection | 153 | 156.833 | 58.126 |
| Refinement candidates | 12 | 63.470 | 19.413 |

Total kernel activity remains 95,613; summed time changes from 4,555.582 to
4,069.024 ms. The actual five targeted kernels use 32-thread blocks. The separate
quality-move candidate kernel retains 256 threads.

All 529 tests pass. The initial suite stopped at a fault-injection test wrapper
that did not forward the new launch keyword. Fixing the wrapper preserved its
rollback assertions; runtime code remained unchanged before the passing rerun.
Ten focused cases pass unfiltered padded memory checking with zero errors,
including fragment recovery, captured producer/refinement/consumer composition
and independent uniform/graded insertion oracles. A separate six-phase resident
audit passes allocation/lifetime checking with memory pooling disabled, covering
fresh, nearby, moved, duplicate, nonfinite and restored inputs.

The [complete native capture audit](../results/validation/vtm/recon-launch-device-audit.json)
records three graph launches and one final synchronization. The graph contains
10,252 kernel nodes, 3,473 memsets, 1,234 conditionals and 919 device-to-device
copies, with no host transfers, callbacks, allocations or frees. Nsight observes
only three copy activities here; every graph copy node is also checked because
trace activity can omit graph copies.

This comparison does not establish CPU outperformance. The full VTM speedup and
feature-validation goal remains open. [Issue #136](https://github.com/alecjacobson/warpower/issues/136)
tracks a separate opportunity to reuse already-completed exact cell checks within
one unchanged-input facet completion.
