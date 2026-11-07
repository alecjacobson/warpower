# Surface-query launch configuration

Extraction classification, retained-cell classification and boundary-distance
measurement now use explicit 32-thread launch blocks. These are independent
per-cell or per-face kernels. The change does not alter their geometric
arithmetic, predicates, query limits, output buffers or error handling. The
selection and quality-proposal callers use the same setting. Other launches
retain their preceding 256-thread default.

The complete 1,000-site torus profile at `25ab98b` spent 49.7% of summed kernel
time in these three kernels. Both classification kernels used 255 registers per
thread and launched just 32 blocks for the 8,000-cell reserve. Smaller blocks
provide more independently scheduled work, particularly when only a small prefix
contains active cells or boundary faces. The configuration is measured on an
NVIDIA L40 with Warp 1.19; it is not an autotuner or a claim of optimal launch
settings on every GPU.

## Complete-solve launch sweep

Each entry is the median of three warm, complete fresh resident solves. Domain
refit, initial sampling, growth, fitting, four quality rounds and final extraction
are included. Setup, compilation, transfers, export and independent checking are
outside replay timing. The input is the 48 by 24 torus, seed 2026, 256 initial
sites, quadrature 4, and the mesher benchmark's default iteration budgets.

| Threads per block | 1,000 sites (ms) | 4,096 sites (ms) |
| ---: | ---: | ---: |
| 256, preceding default | 2,229.046 | 8,629.334 |
| 128 | 1,708.208 | 8,507.992 |
| 64 | 1,598.320 | 8,090.576 |
| 32, selected | 1,519.505 | 7,997.288 |

All eight runs pass independent Delaunay and topology checks and report zero
observed CPU quota throttling. At each site count, all four settings produce
bitwise-identical positions and identical full and material tetrahedron sets.
Floating-point aggregate reductions can differ in their last bits when scheduling
changes. The sweep changes only these three kernels through a benchmark-only
launch wrapper; the production library directly specifies the selected setting.

The [comparison record](../results/validation/vtm/query-launch-summary.json) and
[artifact manifest](../results/validation/vtm/query-launch-provenance.json) retain
raw measurements, independent checks and source hashes. To reproduce the sweep
from a repository checkout with benchmark dependencies installed:

```bash
WARPOWER_QUERY_BLOCK_DIM=32 PYTHONPATH=. python \
  results/validation/vtm/query-launch-sweep-probe.py \
  --case torus --n 1000 --repeats 3 --reference --output query-block32.json
```

Repeat with 64, 128 and 256, and with `--n 4096`. Run GPU experiments separately
and avoid concurrent CPU work during timing. The raw records include setup time
and CPU quota observations separately from warm replay measurements.

## Paired production measurements

The following fresh comparison uses baseline `25ab98b` and runtime `9c5dac3`,
with three warm replays per case and the same timing scope as the sweep. All
recorded replay windows have zero observed CPU quota throttling.

| Complete resident solve | Before (ms) | After (ms) | Speedup |
| --- | ---: | ---: | ---: |
| Torus, 1,000 sites | 2,225.734 | 1,519.625 | 1.465× |
| Torus, 4,096 sites | 8,620.535 | 7,998.445 | 1.078× |
| Graded torus, 1,000 sites | 3,720.355 | 2,526.320 | 1.473× |
| Thin hand, 1,000 sites, quadrature 20/20 | 3,366.712 | 3,190.879 | 1.055× |
| Cavity, 1,000 sites, quadrature 20/20 | 1,420.119 | 1,383.512 | 1.026× |
| Disconnected, 1,000 sites, quadrature 20/20 | 1,241.226 | 1,214.841 | 1.022× |

All six cases preserve full and material tetrahedron sets, oriented boundaries
and vertex roles. Five have bitwise-identical positions; the hand's largest
coordinate difference is `1.51e-18`. The two coarse quadrature-4 recovery cases
also reproduce their preceding repaired meshes exactly, taking 2.973 s for the
cavity and 3.054 s for the disconnected components. All eight saved full hulls
pass independent CGAL 6.2.1 exact orientation, global empty-sphere, face-pairing
and hull-support checks. Independent NumPy checks verify material quality and
topology, and every gallery VTU round trip reproduces the saved arrays.

The measured workspace pool delta is unchanged: 272.603 MB at capacity 1,000
and 1,705.410 MB at capacity 4,096. Warm-cache setup changes from 1.343 to
1.280 s and from 2.510 to 2.536 s respectively. These allocator deltas are not
total process-memory peaks; cold compilation can add time for block-size variants.

Standalone Delaunay at 1,000 sites measures 9.216 → 9.174 ms. The 100,000-site
diagram measures 18.131 → 18.183 ms. An initial five-repeat 100,000-site Delaunay
batch measures 182.118 → 186.248 ms, a 2.27% increase despite identical source
paths. That shift prompted two alternating before/after rounds of nine repeats:
182.517 → 182.110 ms and 181.732 → 181.455 ms. The pooled recheck medians are
182.252 → 181.711 ms. The initial slowdown did not repeat. All original and
recheck samples are retained, with no quota throttling; small standalone timing
differences are treated as variation, not benefits attributed to this VTM change.

## Native profile and validation

The [before profile](../results/validation/vtm/query-launch-before-kernels.json)
and [after profile](../results/validation/vtm/query-launch-kernels.json) each
contain three complete fresh 1,000-site torus solves. Profiling overhead applies;
the clean performance samples are separate.

| Kernel | Calls | Before (ms) | After (ms) |
| --- | ---: | ---: | ---: |
| Retained-cell classification | 297 | 1,487.109 | 515.599 |
| Extraction classification | 81 | 1,184.279 | 395.680 |
| Boundary-distance samples | 81 | 649.086 | 276.317 |

The total kernel activity count remains 95,613. Summed kernel time changes from
6,686.787 to 4,555.582 ms. All three query kernels have observed 32-thread blocks;
the two classifiers launch 250 blocks instead of 32 for the same reserved cell
count. No optimizer iterations or quality work were removed to obtain the gain.

All 529 tests pass. Eight focused cases pass unfiltered padded memory checking
with zero errors, including the captured selective-quality loop, a complete
small repaired graded trajectory, strict extraction thresholds and a nearly flat
positive tetrahedron. The separate repaired/nonfinite/empty/restored resident
audit passes allocation/lifetime checking with zero errors.

The full-query reference in `benchmarks.selective` also uses 32-thread blocks,
so that its comparison with retained-only queries isolates the query algorithm.
Both 128-site, two-iteration verification runs pass independent Delaunay/topology
checks and produce matching element-quality percentiles.

The [complete native capture audit](../results/validation/vtm/query-launch-device-audit.json)
records three graph launches and one final synchronization. Its graph contains
10,252 kernel nodes, 3,473 memset nodes, 1,234 conditionals and 919 copies, all
device-to-device. There are no host transfers, callbacks, allocation or free
nodes. Nsight reports only three copy activities in this trace; the audit also
checks every copy node in the graph because trace activity can omit graph copies.

The standalone diagram and Delaunay source paths are unchanged. No CPU
outperformance is established by this comparison with the preceding Warpower
implementation; the full VTM performance goal remains open.
