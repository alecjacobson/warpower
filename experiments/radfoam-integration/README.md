# RadFoam fitting integration

This experimental adapter runs real Mip-NeRF 360 Bonsai scene fitting with
WarPower Voronoi adjacency and the upstream RadFoam renderer and backward pass.
It also compares native incremental triangulation with current VTM fragment
reuse. Production WarPower and VTM defaults are unchanged.

With `--recipe radfoam`, the harness now calls the upstream initializer and uses
all training views for densification. On the same NVIDIA L40, paired 20,000-step
fits with identical initial parameters produced:

| Geometry backend | Fit seconds | Pooled-RGB PSNR | Upstream PSNR | Final sites |
| --- | ---: | ---: | ---: | ---: |
| Native incremental RadFoam | 311.18 | 29.1606 | 29.1929 | 251,525 |
| WarPower, Morton order | 211.73 | 29.2091 | 29.2365 | 251,306 |

This is an observed 1.47x fitting-loop speedup (32.0% less time) with similar
quality. Geometry time fell from 113.91 to 28.62 seconds. Both runs used 65,536
rays per step, a 262,144-site target, quarter-resolution images, the same four
training evaluations, numerical audits, and checkpoint restoration. Loading,
initialization, warmup and final serialization are outside the loop timer.
These are single runs with different optimization trajectories and final counts.

The official-loop control at this reduced workload scored 29.2363 pooled-RGB dB;
the port recovers its quality closely. This timing comparison is between the
two backends in our harness, not against the official loop's different loading
and evaluation schedule. Full-budget results from the official loop are below.

Both fits had zero nonfinite raw outputs/gradients and zero CPU quota throttling,
and both restored identical rendering and Adam moments. Warp recorded one ray
at the renderer's 1,024-cell traversal cap across its audited run; native recorded
none. These counters cover warmup, training, error maps, evaluation and restore
checks, without recording the phase of the hit.
The saved models have no cap hits in any of the 37 held-out views (maximum
traversal lengths 188 for Warp and 195 for native). The earlier hit remains
unlocalized and unresolved; successful final evaluation does not clear it.

[Ported-recipe results and validation](../../results/validation/radfoam/ported-recipe/summary.json)
include both PSNR definitions, stage timings, numerical counters, exact executed
sources and a checksum inventory of local raw outputs.

## Full-budget comparison

`run_full_comparison.py` runs the pinned official training loop with native
geometry, then with WarPower Morton-ordered geometry, sequentially on the same
GPU. Its goal is to match PSNR while reducing training time. Both use the official
20,000 steps, 1,000,000 rays per step, 2,097,152-site target, seed 42, all-view
densification maps, and quarter-to-half-resolution schedule at step 5,000.
Evaluation uses all 37 half-resolution held-out views every 100 steps.

The native initializer is shared, including its bootstrap triangulation. Warp
replaces subsequent adjacency updates and permutes parameters and Adam moments
together. Both runs retain the original asynchronous batch fetching, renderer,
backward pass, loss, optimizer and densification implementation. The loop timer
includes intermediate evaluation and the resolution switch, but excludes data
loading, initial setup/compilation and final exports. Total wall time is recorded
separately. Original renderable PT checkpoints and final PNGs are kept; redundant
PLY exports outside the loop timer are skipped to fit persistent storage.

```bash
.venv/bin/python experiments/radfoam-integration/fetch_bonsai.py --downsample 2
.venv/bin/python experiments/radfoam-integration/run_full_comparison.py \
  --output results/local/my-full-bonsai-comparison
```

The runner first performs a two-step Warp preflight with full ray and evaluation
budgets, then runs both full fits and a shared evaluation. Use a detached process
for unattended execution. `plan.json` records immutable driver snapshots and
checksums; `status.json` records the current stage and latest PSNR. Each stage has
its own log and report. A failed stage stops the queue. `comparison.json` is only
written after both full fits and the common evaluation succeed.

The paper's Bonsai reference is 32.15 dB and roughly 70 minutes on RTX 4090.
Our primary timing comparison is native versus Warp on the same L40; comparison
with 70 minutes is cross-hardware context. The completed paired fits produced:

| Geometry backend | Full-loop minutes | Upstream PSNR | Pooled-RGB PSNR | Final sites |
| --- | ---: | ---: | ---: | ---: |
| Native RadFoam | 89.234 | 32.4136 | 32.3808 | 1,968,116 |
| WarPower, Morton order | 54.804 | 32.2511 | 32.1772 | 1,944,340 |

Warp reduced full-loop time by 38.58% (1.628x speedup), with 0.1625 dB lower final
PSNR. At the shared 32.25 dB target, the first recorded crossing was 70.86 minutes
for native and 54.80 minutes for Warp, a 1.293x time-to-quality improvement.
These crossings are sampled evaluations, without interpolation or saved
intermediate models. Both final checkpoints exceeded the paper's 32.15 dB.

The quality gap is concentrated in held-out views 36 and 10, accounting for 95.5%
of the net deficit; the other 35 views average a -0.0077 dB difference. The worst
saved Warp view has visible color bias and artifacts. One fit per backend cannot
estimate seed variance, so the deficit is not dismissed as noise. Both final
models have finite raw RGBA in all 37 views and no traversal-limit hits.
Both ran with a four-core quota; recorded CPU throttling was 28.61 seconds for
native and 16.04 seconds for Warp across their complete runs.

[Full results, curves and provenance](../../results/validation/radfoam/full-budget/analysis.json),
[shared per-view evaluation](../../results/validation/radfoam/full-budget/shared-evaluation.json),
and the executed source snapshots are preserved under `results/validation/radfoam/full-budget`.
Training retains upstream gradient masking;
the shared evaluator checks raw final RGBA values and traversal caps. It does
not provide the custom harness's per-call raw training-gradient audit.

## Earlier small-recipe fits

The custom 20,000-step reduced-budget fit on NVIDIA L40 took 184.15 seconds with
`warpower-morton`, versus 208.10 seconds with native RadFoam, an observed 1.13x
speedup. Mean PSNR over 37 held-out views was 26.83 versus 26.73 dB. These are
single full runs with slightly different final site counts; the component
benchmarks use repeated, alternating-order samples. Loading, initial JIT/warmup
and final checkpoint serialization are outside the fitting-loop timer.
These PSNR values use pooled RGB squared error per image. Upstream RadFoam
instead computes PSNR separately for each channel and then averages the three;
the two definitions must not be compared without reevaluating the checkpoints.

| Backend | Fit seconds | Held-out PSNR | Final sites |
| --- | ---: | ---: | ---: |
| Native RadFoam geometry, custom loop | 208.10 | 26.73 | 249,858 |
| WarPower, input order | 372.04 | 26.76 | 253,903 |
| WarPower, Morton order | 184.15 | 26.83 | 247,608 |

All three fits completed pruning/densification and a fresh-Scene checkpoint
restoration, with identical restored Adam moments and rendering. Raw renderer
outputs and gradients had no nonfinite values or traversal-limit hits. The
checkpoint check runs inside the fitting process; this is not a separate-process
resume CLI, and the independent minibatch generator remains alive during it.

## Official loop at the reduced workload

Two complete official-loop controls and a shared evaluation of all four saved
models give the following results on the same L40 and 37 held-out views:

| Training loop and recipe | Pooled-RGB PSNR | Upstream PSNR | Final sites |
| --- | ---: | ---: | ---: |
| Official loop, reduced resolution/rays/final sites, official initialization and error maps | 29.2363 | 29.2646 | 251,477 |
| Official loop, additionally matching our initialization and error-map settings | 26.8684 | 26.8934 | 250,084 |
| Custom loop, native geometry | 26.7285 | 26.7559 | 249,858 |
| Custom loop, WarPower Morton ordering | 26.8348 | 26.8623 | 247,608 |

Matching the smaller recipe brings the official loop within 0.140 dB of the
custom native fit and 0.034 dB of the custom WarPower fit. Matching only
resolution, rays and final sites leaves a 2.508 dB gap relative to our native fit.
The remaining initialization, seed and error-map choices matter collectively;
these two controls do not isolate their individual effects. Single runs with
different random-number sequences establish close quality agreement, not
identical trajectories or a statistically significant backend quality gain.

Both PSNR definitions are computed directly from the same rendered pixels.
Reevaluation reproduces the earlier per-view pooled-RGB values within 3.5e-6 dB;
the definition change adds only 0.025–0.028 dB to these models. All 148 evaluated
view/model combinations have finite raw RGBA outputs. The official training
controls retain upstream's gradient masking, without the custom harness's raw
gradient audit. The official-initialization control logged one recoverable
triangulation retry; the small-initialization control logged none.

[Compact results and provenance](../../results/validation/radfoam/official-controls/summary.json),
[common per-view evaluation](../../results/validation/radfoam/official-controls/shared-evaluation.json),
and both complete learning curves are retained under `results/validation/radfoam/official-controls`.
Models, images and logs remain local and checksum-inventoried. This is a
reduced-budget quality control, not a reproduction of the paper's full-budget
Bonsai result or a timing comparison against official training.

`official_reduced.py` imports the pinned upstream `train.py` directly, with its
native scene class, renderer, geometry, optimizer, initialization, asynchronous
batch fetchers, pruning/densification and evaluation logic. It overrides the
final-site target to 262,144, the image schedule to quarter resolution throughout,
and the hardcoded training batch to 65,536 rays. The 20,000-step count and all
learning-rate/loss settings already match our experiment.

This first control retains the official seed 42, 190,951 initial sites, all-view
densification error maps at stride 2, and evaluation every 100 steps. Our smaller
harness instead uses seed 2026, 9,216 initial sites, eight-view error maps at
stride 4, and four training evaluations. Thus this control tests whether the
smaller resolution/ray/site budget explains the quality gap; it does not isolate
every difference between the two harnesses or establish a speed comparison.
The optional `--recipe adapter-settings` control additionally matches the small
experiment's seed, initial positions and densities, and error-map view/stride
settings. It records a hash of the initial physical parameters, allowing a check
against the earlier runs. The upstream batch fetcher and random-number sequence,
evaluation cadence, and lack of a mid-training checkpoint exercise still differ;
this is a quality comparison, not an assertion of identical optimization paths.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
python experiments/radfoam-integration/official_reduced.py \
  --output results/local/my-official-reduced-fit

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
python experiments/radfoam-integration/official_reduced.py \
  --recipe adapter-settings --output results/local/my-official-adapter-settings-fit

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
python experiments/radfoam-integration/evaluate_shared.py \
  --checkpoint official results/local/my-official-reduced-fit/output/fit/model.pt \
  --checkpoint adapter results/local/my-bonsai-fit/model.pt \
  --output results/local/my-shared-evaluation.json
```

The common evaluator renders all 37 held-out quarter-resolution views and reports
both PSNR definitions from the same pixels. Each saved model retains its own
geometry. The official control saves its source fingerprints, exact executed
driver, configuration, TensorBoard curves, timing scope, CPU quota counters,
checkpoint, PLY, and official target/render/error images. Its loop timer includes
200 evaluations, so compare quality first; elapsed time is not directly
comparable to the existing reduced-harness timings.

## Upstream initialization and densification in our harness

Select `--recipe radfoam` in `fit_scene.py` to use the upstream initializer and
full-view densification passes in the custom fitting loop. This recipe defaults
to seed 42, 20,000 steps, 65,536 rays per step, a 262,144-site target, and evaluation
every 5,000 steps. Images remain at quarter resolution. `--recipe small` keeps the
earlier lightweight initialization and eight-view error maps, with its original
defaults; `--initial-sites` applies only to that recipe.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
python experiments/radfoam-integration/fit_scene.py \
  --recipe radfoam --backend native --output results/local/my-radfoam-native

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
python experiments/radfoam-integration/fit_scene.py \
  --recipe radfoam --backend warpower-morton --output results/local/my-radfoam-warpower
```

The initializer is called directly from the pinned upstream scene class. It
samples 90% of the SfM point count with replacement, adds positional noise and
5,000 background sites, and sets the raw densities and zero SH coefficients.
For Bonsai this yields 190,951 sites. Its point ordering and density assignment
are preserved exactly, including their association in the upstream code; the
adapter subsequently permutes parameters together. Both backends therefore have
a one-time native initialization bootstrap, recorded in setup time and excluded
from the fitting-loop timer. Geometry refreshes during fitting use the selected
backend. The initial parameter hash allows comparison between the two runs.

At each densification event, the inherited error-map routine processes all 255
training views at stride 2. The inherited pruning/densification routine and its
growth schedule are retained. Our resident GPU minibatch sampling, numerical
audits, stage timers and mid-training checkpoint check remain active. Reports
include the chosen recipe, error-map coverage, initialization cost, and both
pooled-RGB and upstream channel-averaged PSNR.

Checkpoint restoration now preserves the original initial-site count used by
the densification schedule. A focused small-recipe run tests growth after the
restore. The paired full-recipe measurements above finished densification
before their checkpoint, so this subsequent bookkeeping correction does not
change their growth schedule; their executed sources are preserved separately.

## Spatial ordering belongs to this adapter

WarPower already sorts its internal spatial hierarchy while preserving input
site IDs. RadFoam's AABB tree groups contiguous array entries, so its camera-cell
lookup needs spatially ordered scene arrays. `warpower-morton` reorders sites,
learned attributes and optimizer moments on full refreshes. Incremental refreshes
preserve the current order. The optional adapter backend keeps this downstream
requirement out of the core power-diagram API.

On an identical fitted graph, integer relabeling preserved every directed edge,
all starting cells and RGBA exactly. Position and attribute gradient differences
were below 5e-9. Forward time fell from 14.44 to 1.65 ms for 65,536 rays; this is
the gain over the unordered adapter, not over native RadFoam, which also sorts.

## Build and run

Run commands from the repository root in its CUDA Python environment. The tested
versions are Torch 2.7.1+cu126, Warp 1.19.0+cu12, Python 3.10, CUDA Toolkit 12.6,
NumPy 2.2.6 and SciPy 1.15.3. Install Eigen 3 and TBB development headers/libraries.
The additional tested Python packages are:

```bash
python -m pip install pycolmap==3.11.1 plyfile==1.0.3 configargparse==1.7 \
  einops==0.8.1 tensorboard==2.19.0

git clone https://github.com/theialab/radfoam.git .work/radfoam-source
git -C .work/radfoam-source checkout 3e7b52cf74e37ab2ab5e695f53570f515f537e3d
git clone https://github.com/max0x7ba/atomic_queue.git .work/atomic-queue
git -C .work/atomic-queue checkout b7f1befe89446752767243c5e5550779a9a5eed2

export RADFOAM_SOURCE_DIR="$PWD/.work/radfoam-source"
export ATOMIC_QUEUE_INCLUDE_DIR="$PWD/.work/atomic-queue/include"
export RADFOAM_BUILD_DIR="$PWD/.work/radfoam-fitting-20261110/native"
export CUDA_HOME=/usr/local/cuda-12.6
export TORCH_CUDA_ARCH_LIST=8.9
python experiments/radfoam-integration/build_headless.py
python experiments/radfoam-integration/fetch_bonsai.py

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
python experiments/radfoam-integration/fit_scene.py \
  --backend warpower-morton --output results/local/my-bonsai-fit \
  --steps 20000 --initial-sites 8192 --final-sites 262144 \
  --rays 65536 --eval-every 5000 --save-trajectory
```

Set the CUDA architecture for the actual GPU. The headless build removes only
viewer bindings and uses Torch's C++ ABI settings. It retains native full and
incremental triangulation, tracing, AABB queries and allocation. The source and
build environment variables also control `integration.py`; their historical
defaults point at the original experiment workspace.

The downloader fetches only Bonsai quarter-resolution images and COLMAP data
from the official Mip-NeRF 360 archive, checking ZIP CRCs and recording SHA-256
hashes. `--downsample 2` adds the official half-resolution images (1559 by 1039)
and reuses verified COLMAP files. It does not download the entire 12.5 GB archive. Outputs include the
renderable model, training state, JSON measurements and held-out target/render
images. Use a new output directory for each run. The defaults without the larger
budget flags give a shorter 2,000-step example. Available backends are `native`,
`warpower`, `warpower-hints`, `warpower-morton` and `paragram`; only the first,
second and fourth have the reported complete 20,000-step fits.

## Component comparisons

`compare_ordering.py` tests exact graph relabeling, starts, pixels and gradients,
and measures tree construction, lookup, tracing and full forward cost.
`compare_geometry.py` replays saved fitting coordinate pairs through fresh/native
incremental triangulation and fresh/hinted WarPower adjacency. The recorded
frames can skip intervening refreshes; each record gives the actual step gap.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
python experiments/radfoam-integration/compare_ordering.py \
  --checkpoint results/local/my-bonsai-fit/model.pt \
  --output results/local/my-ordering.json

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
python experiments/radfoam-integration/compare_geometry.py \
  --trajectory results/local/bonsai-input-order/trajectory.npz \
  --output results/local/my-geometry-replay.json
```

For geometry replay, first repeat the fitting command with `--backend warpower`
and `--output results/local/bonsai-input-order`. Use that input-order trajectory:
the current legacy
capture does not record permutations or persistent IDs across reorderings.
Do not interpret a Morton-ordered trajectory's row changes as physical motion.
Pruning/densification boundaries are excluded from these pair measurements.

`compare_updates.py` uses the retained 180-frame, 8,192-site Armadillo VTM
trajectory, with current fragment reuse including wall/fresh fallbacks, native
RadFoam incremental and native full reconstruction. It expects an offline CGAL
binary at `.work/vtm/cgal-delaunay-final`, built from
[`benchmarks/cpu/delaunay.cpp`](../../benchmarks/cpu/delaunay.cpp) using the
[CPU reference instructions](../../benchmarks/cpu/README.md).

| VTM transition | RadFoam incremental | VTM reuse |
| --- | ---: | ---: |
| 1, unchanged positions | 2.129 ms | 1.732 ms |
| 2 | 7.336 ms | 14.481 ms |
| 162 | 3.029 ms | 2.782 ms |
| 179 | 4.131 ms | 4.237 ms |

All tetrahedron sets matched independent CGAL 6.2.1 and SciPy constructions.
These four cases do not identify a general winner. VTM's complete exact/symbolic
certificate is a stronger runtime contract than native RadFoam's checks. The
broader isolated replay investigation is tracked in
[#297](https://github.com/alecjacobson/warpower/issues/297).

## Evidence and scope

[Compact fitting results](../../results/validation/radfoam/fitting/summary.json),
[per-run measurements](../../results/validation/radfoam/fitting/warpower-morton.json)
and [provenance](../../results/validation/radfoam/fitting/provenance.json) are
committed. Per-run JSON retains all geometry samples, evaluations and numerical
checks, while sampling the 20,000-entry loss history every 250 steps. Exact source
snapshots used for the measurements are retained alongside those reports; the
current entrypoints add environment overrides for external checkout/build paths.

The preceding four-scene comparison against Paragram and sequential/parallel
CGAL is in [checkpoint-comparison](../../results/validation/radfoam/checkpoint-comparison/summary.json).
Its drivers are `benchmarks/radfoam_compare.py`, `benchmarks/radfoam_audit.py`,
`benchmarks/cpu/radfoam_adjacency.cpp` and `benchmarks/cpu/voronoi_facet_audit.cpp`.
The earlier small differentiable-rendering probe and bridge are retained as
historical feasibility evidence.

Large checkpoints, raw graphs, render images and binaries remain in the local
paths named by the committed checksum inventories. This branch publishes compact
evidence and reproduction source; it does not publish those raw files as a
release bundle. One-off animation outputs and their driver are outside Git.

The earlier reduced-budget fitting examples use 65,536 rays per step,
approximately 250,000 final sites and SH3. The small recipe uses eight-view densification error
maps; the RadFoam recipe uses all training views. These are reduced-budget
integration examples; the full-budget comparison above uses one million rays and
approximately two million final sites. WarPower returns
bounded adjacency, while native triangulation returns a full-hull mesh and
adjacency. The earlier exact audit found seven incorrect undirected WarPower
pairs across four larger checkpoints; successful fitting does not resolve them.
Density here is optical density, not a power weight. CUDA default-stream use is
tested; arbitrary stream interleaving is not qualified.
