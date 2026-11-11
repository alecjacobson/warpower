# VoroTracing in Warp

This experiment implements the GPU geometry, nearest-site queries, textured
Voronoi ray tracing, replay backward pass, and FP16 inference of
[VoroTracing](https://arxiv.org/abs/2608.17682) in Warp. It uses the original
VoroTracing Python model and trainer, including RoMa initialization, losses,
regularization, learning-rate schedules and fixed site budget. Torch owns the
parameters, loss and optimizer; RoMa retains its own implementation. The Warp
backend does not import or build the RadFoam/VoroTracing CUDA extension.

The upstream reference is
[`zenseact/VoroTracing` at 75910484](https://github.com/zenseact/VoroTracing/tree/75910484d6d46af25e180044ca6b9091204fe6fa).
The implementation is experimental and does not change production WarPower
defaults. Optical density is separate from power-diagram weights; these scenes
use ordinary, unweighted Voronoi cells.

## Method and implementation

Each cell has an 8-by-8 RGB octahedral map sampled at the entry-surface direction
and another sampled at the viewing direction. Bilinear interpolation, seam
wrapping, sigmoid appearance and exponential optical density match the pinned
reference. The backward kernel replays traversal and scatters gradients without
storing a tape for every ray/cell intersection.

Reference training uses FP16-rounded neighbor differences for face selection,
FP32 plane derivatives and appearance, and a custom backward. That backward stops
derivatives through texture coordinates and applies distortion gradients only to density. We match
these deliberate implementation choices. Position derivatives therefore are
not the full mathematical derivatives of the rendered function. The code also
retains the reference's handling of the first cell and traversal termination.

Inference packs RGB plus padding into an aligned 64-bit word per texel, reorders
cells by Morton code, schedules rays in 4-by-8 tiles, and can skip texture lookup
for small contributions while still updating transmittance. The paper settings
are a 0.01 transmittance termination threshold and a 0.001 contribution threshold;
training uses 0.001 termination and no contribution skipping. The optional Q8,
depth-quantile and contribution-map interfaces are not implemented here.

Training camera queries use a Warp hash table of representative origins. Each
reuse checks exact coordinate equality; hash collisions fall back to a direct
hierarchy query. This removes the original `torch.unique(dim=0)` over the full
ray batch without changing the nearest-site query or assuming a fixed camera
set. The small direct-query path handles inference camera lookup.

## Reproduction

The tested environment is Python 3.10, Torch 2.7.1+cu126, Warp 1.19.0, NumPy 2.2.6,
SciPy 1.15.3 and NVIDIA L40. Native reference builds use CUDA Toolkit 12.6 and
Paragram 0.1.0. VoroTracing uses pycolmap 3.13.0; the earlier RadFoam environment
uses 3.11.1, so the experiment prepends an isolated installation for VoroTracing.

Additional tested packages include torchvision 0.22.1+cu126,
fused-local-corr 0.3.27, torchmetrics 1.9.0, pytorch-msssim 1.0.0, tyro 1.0.17,
wandb 0.31.0, viser 1.1.1, nerfview 0.1.3, splines 0.3.3 and jaxtyping 0.3.7.
Use these Torch-compatible pins when preparing an environment.

Clone VoroTracing and initialize its submodules at the commit above. RoMaV2 is
pinned to `95c9968145c8906b7b59383258e9f73b02853d89`. Configure external paths with
`VOROTRACING_SOURCE_DIR`, `ROMAV2_SOURCE_DIR`, `VOROTRACING_PYCOLMAP_PATH`,
`TORCH_HOME`, `TORCH_EXTENSIONS_DIR`, and `TORCH_CUDA_ARCH_LIST` as needed.
Historical defaults point to this experiment's workspace. Unique outputs and
source checkouts belong on persistent storage; model-download and build caches
are reproducible.

Run from the repository root. Use a new output directory for each fit.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.

# The shared dataset downloader belongs to the earlier RadFoam experiment.
.venv/bin/python experiments/radfoam-integration/fetch_bonsai.py --downsample 2

.venv/bin/python experiments/vorotracing-integration/prepare_scene.py \
  --output results/local/voro-initialization

taskset -c 0-3 .venv/bin/python experiments/vorotracing-integration/run_fit.py \
  --backend native --initialization results/local/voro-initialization/dense-points.pt \
  --output results/local/voro-native

taskset -c 0-3 .venv/bin/python experiments/vorotracing-integration/run_fit.py \
  --backend warp --morton --initialization results/local/voro-initialization/dense-points.pt \
  --output results/local/voro-warp
```

The defaults are the full indoor recipe: 20,000 steps, 2,000,000 sites,
1,000,000 rays per step, half-resolution images, and validation every 1,000 steps
on all 37 held-out views. Initialization uses only the 255 training images. The
original initializer samples the cached dense cloud with seed 42; the original
uniform-ray sampler uses a separate generator with seed 2026 to avoid races with
model/metric initialization. Reports hash the initial physical parameters and
the warmup batch. `--morton` optionally reorders initial parameters together.

The loop timer includes optimizer initialization, all geometry updates and
intermediate validation. Loading, initial setup/JIT, warmup and final checkpoint
serialization are recorded outside that timer. RoMa preprocessing is reported
separately. CPU affinity and quota counters are retained. Both reference and Warp
runs use the same counted version of the reference's nonfinite-gradient masking;
the reference's CUDA ray kernels and Paragram geometry are unchanged.

`report.json` is complete only after fitting and checkpoint serialization finish.
`progress.json` records the latest evaluation. Source snapshots, per-view scores,
numerical counters, geometry-update timings and source hashes accompany new runs.
The conservative `capped_rays` counter counts traversal lengths **at least** the
configured limit, including a ray that naturally terminates exactly there.
It covers training and validation after warmup. New runs also separate validation
and per-view counters. A successful fit is not an exact certification of every
WarPower adjacency edge.

## Validation and benchmarks

```bash
.venv/bin/python experiments/vorotracing-integration/validate_renderer.py \
  --output results/local/voro-parity.json
.venv/bin/python experiments/vorotracing-integration/validate_contract.py \
  --output results/local/voro-contract.json
.venv/bin/python experiments/vorotracing-integration/benchmark_renderer.py \
  --checkpoint results/local/voro-native/model/model.pt \
  --output results/local/voro-kernel-comparison.json --training
.venv/bin/python experiments/vorotracing-integration/evaluate_inference.py \
  --backend warp --checkpoint results/local/voro-warp/model/model.pt \
  --output results/local/voro-warp-inference.json
.venv/bin/python experiments/vorotracing-integration/evaluate_checkpoint.py \
  --backend warp --checkpoint results/local/voro-warp/model/model.pt \
  --output results/local/voro-warp-checkpoint.json
```

The renderer comparison uses an independent SciPy Delaunay graph and tests
outside/inside cameras, axis/seam directions, transparent and opaque cells,
traversal truncation, and small-scale cells. It compares outputs, distortion,
traversal counts, four parameter-gradient groups, and both FP16 inference settings. The independent
contract test checks density/texture directional derivatives and camera-cache
reuse with repeated and colliding origins.

The kernel benchmark compares native and Warp on the **same checkpoint and
graph**, with alternating timing order, and separates camera lookup, inference,
and forward/backward replay. The all-view evaluator calls the original inference
benchmark loop. Its timing includes nearest-camera lookup, but excludes input
ray generation and RGB compositing, as in the upstream benchmark. Inference PSNR
is measured with the actual timed thresholds and quantization settings.
The separate checkpoint evaluator restores FP32 parameters and rechecks held-out
PSNR, raw nonfinites and traversal lengths with the training renderer. Optional
previews are written outside the committed evidence.

The [full comparison](../../results/validation/vorotracing/full-comparison.json)
contains completed fits and saved checkpoints for both backends. All four fits
below use the same L40, Bonsai split and half-resolution held-out evaluation.
The VoroTracing implementation comparison uses the original full-budget trainer
with identical physical initialization and warmup rays. The Warp fit also uses
initial Morton ordering; the native fit retains its original order.

| Method | Loop time | Whole process | Final FP32 PSNR |
| --- | ---: | ---: | ---: |
| RadFoam, native | 89.23 min | 90.48 min | 32.414 dB |
| RadFoam, Warp geometry | 54.80 min | 56.05 min | 32.251 dB |
| VoroTracing, native | 86.69 min | 87.21 min | 33.739 dB |
| VoroTracing, Warp | **73.85 min** | **74.38 min** | **33.774 dB** |

Warp reduces the VoroTracing loop time by **14.8% (1.174x speedup)** while
preserving quality. Both runs have zero counted raw nonfinite gradients and
outputs. The cumulative conservative traversal-limit counts are 131 for native
and 33 for Warp; the final held-out evaluations have zero. The shared RoMa
preprocessing took 113.17 seconds, reported separately from the table.

This is one fit per backend on one scene, with no estimate of seed variance.
The +0.035 dB mean difference is not evidence of a systematic quality gain:
Warp improves 14 of 37 views, the median delta is -0.0055 dB, and view 36 accounts
for a +0.913 dB improvement. SSIM and LPIPS are essentially unchanged. The
native-versus-Warp comparison includes geometry, camera queries, tracing and
initial ordering; it does not isolate one optimization.

The first recorded Warp evaluation reaching the native VoroTracing final PSNR
is step 18,000 at 66.65 minutes (33.758 dB). It exceeds the prior RadFoam/Warp
endpoint at step 7,000 and 26.71 minutes (32.404 dB). These are sampled quality
threshold crossings; intermediate models were not saved. The complete 20,000-step
Warp checkpoint gains 1.523 dB over the old RadFoam/Warp checkpoint, at a longer
full-budget training time. The methods use different published recipes and
validation schedules, so those rows are method comparisons, not isolated kernel
speedups.

The optimized renderer matches native forward outputs, traversal counts and
FP16 outputs exactly in all seven parity cases, and FP16 output exactly on sampled full-resolution checkpoint views at both
quality settings. Maximum parameter-gradient relative L2
error is below 5e-7 in the final run. Independent directional-derivative checks have a maximum
relative error of 0.00211, and camera-cache tests have zero mismatches.

Matched 1,000-step, 262,144-site pilots reached 27.356 dB in 22.136 seconds with
native CUDA, 27.363 dB in 21.040 seconds with Warp, and 27.367 dB in 20.561 seconds
with initially Morton-ordered Warp parameters. They use 65,536 rays and
quarter-resolution images and record zero nonfinite gradients/outputs or
traversal-limit hits. A separate two-million-site preflight verifies identical
initialization and warmup-batch hashes and runs without the native extension.

On the completed native checkpoint with one million uniformly sampled training
rays, the Warp forward/backward pass takes 130.46 ms versus native CUDA's
151.02 ms. Both use the same Morton-relabeled graph and parameters. Comparing
64-, 128- and 256-thread launch blocks retains the original 256-thread choice;
smaller blocks are slower. This component test excludes optimizer, geometry and
sampling, and is separate from the whole-fit timing above. Reproduce it with
`benchmark_training_blocks.py --checkpoint PATH --output PATH`.

## Final checkpoint and inference checks

All 37 saved-checkpoint FP32 evaluations reproduce the fitting PSNR with either
renderer. The all-view FP16 benchmark evaluates the actual timed images: paper
settings use termination 0.01 and contribution skipping 0.001, while the
higher-fidelity setting uses 0.001 and no skipping. Do not attach FP32 training
PSNR to accelerated FP16 frame rates.

The [inference comparison](../../results/validation/vorotracing/inference-comparison.json)
records all 37 views. At the paper's FP16 thresholds, including nearest-camera
lookup but excluding ray generation/compositing:

| Fitted checkpoint | Inference renderer | Actual inference PSNR | FPS |
| --- | --- | ---: | ---: |
| Native VoroTracing | Native CUDA | 33.664 dB | 171.3 |
| Native VoroTracing | Warp, tuned | 33.664 dB | 178.2 |
| Warp VoroTracing | Native CUDA | 33.699 dB | 197.9 |
| Warp VoroTracing | Warp, tuned | 33.699 dB | 193.4 |

The same-checkpoint, alternating-order four-view measurements put tuned Warp
within 4% of native CUDA at both quality settings, with bit-exact FP16 output.
The larger difference between the two fitted checkpoints can reflect geometry,
appearance and layout; it cannot be attributed to the renderer alone. Separate
all-view jobs are serial, and small timing differences are not established over
independent sessions. The higher-fidelity setting gives 33.738 dB at 138.6 FPS
for the native checkpoint/native renderer, and 33.773 dB at 153.1 FPS for the
Warp checkpoint/Warp renderer.

Including ray generation, camera lookup, compositing and RGB8 packing gives:

| Fitted method and renderer | Actual RGB8 PSNR | FPS |
| --- | ---: | ---: |
| RadFoam native fit, native fused inference | 32.314 dB | 123.0 |
| RadFoam Warp-geometry fit, native fused inference | 32.170 dB | 136.0 |
| VoroTracing native fit, native tracing | 33.653 dB | 182.9 |
| VoroTracing Warp fit, all Warp GPU stages | **33.689 dB** | **200.7** |

These use termination 0.01, plus VoroTracing's contribution skipping 0.001.
Each uses its original background convention. Relative to our prior
RadFoam/Warp fit's fast inference, the complete Warp VoroTracing camera pipeline
is 1.48x faster and gains 1.519 dB on its actual RGB8 output. RadFoam's more
aggressive 0.05 termination reaches 148.7 FPS at 31.343 dB on that old checkpoint.
These L40 results preserve a useful quality/speed gain; they are not a
reproduction of the paper's RTX 5090 frame-rate ratio or full multi-scene table.
FPS throughout is 1000 divided by mean frame milliseconds, not mean per-view FPS.

The inference launch sweep uses the full native checkpoint and alternating
variant order. Flattening the fixed-size packed texture arrays and allowing up
to 80 registers improves inference on L40. These changes affect inference only;
the training kernels and full-fit measurements are unchanged. The final parity
suite passes after the change. The raw sweep and its exact source are retained
in the validation evidence, including slower alternatives. Its base renderer
is frozen at commit `d40269a`; see `provenance.json` before replaying the sweep.

`benchmark_camera_pipeline.py` additionally times ray generation, nearest-cell
lookup, tracing, background compositing and RGB8 packing. Its default white
background follows VoroTracing's original evaluation; `--background black`
reproduces RadFoam's native fast-kernel convention. VoroTracing was trained with
white residual background and loses substantial PSNR if forced to black. The
camera driver checks generated rays against dataset rays and packed RGB against
an independent Torch calculation. `benchmark_radfoam.py` retains both the older
native generic renderer and the fused camera-to-RGB8 kernel, with quality measured
at each timed setting.

`audit_checkpoint_graph.py` holds positions and appearance fixed, rebuilds the
adjacency with the other builder and renders all 37 views using native FP32
tracing. On the native checkpoint, replacing Paragram with WarPower removes
10,092 directed edges and adds 7,176 of about 30.4 million, changing mean PSNR by
-0.00105 dB. On the Warp checkpoint, replacing WarPower with Paragram removes
7,098 and adds 11,198, changing mean PSNR by -0.0660 dB. The latter Paragram
rebuild reports 2,654 nonzero statuses, including plane/vertex overflow and
inconsistent boundaries; the original trainer ignores these statuses. Both
graph-swap evaluations have zero raw nonfinites and traversal-limit hits. These
diagnostics do not identify the mathematically correct edge set and do not
resolve the earlier exact-predicate WarPower audit findings. Renderer parity is
measured on an identical graph; successful fitting does not certify geometry.

The [completion manifest](../../results/validation/vorotracing/status.json)
indexes the result summaries, source provenance and limitations. Full checkpoints,
rendered previews and the learning-curve plot remain under ignored persistent
`results/local/vorotracing-20261110/`; no checkpoints or image binaries are
committed.
