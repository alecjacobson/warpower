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
  --backend warp --initialization results/local/voro-initialization/dense-points.pt \
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

The [validation evidence](../../results/validation/vorotracing/status.json)
records the completed native reference: **33.739 dB in 86.69 minutes** of loop
time, or 87.21 minutes including setup and saving. Its counted raw gradient and
output nonfinites are zero. Full-budget Warp fitting and final inference
comparisons are still in progress.

The optimized renderer matches native forward outputs, traversal counts and
FP16 outputs exactly in all seven parity cases, and FP16 output exactly on three
sampled full-resolution pilot views. Maximum parameter-gradient relative L2
error is below 4e-7. Independent directional-derivative checks have a maximum
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
sampling, and does not establish whole-fit speed. Reproduce it with
`benchmark_training_blocks.py --checkpoint PATH --output PATH`.
