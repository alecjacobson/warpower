# RadFoam fitting integration

This experimental adapter runs real Mip-NeRF 360 Bonsai scene fitting with
WarPower Voronoi adjacency and the upstream RadFoam renderer and backward pass.
It also compares native incremental triangulation with current VTM fragment
reuse. Production WarPower and VTM defaults are unchanged.

The 20,000-step reduced-budget fit on NVIDIA L40 took 184.15 seconds with
`warpower-morton`, versus 208.10 seconds with native RadFoam, an observed 1.13x
speedup. Mean PSNR over 37 held-out views was 26.83 versus 26.73 dB. These are
single full runs with slightly different final site counts; the component
benchmarks use repeated, alternating-order samples. Loading, initial JIT/warmup
and final checkpoint serialization are outside the fitting-loop timer.

| Backend | Fit seconds | Held-out PSNR | Final sites |
| --- | ---: | ---: | ---: |
| Native RadFoam | 208.10 | 26.73 | 249,858 |
| WarPower, input order | 372.04 | 26.76 | 253,903 |
| WarPower, Morton order | 184.15 | 26.83 | 247,608 |

All three fits completed pruning/densification and a fresh-Scene checkpoint
restoration, with identical restored Adam moments and rendering. Raw renderer
outputs and gradients had no nonfinite values or traversal-limit hits. The
checkpoint check runs inside the fitting process; this is not a separate-process
resume CLI, and the independent minibatch generator remains alive during it.

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
hashes. It does not download the entire 12.5 GB archive. Outputs include the
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

The fitting example uses 65,536 rays per step, approximately 250,000 final sites,
SH3 and eight-view densification error maps. It is a reduced-budget integration
example, not a reproduction of the paper's quality numbers. WarPower returns
bounded adjacency, while native triangulation returns a full-hull mesh and
adjacency. The earlier exact audit found seven incorrect undirected WarPower
pairs across four larger checkpoints; successful fitting does not resolve them.
Density here is optical density, not a power weight. CUDA default-stream use is
tested; arbitrary stream interleaving is not qualified.
