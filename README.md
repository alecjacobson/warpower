# Installation

Requires Python 3.10+ and an NVIDIA CUDA GPU.

```sh
git clone https://github.com/alecjacobson/warpower.git
cd warpower
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,bench]'
```

For CUDA 12 drivers, install the compatible Warp wheel before installing Warpower:

```sh
pip install 'https://github.com/NVIDIA/warp/releases/download/v1.19.0/warp_lang-1.19.0+cu12-py3-none-manylinux_2_28_x86_64.whl'
```

# Best measured result

The initial complete implementation constructs 100,000 uniformly distributed Voronoi cells on an NVIDIA L40 in **281.2 ms** with all cells reporting success. Paragram takes **13.0 ms** on the same dataset; performance parity has not yet been reached.

Both timings include construction allocations, use resident GPU inputs, and average ten runs after three warm-ups. [Reproducible measurements](results/warpower-l40/initial-uniform-100k.json).
