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

On the paper's five-cluster distribution with 100,000 sites, Warpower constructs Voronoi adjacency on an NVIDIA L40 in **30.4 ms**, versus **66.0 ms** for Paragram: **2.17× faster**. All Warpower cells report success; Paragram reports four boundary failures and one empty cell.

Both timings include construction allocation calls, use resident GPU inputs, and average ten runs after three warm-ups. Warp retains up to 32 GiB in its allocator; PyTorch uses its normal caching allocator. Neither measured run was CPU-throttled. An independent double-precision audit of all 359 disagreeing rows finds no extra or missing Warpower edges; shared rows were not audited. Broader performance and numerical evaluation remain in progress. [Warpower measurements](results/warpower-l40/matrix/cluster5-100000-voronoi.json), [Paragram measurements](results/paragram-l40/matrix/cluster5-100000-voronoi.json), [adjacency audit](results/warpower-l40/consistency-cluster5-100k-voronoi-audit.json), [reproduction](docs/benchmarks.md).
