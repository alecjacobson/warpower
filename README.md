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

On the paper's five-cluster distribution with 100,000 sites, Warpower constructs Voronoi adjacency on an NVIDIA L40 in **32.9 ms**, versus **65.7 ms** for Paragram: **2.00× faster**. All Warpower cells report success; Paragram reports four boundary failures and one empty cell.

Both timings include construction allocations, use resident GPU inputs, and average ten runs after three warm-ups. Adjacency is not identical: the edge sets differ on 816 of 1,545,814 directed edges in their union. Broader performance and numerical parity remain in progress. [Warpower measurements](results/warpower-l40/radix-cluster5-100k.json), [Paragram measurements](results/paragram-l40/radix-cluster5-100k.json), [adjacency comparison](results/warpower-l40/radix-cluster5-100k-comparison.json).
