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

On the paper's five-cluster distribution with 100,000 sites, Warpower constructs power-diagram adjacency on an NVIDIA L40 in **22.6 ms**, versus **66.3 ms** for Paragram: **2.93× faster**. Warpower reports 98,871 successful and 1,129 empty cells, with no failed cells.

Both timings include construction allocation calls, use resident GPU inputs, and average ten runs after three warm-ups. Warp retains 1 GiB in its allocator; PyTorch uses its normal caching allocator. An independent double-precision audit of all 708 disagreeing rows finds two extra directed Warpower edges and none missing; shared rows were not audited. Broader performance and numerical parity remain in progress. [Warpower measurements](results/warpower-l40/matrix/cluster5-100000-power.json), [Paragram measurements](results/paragram-l40/matrix/cluster5-100000-power.json), [adjacency audit](results/warpower-l40/robust-cluster5-100k-power-audit.json), [reproduction](docs/benchmarks.md).
