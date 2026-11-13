# Installation

Requires Python 3.10+ and an NVIDIA CUDA GPU.

```sh
git clone https://github.com/alecjacobson/warpower.git
cd warpower
python -m venv .venv
source .venv/bin/activate
```

For CUDA 12 drivers, install the compatible Warp wheel before installing Warpower:

```sh
pip install 'https://github.com/NVIDIA/warp/releases/download/v1.19.0/warp_lang-1.19.0+cu12-py3-none-manylinux_2_28_x86_64.whl'
```

Then install Warpower:

```sh
pip install -e '.[dev,bench]'
```

For the measured repeated-call setup on an L40, configure Warp before construction:

```python
import warp as wp
wp.set_mempool_release_threshold("cuda:0", 32 * 1024**3)
```

# Best measured result

On the paper's five-cluster distribution with 100,000 sites, Warpower constructs box-clipped Voronoi neighbor CSR on an NVIDIA L40 in **30.4 ms**, versus **66.0 ms** for Paragram: **2.17× faster** in this recorded case. All Warpower cells report success; Paragram reports four boundary failures and one empty cell.

Both timings include construction allocation calls, use resident GPU inputs, and average ten runs after three warm-ups. Warp retains up to 32 GiB in its allocator; PyTorch uses its normal caching allocator. Both timing series passed the CPU-throttling check. An independent double-precision audit of all 359 disagreeing rows finds no extra or missing Warpower edges; shared rows were not audited. [Warpower measurements](results/warpower-l40/matrix/cluster5-100000-voronoi.json), [Paragram measurements](results/paragram-l40/matrix/cluster5-100000-voronoi.json), [audit and graph equivalence](docs/numerical-audit.md), [full comparison](results/summary/report.md), [reproduction](docs/benchmarks.md).

These are fresh neighbor-construction timings, not a ranking of Delaunay tetrahedron builders or incremental updates. The historical 74-case matrix favors Warpower in 62 cases (1.220× geometric mean); the 9,216-site live Bonsai input instead favors Paragram and native RadFoam. A later exact facet audit found seven incorrect Warpower neighbor pairs across four large scene checkpoints, so the checkpoint timing wins do not establish equally exact output. The [claim audit](experiments/benchmark-claims-audit/README.md) preserves the recorded measurements, distinguishes implementations and correctness coverage, and tracks the required regression checks in [issue #300](https://github.com/alecjacobson/warpower/issues/300).
