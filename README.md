# Warpower

GPU Voronoi/power neighbors, Delaunay/regular tetrahedra, and variational
tetrahedral meshing implemented in NVIDIA Warp.

See the [consolidated public API](docs/api.md) for all exported functions and
workspaces, or the [documentation guide](docs/README.md) for examples, algorithms
and benchmark records.

## Installation

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

## Minimal example

```python
import numpy as np
import warpower

points = np.random.default_rng(42).random((1000, 3), dtype=np.float32)
diagram = warpower.voronoi_diagram(points)
status = diagram.status.numpy()
if not np.isin(status, [0, 3]).all():  # Success or a valid empty cell.
    raise ValueError("Neighbor construction failed")
offsets = diagram.offsets.numpy()
start, end = map(int, offsets[:2])
neighbors_of_site_zero = diagram.adjacency[start:end].numpy()
```

This constructs bounded neighbor CSR. Use [`DelaunayWorkspace`](docs/api.md#delaunayworkspace)
for certified unweighted convex-hull tetrahedra, and
[`VariationalMeshingWorkspace`](docs/api.md#variationalmeshingworkspace)
to fit a supplied triangle boundary. Repeated device loops use preallocated
[workspaces and Warp CUDA capture](docs/graphs.md).

## Recorded performance

On the paper's five-cluster distribution with 100,000 sites, Warpower constructs box-clipped Voronoi neighbor CSR on an NVIDIA L40 in **30.4 ms**, versus **66.0 ms** for Paragram: **2.17× faster** in this recorded case. All Warpower cells report success; Paragram reports four boundary failures and one empty cell.

Both timings include construction allocation calls, use resident GPU inputs, and average ten runs after three warm-ups. Warp retains up to 32 GiB in its allocator; PyTorch uses its normal caching allocator. Both timing series passed the CPU-throttling check. An independent double-precision audit of all 359 disagreeing rows finds no extra or missing Warpower edges; shared rows were not audited. [Warpower measurements](results/warpower-l40/matrix/cluster5-100000-voronoi.json), [Paragram measurements](results/paragram-l40/matrix/cluster5-100000-voronoi.json), [audit and graph equivalence](docs/numerical-audit.md), [full comparison](results/summary/report.md), [reproduction](docs/benchmarks.md).

These are historical fresh neighbor-construction timings, not a ranking of
Delaunay tetrahedron builders or incremental updates. The recorded 74-case
matrix favors Warpower in 62 cases (1.220× geometric mean). A subsequent audit
found seven wrong Warpower neighbor pairs; later numerical recovery fixed those
cases. The original measurements remain unchanged.

The latest merged [capacity-recovery comparison](experiments/cell-capacity-recovery/README.md)
passes the 74-case regression gate and reports zero errors in 3,762 exact
directed-facet audits across five scene inputs, including all seven original
pairs. Unexamined shared edges are not certified. On the captured 9,216-site
live Bonsai input, separate fresh-call means are **37.790 ms Warpower**,
**42.180 ms Paragram** (143 failed cells), and **21.320 ms native RadFoam**.
RadFoam remains faster on this small case and produces unbounded neighbors plus
tetrahedra; Warpower/Paragram produce bounded neighbors. These are qualified
construction timings, not identical-output speed ratios.

See the [larger RadFoam comparison](experiments/radfoam-fresh/README.md),
[claim audit](experiments/benchmark-claims-audit/README.md), and
[reproduction requirements](docs/benchmarks.md).
[Issue #300](https://github.com/alecjacobson/warpower/issues/300) tracks remaining
qualification work, including broader captured-workspace sanitizer findings.
