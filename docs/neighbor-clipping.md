# Previous-neighbor clipping experiment

The prototype for [#272](https://github.com/alecjacobson/warpower/issues/272)
does not replace the current VTM tetrahedron-fragment reuse strategy. On the
tested clouds and real Armadillo reconstruction transitions, fragment reuse is
faster. The implementation and measurements remain available to guide further
cell-construction work. No `warpower/` file or public default changed.

## What was tested

After a successful exact-certified construction, a device hash table collects
tetrahedron edges as candidate Voronoi neighbors. Each site reserves 128 hint
slots. Before the next construction, the hints are ordered by current distance.
Each cell starts from its **current clipping box**, clips current bisectors for
those neighbor IDs, and then performs the complete current-input spatial search.
The prototype carries no old vertex coordinates or validity proofs forward.
Canonical owners are resolved again; truncated hint lists cannot omit the final
search. Original-coordinate certification and normalized/symbolic recovery remain
in place. A failure clears the cache.

The cache occupies 8,421,380 bytes for 8,192 sites, including ordered hints and
counts. Timings include ordering, cache construction and update. The experimental
instance hooks are in `benchmarks/`, and are never imported by public routines.
This prototype covers ordinary Delaunay reconstruction; it is not a qualified
weighted meshing API.

## Fixed and perturbed point clouds

NVIDIA L40, Warp 1.19.0; median of five warmed GPU-event samples. These initial
strategy sweeps are sequential, not alternating paired release gates. Each sample
starts from the same preceding input and measures construction of the same next
input. Preparation, transfer, setup and independent checking are excluded.
Motion is independent Gaussian noise per coordinate, with standard deviation
expressed as a fraction of the bounding-box diagonal.

| Input | Motion | Fresh (ms) | Fresh, capacity 256 (ms) | Neighbor hints (ms) | Hints, capacity 256 (ms) | Existing fragment reuse (ms) |
|---|---:|---:|---:|---:|---:|---:|
| Uniform Armadillo 8K | 0 | 107.22 | 91.17 | 86.61 | 79.85 | 1.76 |
| Uniform Armadillo 8K | 0.0001 | 104.39 | 89.76 | 84.84 | 79.36 | 8.80 |
| Uniform Armadillo 8K | 0.01 | 50.46 | 50.46 | 55.51 | 55.41 | 35.81 |
| Graded Armadillo 8K | 0 | 109.41 | 92.53 | 87.52 | 80.35 | 1.75 |
| Graded Armadillo 8K | 0.0001 | 108.24 | 92.43 | 86.43 | 79.28 | 7.38 |
| Graded Armadillo 8K | 0.01 | 45.05 | 44.95 | 47.94 | 47.96 | 34.13 |
| Uniform random 1K | 0 | 8.81 | 8.86 | 9.31 | 9.32 | 0.42 |
| Uniform random 1K | 0.0001 | 8.77 | 8.78 | 9.32 | 9.31 | 1.33 |
| Uniform random 1K | 0.01 | 7.74 | 7.73 | 9.70 | 9.67 | 3.91 |

All 45 results have zero construction/certification errors and match independent
SciPy/Qhull connectivity. Timing windows report zero CPU-quota throttling. These
are reconstruction comparisons, not complete-solve or CPU-mesher speedups.
The current strategy also has a fresh local-wall fallback: on the two large-motion
Armadillo cases it succeeds after fragment completion fails. That fallback is
included in the reported current-strategy time.

## Actual VTM trials and qualification

`benchmarks.neighbor_clipping_trials` records full-budget reconstruction inputs
from a fresh native-query 8K uniform Armadillo solve, including rejected trials.
It selects the first two transitions, the last transition and the transitions
with minimum, median and maximum largest-site displacement, deduplicating indices.
The comparison alternates strategy order across five measured repetitions. Each
pair is initialized from its preceding input; this is an isolated transition
study, not timing the entire VTM trajectory with a replacement constructor.

The recorded solve produced 180 full-budget inputs. Deduplication selected four
transitions; all 12 outputs matched independent SciPy/Qhull connectivity, with
zero construction errors and zero CPU-quota throttling:

| Transition to frame | Fresh (ms) | Hints, capacity 256 (ms) | Existing fragment reuse (ms) |
|---|---:|---:|---:|
| 1 | 142.82 | 95.68 | 1.69 |
| 2 | 138.45 | 114.38 | 14.37 |
| 162 | 106.98 | 79.70 | 2.78 |
| 179 | 106.24 | 79.60 | 4.26 |

Compact results and source hashes are in
[`experiments/neighbor-clipping`](../experiments/neighbor-clipping/README.md).
Changing-input tests cover stale neighbors, duplicate-owner remapping, deliberately
truncated hints, nonfinite failure and recovery, and cospherical symbolic recovery.
The default runtime is byte-identical to main before this experiment; production
performance gates are therefore not rerun for this benchmark-only addition.

The measured negative result does not rule out a cooperative clipping algorithm,
partial reconstruction, or a different cache representation. It does show that
this implementation should not be selected instead of the current VTM reuse path.
Larger initial cells separately reduce fixed-scan construction cost by about 15%;
that observation feeds [#271](https://github.com/alecjacobson/warpower/issues/271).
It does not justify a public default change without the broader gates in #270.

## Reproduce

Install the development dependencies. The scan inputs can be generated with the
gallery recipe in [native surface queries](native-surface-queries.md).

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python -m benchmarks.neighbor_clipping \
  --input results/local/armadillo-uniform/mesh.npz --movement 0.0001 \
  --strategies fresh larger neighbors neighbors-larger fragment --repeats 5 \
  --output results/local/neighbor-clipping/sweep.json

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python -m benchmarks.neighbor_clipping_trials \
  --record-input results/local/armadillo-input/armadillo.off \
  --output results/local/neighbor-clipping/actual-trials

python -m pytest -q tests/test_neighbor_clipping_experiment.py
```

Omit `--record-input` to remeasure an existing recorded `frames.npy`. Raw frame
arrays, meshes and logs belong under ignored `results/local/`, not ordinary Git.
