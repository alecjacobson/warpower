# Current-cell continuation experiment

The bounded continuation prototype for [#271](https://github.com/alecjacobson/warpower/issues/271)
produces correct tested meshes but **does not improve construction time**. All four
implementations are retained; none is selected by public routines. This differs
from [previous-neighbor clipping](neighbor-clipping.md): it resumes unfinished work
within one current-input construction, rather than reusing neighbors across frames.

Before clipping a nearly full cell, the prototype saves its valid polytope,
unprocessed BVH queue and pending leaf offset into a preallocated queue. A larger
capacity pass resumes that state. Queue exhaustion retains the existing fresh
retry. Every new construction and clipping-box expansion clears saved prefixes.
The final exact certificate and normalized/symbolic recovery remain unchanged.

## Measurements

NVIDIA L40, Warp 1.19.0. Each entry is the median of nine alternating pairs of
complete certified graph replays, including queue management and all fallbacks.
Each graph uses the same workspace/output arrays. Setup, uploads and independent
readback are outside replay timing. No CPU-quota throttling was observed. These
are fresh Delaunay constructions, not full VTM solves or CPU comparisons.

| Implementation | Random 1K baseline / experiment (ms) | Uniform Armadillo 8K baseline / experiment (ms) | Graded Armadillo 8K baseline / experiment (ms) |
|---|---:|---:|---:|
| v1: array descriptors in a struct | 9.198 / 11.947 | 107.107 / 110.555 | 109.406 / 113.569 |
| v2: specialize fresh pass; out-of-line save/restore | 9.191 / 12.882 | 107.253 / 117.091 | 109.546 / 117.854 |
| v3: cap fresh-pass registers at original 96 | 9.199 / 13.654 | 107.278 / 123.507 | 109.361 / 124.969 |
| v4: pack state into two buffers | 9.203 / 12.931 | 107.257 / 117.977 | 109.355 / 119.487 |

The random cloud saves no cells, yet pays 30–48% overhead. Both scan clouds save
and resume one cell per construction. Even v1's best scan result is 3.2–3.8%
slower. The separately measured larger-initial-capacity control remains about
15% faster on these fixed scans; that does not qualify a global capacity change.

Kernel properties for v2 report 128 registers versus the original 96, and 792
additional local-memory bytes per thread at each tested capacity. This suggests
that adding continuation state to the common kernel is costly, but is not a
complete causal profile. The register cap worsened performance; reducing the
cache to two buffers did not recover the loss. The v4 cache adds 3,576,592 device
bytes for 8,192 sites and 64 continuation slots, versus 3,577,104 bytes for v1–v3.
Setup measurements and individual samples are retained in the compact evidence.

## Checks and limits

All 216 measured outputs (four variants, three inputs, two graphs, nine samples)
match independent SciPy/Qhull connectivity and have zero construction errors.
Each variant passes three functional tests covering high-degree cells, changed
captured inputs, invalid-input recovery, bounded queue exhaustion and cospherical
symbolic recovery. The final variant also passes all six isolated padded and
allocation-lifetime memory checks, with zero reported errors and explicit 32 KiB
instrumentation stack allowance.

This is a benchmark-local ordinary-Delaunay prototype. Weighted APIs and the
broader production regression matrix were not qualified for it. No `warpower/`
file is changed by this experiment, so the negative result does not add runtime
cost to power diagrams, standalone Delaunay or VTM. Difficult-cell routing and
cooperative clipping remain open in #271/#98; this result does not rule them out.

Source pins, inputs, every sample, memory-check commands and raw-file hashes are
in [`experiments/cell-continuation`](../experiments/cell-continuation/README.md).
The raw logs stay outside ordinary Git.

## Reproduce

```bash
python -m pytest -q tests/test_cell_continuation_experiment.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python -m benchmarks.cell_continuation \
  --input results/local/armadillo-uniform/mesh.npz --repeats 9 \
  --output results/local/cell-continuation/uniform.json
```

Omit `--input` for the seeded random cloud. Scan mesh generation is documented in
[native surface queries](native-surface-queries.md). Historical variants are pinned
by their commits in `pilots.json`; the current benchmark contains v4.
