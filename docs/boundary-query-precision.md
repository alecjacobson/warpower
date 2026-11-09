# Boundary nearest-site query precision

Boundary fitting can opt into FP32 distances and bounds for assigning surface and
crease samples to moving sites. Masses, moments, selected-owner distances used
in energy diagnostics, and global energy sums remain FP64. Distinct-corner
matching and topology predicates retain their existing arithmetic.

```python
from warpower import BoundaryWorkspace, VariationalMeshingWorkspace

fit = BoundaryWorkspace(domain, sites, query_precision="float32")
mesh = VariationalMeshingWorkspace(domain, 8192,
                                  boundary_query_precision="float32")
```

`VTMWorkspace` also accepts `boundary_query_precision`. The gallery option is
`--boundary-query-precision float32`. All defaults remain `float64`; default
device kernels and standalone power-diagram/Delaunay paths are unchanged.

This choice is independent of `SurfaceDomain(query_backend="native")`. Native
surface queries search the input triangle mesh. Boundary queries search the
moving **sites** using the existing Morton hierarchy; fixed surface/crease
samples are the query locations. No reference query or audit runs behind each
FP32 query. Selection occurs before capture, with no new replay allocation,
host geometry decision or host transfer.

FP32 can change nearest-site assignments close to ties, and consequently can
change optimization decisions and connectivity. Energy is recomputed in FP64
for the selected owner; this does not undo an approximate assignment. Final
mesh validity still uses the existing complete certificates and extraction
checks. Prefer the reference mode for extreme coordinate scales or when the
FP32 range is inadequate. A query with no finite candidate reports boundary
error 16 and suppresses updates; it does not silently choose an arbitrary site.

## Complete-solve results

On NVIDIA L40 / Warp 1.19.0, median of three additional fresh captured solves:

| Example | FP64 queries | FP32 queries | Change in final output |
|---|---:|---:|---|
| Uniform Armadillo 8K | 2,599.71 ms | 2,348.06 ms | Bitwise identical |
| Graded Armadillo 8K | 2,744.90 ms | 2,428.61 ms | Bitwise identical |
| Torus 1K | 483.38 ms | 456.29 ms | Bitwise identical |
| Mechanical plate 1K | 746.57 ms | 705.35 ms | Bitwise identical |
| Cavity 1K | 339.27 ms | 319.11 ms | Different valid mesh; quality tradeoff below |

All configurations use native surface queries and trusted input. Timing includes
refit, sampling, growth, fitting, quality stages and extraction; setup, transfers,
export and independent checking are excluded. These are sequential complete-solve
ablations, not CPU comparisons. All timing windows are free of CPU-quota throttling.
Each output passes independent full-hull/Delaunay, positive-cell/material-incidence,
boundary-link, self-intersection and exact VTU round-trip checks.

The cavity's accepted selective steps change from eight to four. Minimum radius
ratio falls from 0.236810 to 0.094167; mean quality changes from 0.840038 to 0.833367.
Independent two-sided sampled RMS improves from 0.167783% to 0.155172% of the input
bounding-box diagonal. Its timing is **not a quality-matched speedup**.
The precision option remains explicit; [#276](https://github.com/alecjacobson/warpower/issues/276)
tracks a mixed-precision tie experiment and diagnosis of this trajectory change.

## Why accumulation remains FP64

The four-mode experiment at commits `fed4b65` / `9d64e52` tested query precision
and atomic accumulation precision independently. Direct FP32 accumulation caused
both uniform and graded 8K Armadillo solves to fail final material-boundary
validation: parent error 64, extraction diagnostics 2, and zero published mesh
counts. These are rejected experiments, not speedups. The uniform/graded reference
and FP32-query-only runs succeeded.

On four recorded 8K trajectory states, FP32 accumulation changed target positions
by at most about 6.7e-8 and relative masses by about 1.5e-6. Those small local
differences did not establish equivalent final behavior. FP32 queries alone
changed no owners or target coordinates on those sampled states, while a
deliberate near-tie unit test demonstrates that owner changes are possible.
[Issue #274](https://github.com/alecjacobson/warpower/issues/274) tracks accumulation
and trajectory sensitivity. The rejected prototype remains in Git history and
its raw reports remain separate from the accepted implementation's measurements.

## Reproduce numerical diagnostics

The independent diagnostic compares complete boundary passes on identical site
coordinates, alternates timing order, and reports changed owners, distance excess,
target displacement relative to local spacing, mass errors, and energy changes.
Reference comparison and readback happen outside timing and production replay.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python -m benchmarks.boundary_precision \
  --input results/local/armadillo-input/armadillo.off \
  --sites results/local/neighbor-clipping/actual-trials/frames.npy \
  --output results/local/boundary-query/numerical.json
```

Use a gallery `mesh.npz` instead of a trajectory array to audit final sites, and
add `--graded` for the gallery's prescribed radial sizing field. The trajectory
recorder is described in [the clipping experiment](neighbor-clipping.md).
To reproduce the four-mode rejected accumulation study, use its pinned commit
in a separate checkout; the current API intentionally exposes only the qualified
query-precision choice.

## Default-path regression controls

The full ordinary suite passes 1,411 tests. The alternating paired public geometry
matrix passes its 2% regression gate, with equal outputs and unchanged launch and
allocation traces. Standalone Delaunay full-replay medians change by +0.03%, +0.19%,
-0.14% and +0.32% on random 1K, random 100K, lattice and fitted-hand inputs.
The default reference torus solve changes from 488.25 to 489.17 ms (+0.19%).
These full-call timing windows have no CPU-quota throttling. One isolated lattice
base-stage diagnostic has a throttled period; it is retained and excluded from
clean timing claims. Isolated certificate timings vary, and are not speed claims.

An AST audit finds all existing device definitions in the three modified runtime
modules unchanged. The other 74 existing runtime files are byte-identical.
Compact reports, independent checks, full sample timings, source hashes and raw
artifact hashes are in [experiments/boundary-query-precision](../experiments/boundary-query-precision/).
Full local meshes and rejected experiments are retained outside Git.

All 11 selected precision/capture/recovery cases pass both padded memcheck and
pool-disabled allocation-lifetime tracking in isolated processes, with zero
errors (diagnostic stack 32 KiB). This includes two fresh complete reference-domain
mesher replays. A combined run stalled during a later domain constructor after
four completed tests; it was cancelled and remains recorded as incomplete under
[#155](https://github.com/alecjacobson/warpower/issues/155). Isolation does not
resolve that issue. The separate native surface backend retains the limitations
in [its documentation](native-surface-queries.md), including #267 and #269.
