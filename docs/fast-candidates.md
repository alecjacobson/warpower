# Fast candidates for certified Delaunay meshes

This developing optimization belongs only to `DelaunayWorkspace`. The ordinary
power-diagram, original Delaunay/regular-mesh APIs and their kernels retain their
implementations. No float32 candidate is accepted without the existing exact
original-coordinate certificate.

`fast_candidates=True` is the default. Buffers smaller than 32,768 sites keep the
original constructor. Larger buffers also count distinct canonical sites on
device each reconstruction: fewer than 32,768 distinct sites select the original
float64 path. This protects coarse active prefixes during progressive meshing.
`fast_candidates=False` disables the new path for comparisons.

For eligible input, the first cell pass uses float32 geometry inside the site
bounds padded by their maximum extent. This clipping box is independent of the
full-hull expansion factor. A cell that fails clipping or retains any box-plane
vertex emits nothing and enters the existing compact float64 retry passes.
Every successfully emitted cell uses the same minimum-index tetrahedron owner
rule. The retry passes construct all candidates for failed owners, so they do
not append replacements beside stale candidates from those owners.

A finite float32 cell can still be geometrically wrong. The candidate first
passes the existing floating-point validation, then the full exact certificate
against original coordinates and the common symbolic lifting. If either rejects
it, the original full float64 constructor runs before auxiliary or symbolic
recovery. All stages have finite existing capacities and expansion budgets.
Input positions never move, and any final failure publishes a zero count.
This design does not claim that float32 clipping itself is conservative or exact.

Failed fast attempts are disabled on subsequent replays when `cache_strategy`
is enabled. Current geometry is nevertheless rebuilt and certified every replay.
`cache_strategy=False` retries the fast path each time. Invalid input resets
this preference, and a fresh complete meshing solve resets it alongside the
existing reconstruction preference. Strategy state never stands in for a
geometry check.

The private candidate workspace reserves float32 cell scratch and scalar state
at setup. It reuses the ordinary builder's candidate buffers and float64 retry
storage. Its full-hull box factor is separate, preserving the original builder's
factor for fallback. Temporary Python reference swaps select scratch and kernel
variants while recording a graph; they are restored before symbolic capture and
perform no work during replay. Consumers retain fixed output pointers and use
the published device count/error as before.

Diagnostics are device scalars, reset per replay:

| Field | Meaning |
|---|---|
| `fast_candidates` | Setup Boolean: the option is enabled and the buffer has at least 32,768 sites |
| `fast_attempted` | The distinct-site gate selected the hybrid candidate in this replay |
| `fast_accepted` | That candidate passed the exact certificate |
| `fast_error` | Rejected fast attempt: bit 1 for construction, bit 2 for certification |
| `fast_repaired_cells` | Number of failed or box-touching cells after the last float32 pass |

`fast_error` describes an intermediate attempt; only the existing `error` and
output count govern whether the final mesh can be consumed. A successful original
fallback can leave `fast_error` nonzero. `base_attempted` continues to describe
the unweighted construction path, which may now include both builders.

## Validation and measured scope

The [focused cases](../results/validation/vtm/fast-candidate-focused-tests.txt)
cover independent SciPy connectivity, large storage with a small distinct
prefix, injected construction/certificate failures, cached and uncached
fallback, invalid-input recovery, a fixed-pointer producer/consumer graph,
disabling the option, fresh-mesher preference reset, and a natural float32
construction failure. That natural failure falls back to a matching original
mesh; it is not hidden by the successful random-cloud examples. Failed fast
attempts can add work to a first call before the cached original path takes over.

The [100,000-site reference](../results/validation/vtm/fast-candidate-100k-cgal.json)
matches all 671,908 tetrahedra against native CGAL 6.2.1 and the preceding
certified constructor. The [isolation record](../results/validation/vtm/fast-candidate-isolation.json)
confirms that all 44 other pre-existing runtime modules and the main README
retain their contents. The changed existing modules are the optional certified
wrapper and the complete mesher's fresh-solve preference reset.

The [paired experiment](../results/validation/vtm/fast-candidate-performance.json)
on an NVIDIA L40 with Warp 1.19.0 reduced complete certified reconstruction of
100,000 random sites from 563.00 to 263.07 ms (2.14×). First replay fell from
3,383.18 to 1,451.81 ms (2.33×). Five warmed samples exclude allocation, compilation,
input upload and host validation; both timing windows were unthrottled. The fast
pass repaired 1,512 cells in float64, and required no auxiliary or symbolic
recovery. These measurements concern reconstruction, not complete VTM or a CPU
comparison.

The original [cutoff experiment](../results/validation/vtm/fast-candidate-distributions.json)
showed a 13.4% regression at 8,192 sites (34.71 to 39.38 ms), a small gain at
10,000, and larger gains at 32,768. Consequently the automatic cutoff is now
32,768. The smaller measurements are retained as experimental evidence, not as
performance claims for the final default. A final-policy distribution matrix,
full regression, padded memory and native changing-input checks are pending.

The complete default 10,000-site torus solve was effectively unchanged:
[76,883.76 ms before](../results/validation/vtm/fast-candidate-mesher-before10000.json)
and [76,967.00 ms with the provisional cutoff](../results/validation/vtm/fast-candidate-mesher-after10000.json).
Each is the median of three unthrottled full replays, including domain refit,
sampling, growth, fitting, quality rounds and extraction. Point positions,
roles and canonical full/material/boundary connectivity are
[identical](../results/validation/vtm/fast-candidate-mesher-equivalence.json).
There are 60,013 full cells, 51,140 retained cells and 6,166 boundary triangles.
Minimum radius ratio is 0.03132, median 0.89041, relative volume error −0.894%,
and directed surface RMS 0.003791. The final cutoff keeps this size on the
original constructor. This optimization does not establish an end-to-end VTM
speedup on that case. The native CPU performance goal remains open.

Reproduce standalone comparisons from a source checkout with the development
extra installed:

```bash
python -m benchmarks.delaunay_recovery --case random --n 100000 --repeats 5 --no-fast --output original.json
python -m benchmarks.delaunay_recovery --case random --n 100000 --repeats 5 --output hybrid.json
python -m benchmarks.fast_candidate_matrix --output distributions.json
```
