# Fast candidates for certified Delaunay meshes

This developing optimization belongs only to `DelaunayWorkspace`. The ordinary
power-diagram, original Delaunay/regular-mesh APIs and their kernels retain their
implementations. No float32 candidate is accepted without the existing exact
original-coordinate certificate.

`fast_candidates=True` is the default. Buffers smaller than 8,192 sites keep the
original constructor. Larger buffers also count distinct canonical sites on
device each reconstruction: fewer than 8,192 distinct sites select the original
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
| `fast_candidates` | Setup Boolean: the option is enabled and the buffer has at least 8,192 sites |
| `fast_attempted` | The distinct-site gate selected the hybrid candidate in this replay |
| `fast_accepted` | That candidate passed the exact certificate |
| `fast_error` | Rejected fast attempt: bit 1 for construction, bit 2 for certification |
| `fast_repaired_cells` | Number of failed or box-touching cells after the last float32 pass |

`fast_error` describes an intermediate attempt; only the existing `error` and
output count govern whether the final mesh can be consumed. A successful original
fallback can leave `fast_error` nonzero. `base_attempted` continues to describe
the unweighted construction path, which may now include both builders.

## Integration status

Ten focused tests pass, including independent SciPy connectivity, large storage
with a small distinct prefix, injected construction/certificate failures, cached
and uncached fallback, invalid-input recovery, fixed-pointer producer/consumer
capture, disabling the option, and fresh-mesher preference reset. Full regression,
paired complete-pipeline timing, independent large reference, padded memory and
native changing-input capture checks are pending. This page does not claim a
native CPU speedup; the overall meshing performance goal remains open.
