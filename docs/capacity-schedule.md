# Original-coordinate first-pass capacity

Starting FP64 cell construction at 256 slots avoids rebuilding a rare expensive
cell that outgrows 128 slots. On fixed 8K Armadillo clouds it reduces complete
certified construction by about 15%, with equal canonical full-hull connectivity.
This is a benchmark experiment; public runtime defaults are unchanged here.

## Full matrix

NVIDIA L40 / Warp 1.19.0, nine alternating paired captured constructions. Each
replay starts fresh with `cache_strategy=False`; setup, upload, output readback
and independent CPU checks are separate. In particular, the lattice and hand
times include failed construction strategies on every replay. They are not the
much shorter cached-strategy timings of an unchanged cloud. All 450 measured
outputs are repeatable within each variant; 50 saved outputs are independently
validated, and each pair has identical canonical connectivity. All timing windows
have full CPU quota.

| Case | 128 first (ms) | 256 first (ms) | Change |
|---|---:|---:|---:|
| uniform-1000 | 9.203 | 9.229 | +0.29% |
| uniform-8192 | 27.197 | 27.230 | +0.12% |
| uniform-100000 | 182.595 | 482.798 | +164.41% |
| gradient-1000 | 9.387 | 9.417 | +0.32% |
| gradient-8192 | 29.683 | 29.678 | -0.02% |
| gradient-100000 | 185.209 | 478.424 | +158.32% |
| cluster5-1000 | 14.447 | 14.414 | -0.22% |
| cluster5-8192 | 44.058 | 44.107 | +0.11% |
| cluster5-100000 | 265.028 | 463.406 | +74.85% |
| cluster10-1000 | 14.239 | 14.254 | +0.11% |
| cluster10-8192 | 41.516 | 41.554 | +0.09% |
| cluster10-100000 | 889.391 | 3352.098 | +276.90% |
| sphere-1000 | 50.743 | 50.814 | +0.14% |
| sphere-8192 | 1642.575 | 1183.600 | -27.94% |
| lattice-64 | 1824.458 | 1789.577 | -1.91% |
| lattice-512 | 2179.860 | 2147.708 | -1.47% |
| lattice-4096 | 2540.541 | 2415.548 | -4.92% |
| duplicates-1000 | 8.476 | 8.475 | -0.02% |
| duplicates-8192 | 22.643 | 22.668 | +0.11% |
| near-plane-1000 | 30.121 | 20.077 | -33.35% |
| fitted-hand-256 | 962.305 | 932.572 | -3.09% |
| scale-1e-10 | 27.206 | 27.223 | +0.06% |
| scale-1e10 | 27.204 | 27.217 | +0.05% |
| scan-uniform | 107.040 | 91.257 | -14.75% |
| scan-graded | 109.385 | 92.538 | -15.40% |

The 100K rows are rejected regressions, not a proposed global default. Source
inspection explains why: the existing large-cloud hybrid constructor uses the
128-slot stage for its FP32 candidate pass. Removing that stage bypasses the fast
pass even though the hybrid branch still reports `fast_attempted=1` and
`fast_accepted=1`. The original raw scope string described only FP64 schedule
changes; the compact report corrects that interpretation explicitly. Preserve
the 128-slot fast pass on large clouds. None of the smaller tested cases has a
paired 95% confidence interval wholly above the +2% regression threshold.

There are no additional device arrays in this experiment. Only the base schedule
changes; normalized construction and symbolic recovery retain their schedules.
This table is not a complete VTM result or a weighted-diagram qualification.
Any adoption needs changed-input/prefix, memory, complete time-to-quality and
public power/regular-diagram gates (#270), as well as preserving large-cloud paths.

## Independent validation

Small and degenerate controls use exhaustive CGAL exact orientation, all-point
empty-sphere, face pairing and hull-support checks, allowing valid cospherical
alternatives. The initial 100K attempt was stopped during this quadratic offline
check (roughly 67 billion sphere tests); its partial files and interruption record
are retained. Its timings are not part of the table.

The four large controls instead require exact connectivity equality to a fresh,
validated CGAL Delaunay triangulation on the same stored coordinates. Every
candidate tet also passes exact orientation, face pairing and hull-support checks.
This stricter mode rejects nonmatching valid cospherical alternatives rather than
approving them without a proof; exhaustive mode remains available for those cases.
The new checker flag is `--cgal-connectivity`. Fourteen native contracts cover
valid random geometry, reversed orientation, missing/duplicated cells, an omitted
interior site and two distinct valid cube triangulations. Both cube alternatives
pass exhaustive checking; at least one is rejected by strict connectivity matching.
The mode never counts a rejected mesh as a successful speedup.

The current checker is compiled with GCC 11.4 / CGAL 6.2.1. Source/binary hashes,
commands, every sample, diagnostics, proofs, paired intervals and retained failures
are in [experiments/capacity-schedule](../experiments/capacity-schedule/).
The original small controls use source `770b57c`; resumed controls use `6d4ec96`.
Raw meshes, binaries and logs remain outside ordinary Git.
