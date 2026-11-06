# Parallel global sphere certification

The existing local certificate and original-site hierarchy are unchanged. When
local coverage cannot be proved, global checks on buffers of 128–512 sites
distribute each tetrahedron's site queries over 32 workers. The
upper limit follows the fixed-buffer grid benchmark: direct point scans improve
the 512-site case but regress the 1,000-site case, which retains the tree.
Larger and smaller buffers retain hierarchical traversal. No new arrays or
per-replay allocations are required.

Each worker scans a disjoint residue class of original site indices. It skips
canonical duplicates and defining vertices, applies the same conservative
circumsphere box and cached polynomial interval to the point, and evaluates the
same exact symbolic sphere predicate for every unresolved sign. Bounds skip only
strictly outside points. Invalid predicates and nonnegative symbolic signs set
the same shared error bits with atomic OR. Therefore the union of workers tests
exactly the same lower-facet condition as the original traversal, independent of
point order. Exact orientations, face incidence, support and site coverage are
still checked by the existing certificate.

For this path, `statistics[0]` counts unresolved symbolic predicate queries after
point bounds, and `statistics[1]` is zero because no tree nodes are traversed.
The legacy tree path counts visited leaf candidates before its final point
filter, so these branch-specific diagnostic counters should not be interpreted
as identical units of work. Output connectivity and error semantics are the
same; these counters are diagnostics rather than correctness evidence.

The final implementation passes **438 tests**, including 28 differential tests
comparing old and new global checks on identical inputs. Cases include very
small/large float32 scales, subnormal tetrahedron heights, duplicate queries,
interior/exterior witnesses, and sizes around dispatch and warp boundaries.
Thirty targeted certificate tests pass padded memcheck with zero errors; the
captured changing-input loop passes full allocation/lifetime tracking. Those
checks use the documented diagnostic 32 KiB stack; lifetime tracking uses
synchronous allocations. They do not change runtime defaults.

A native trace of four warmed producer/Delaunay/consumer replays records 514 GPU
kernels, four launches, one final synchronization, and 21 device-to-device graph
copy nodes. It switches between local certification, global fallback, invalid
input and restored input, with no host or allocation/free graph nodes.

## Performance and threshold

NVIDIA L40, Warp 1.19.0. Five samples per standalone complete operation, three
full torus solves and two cached refined-hand replays; setup/JIT/readback are
excluded, all recovery strategies are included, and all measured CPU quota
windows are unthrottled.

| Complete operation | Before | After |
|---|---:|---:|
| Random 1,000 sites | 9.180 ms | 9.164 ms |
| Random 10,000 sites | 40.726 ms | 40.704 ms |
| Random 100,000 sites | 183.055 ms | 182.371 ms |
| Cube | 0.529 ms | 0.526 ms |
| Grid, 64 sites | 2.207 ms | 2.206 ms |
| Grid, 512 sites | 4.293 ms | 3.957 ms |
| Grid, 1,000 sites | 5.448 ms | 5.401 ms |
| Corner cloud, 200 sites | 6.795 ms | 6.082 ms |
| Torus boundary, 128 sites | 228.183 ms | 227.948 ms |
| Fitted hand, 256 sites | 13.415 ms | 13.431 ms |
| Refined hand, 384 sites, cached | 6,490.757 ms | 6,473.251 ms |
| Full torus VTM, 1,000 sites | 3,204.664 ms | 3,204.364 ms |

The corner cloud improves 1.12× and the 512-site grid 1.08×; other whole-operation
times remain within about 0.9% in this matrix. First recovery of the refined
hand improves from 38,778.938 to 37,730.508 ms. Its cached strategy already has a
large enough box and mostly avoids incomplete-mesh global checks, so repeat
runs barely change. Full torus point/role/input arrays and oriented
full/material/boundary connectivity, and the refined-hand mesh, are identical.
These are modest complete-operation gains, not a large CPU speedup.

Matched fixed-buffer comparisons isolate the global kernels and explain the
upper threshold. They alternate kernel order on identical positive-cell buffers:

| Sites / distribution | Tree | Point workers |
|---|---:|---:|
| Random 128 | 0.283 ms | 0.058 ms |
| Random 512 | 0.472 ms | 0.278 ms |
| Random 1,024 | 0.728 ms | 0.550 ms |
| Random 2,048 | 1.267 ms | 1.121 ms |
| Grid 216 | 0.482 ms | 0.217 ms |
| Grid 512 | 0.594 ms | 0.522 ms |
| Grid 1,000 | 0.590 ms | 1.073 ms |

The 1,000-site lattice regresses in the direct scan, so the final dispatch keeps
the tree above 512 sites. The initial 128–1,024-range measurements are retained;
changed random-1k/full-torus paths were remeasured with the final threshold,
and the complete 512/1,000-site lattice comparisons use that final source.
The remaining matrix cases choose the same paths at either threshold.
The fixed-buffer lattice candidates deliberately permit alternate symbolic tie
choices: both kernels report the same rejection. That benchmark measures the
sphere condition alone, not construction, completeness or mesh acceptance.

```bash
python -m benchmarks.global_certificate --repeats 5 --output kernels.json
python -m benchmarks.delaunay_recovery --case grid --grid-side 8 --repeats 5 --output grid.json
```

[Complete timings and hashes](../results/validation/vtm/parallel-performance.json),
[matched kernels](../results/validation/vtm/parallel-kernel-comparison.json),
[438-test record](../results/validation/vtm/parallel-512-all-tests.txt),
[memcheck](../results/validation/vtm/parallel-memcheck.txt),
[lifetime check](../results/validation/vtm/parallel-lifetime.txt),
[native audit](../results/validation/vtm/parallel-device-audit.json),
[output equivalence](../results/validation/vtm/parallel-output-equivalence.json),
and [source isolation](../results/validation/vtm/parallel-isolation.json)
retain the evidence. Existing certificate code is unchanged apart from the
global dispatch and added kernel; all other runtime modules and README are
unchanged. [Issue #105](https://github.com/alecjacobson/warpower/issues/105)
tracks this change. Fragment completion remains separate work in
[issue #104](https://github.com/alecjacobson/warpower/issues/104), and the full
VTM CPU performance target remains open.
