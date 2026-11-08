# Boundary global reductions: unadopted probe

The private experiment in [issue #223](https://github.com/alecjacobson/warpower/issues/223)
reduces shared boundary diagnostics within the existing 32-thread blocks. It
provides no measured complete-solve gain and is not used in production.

Each sample retains the original nearest-site query, contribution formulas and
per-owner mass/moment scatter. Pure-Warp tile sums combine energy and total
weight; a tile maximum combines distance. Each block then updates the three
global outputs. Early-return paths supply neutral values so participating lanes
reach the collective operations, including partially active blocks. No arrays,
kernel launches or replay transfers are added. Summation order changes.

Ninety direct comparisons cover dimensions one/two, scales 1e-30/1/1e30,
partial blocks, zero weights, invalid inputs, empty active prefixes and existing
errors. Owner IDs, maximum-distance bits, error flags and untouched components
match. Sum differences stay within the explicitly recorded conservative test
envelope; this is **not a general floating-point error proof**.

The complete fresh-solve pilot uses main
`971e6f2e5aa641dce95c1b4ad2124e7454f11a6a`, Warp 1.19.0 and the NVIDIA L40.
Each job records three additional captured replays. All initial/timed quota
windows are clean; independent CPU validation runs after GPU timing. Allocation,
compilation, transfer and export are outside replay timing.

| Complete solve | Original ms | Grouped ms | Change |
|---|---:|---:|---:|
| Torus, 4,096 sites | 1,508.657 | 1,511.657 | +0.20% |
| Hand seed7, 1,000 sites | 1,290.939 | 1,297.647 | +0.52% |

All four saved outputs pass independent exact full-hull, orientation and
empty-sphere checks, material quality/topology, boundary links, zero boundary
self-intersections and export round trips. Both pairs match canonical full and
material tetrahedra, oriented boundary faces, roles, counts and recorded work.
Torus positions are bitwise equal; hand coordinates differ by at most
3.469446951953614e-18. These small timing differences do not establish either
a useful gain or a statistically resolved slowdown.

The [manifest](../results/validation/vtm/boundary-global-reduce/manifest.json)
preserves all prototype sources, direct checks, clean timings and independently
checked meshes. Since this variant is rejected, it has no production memory or
native capture qualification. Runtime code, independently callable diagram and
Delaunay routines, defaults and the README flagship remain unchanged.
