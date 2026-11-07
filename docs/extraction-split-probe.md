# Full-classification split: rejected performance probe

Separating physical tetrahedron geometry from surface queries did not give a
meaningful complete-solve improvement in the tested full extractor. The private
prototype is **not adopted**. The selective classifier remains unchanged. A
subsequent [cooperative query probe](cooperative-surface-probe.md) measures a
staged retained classifier on final snapshots; its complete-solve and failure
validation remain outstanding under issue #159.

The original `classify` kernel computes physical centers, radii, volume and
quality, including exact integer fallback for uncertain geometry, then queries
the surface. The experiment emits two ordered Warp launches using the existing
output arrays: prepare current geometry, then load those centers/radii and query
the surface. Every call refreshes the values. No cross-call classification or
proof is cached, and no additional arrays are allocated. Predicate arithmetic,
thresholds and query behavior are copied from runtime `9228a3e`.

| Complete fresh torus solve | Original | Split | Difference |
|---|---:|---:|---:|
| 1,000 sites; five timed replays | 822.454 ms | 820.580 ms | −0.23% |
| 4,096 sites; three timed replays | 7,065.010 ms | 7,066.681 ms | +0.02% |

These differences are too small to establish a useful gain. All initial and timed
CPU quota windows are unthrottled. Both cases preserve points and roles bitwise,
full/material tetrahedron sets and oriented boundary triangles exactly, and all
recorded iteration/quality work. Each control also matches its independently
validated [#158 mesh](shared-seed-geometry.md) exactly. Setup and first execution
are in the raw records; prototype kernels were preloaded outside the benchmark's
setup timer, so there is no setup-speed comparison.

The compiler properties do improve for the query kernel, but this does not
translate to the measured complete solves:

| Kernel; block size 32 | Registers/thread | Local bytes/thread |
|---|---:|---:|
| Original `classify` | 255 | 33,800 |
| Split `prepare_geometry` | 255 | 32,584 |
| Split `query_surface` | 94 | 1,136 |

These are CUDA kernel properties, not measured occupancy or memory traffic.
The experiment does not establish register pressure as the original bottleneck.
The additional launch and stored/reloaded geometry are included in replay.

All 40 existing extraction, wide-center and selective tests pass with full
classification intercepted by the prototype, including the independent physical
geometry comparisons. Both split kernels were preloaded at block sizes 32 and
256. This is exploratory validation; no memory/lifetime or native-trace adoption
audit was run for a rejected implementation.

The [artifact manifest](../results/validation/vtm/extraction-split-probe/manifest.json)
covers source, exact commands, raw logs/timings, four saved meshes, compiler
properties and equality checks. The
[private launcher](../results/validation/vtm/extraction-split-probe/run-extraction-split.py)
intercepts only `_extraction.classify`; the production package, standalone
power-diagram/Delaunay paths, defaults and README are unchanged. Broader
reconstruction scaling is being investigated in #161. CPU outperformance remains
unmet (#40).
