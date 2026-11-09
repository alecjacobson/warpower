# Local rollback at invalid material edge stars: initial experiment

The material-selection observer found three nonmanifold edge stars during early
uniform Armadillo selective trials even though the underlying Delaunay hull was
valid. This benchmark-only experiment freezes moved endpoints of the current bad
edges, or the vertices of tetrahedra incident to those endpoints, and reruns
the ordinary construction and acceptance checks. Production never imports the module.

Eligibility requires a fresh recovered-selection rejection: selection error 16,
reason 3, original extraction error zero, invalid extraction diagnostic 2, and
successful current Delaunay and boundary evaluations. Sticky prior errors cannot
trigger it. Only active, moved, selection-eligible noncorner sites can be blocked.
The per-step device retry budget is one. No new movable site, an exhausted budget,
or another failed guard restores ordinary global search and eventual rollback.
No material labels or acceptance criteria change. A local retry invalidates the
quality-only step-history prediction.

## Complete Armadillo solves

NVIDIA L40 / Warp 1.19.0; median five additional complete captured solves on 8K
sites, native surface queries, mixed nearest-site queries, trusted input, 120°
feature angle. Setup/export/independent checking is excluded; every accepted
window has full CPU quota. This source is `29991f5`, before the separate production
clipping-storage change. Counts, quality, fit and all timing samples are retained.

| Case | Policy | Time (ms) | Change | Local retries | Blocked site events | Accepted local trials |
|---|---|---:|---:|---:|---:|---:|
| uniform | baseline | 2318.663 | +0.00% | 0 | 0 | 0 |
| uniform | endpoints | 2208.887 | -4.73% | 6 | 24 | 1 |
| uniform | star | 2215.869 | -4.43% | 6 | 144 | 1 |
| graded | baseline | 2303.068 | +0.00% | 0 | 0 | 0 |
| graded | endpoints | 2308.675 | +0.24% | 0 | 0 | 0 |
| graded | star | 2306.890 | +0.17% | 0 | 0 | 0 |

The uniform case gains one accepted selective step (3 → 4), with six local
retries and one acceptance immediately following a local retry. The endpoint
policy blocks 24 site events; the union of endpoint vertex stars blocks 144.
The `star` option includes all tetrahedra touching a flagged endpoint; it does
not restrict the neighborhood to tetrahedra containing both ends of a bad edge. The graded
case never enters the new path and remains bitwise identical to its baseline.
Each variant adds 137,250 active pool bytes in this benchmark. Setup takes about
4.93–5.51 s versus 3.33 s, including an extra experimental graph recapture.

This is a quality tradeoff, not strict improvement in every metric:

| Uniform policy | Minimum radius ratio | Mean radius ratio | 1st percentile radius ratio | Surface RMS (% bbox diagonal) |
|---|---:|---:|---:|---:|
| baseline | 0.0310191823605 | 0.874565024348 | 0.456574758755 | 0.147990383963 |
| endpoints | 0.0310047215082 | 0.874619693022 | 0.457487563077 | 0.147975879834 |
| star | 0.0310047215082 | 0.874618876488 | 0.457487563077 | 0.147982852842 |

The worst radius ratio decreases by about 0.047% relative. Mean and lower-tail
quality and sampled surface RMS improve slightly. The uniform full/material mesh
loses two tetrahedra and changes internal connectivity. Roles and oriented
boundary connectivity match; sites move by up to 0.00128626823425293. Endpoint and
star variants also differ slightly from each other. Full numerical quality,
angle extrema, volume error and fit are available in the independent report.

All six outputs pass independent original-coordinate exact CGAL full-hull checks,
material face/edge incidence, boundary vertex links, exact boundary intersection
checks and exact VTU round trips. Surface RMS uses quadrature and is not a
Hausdorff certificate. Thirteen focused contracts cover rejection freshness,
retry budgets, pinned/masked sites, changed prefixes and actual tetrahedra meeting
only at an edge. All 13 pass ordinary execution, padded memcheck and full
allocation-order instrumentation. These focused tests are not a native audit or
memory qualification of the complete experimental pipeline.

## Remaining qualification

This is an initial benchmark result under #287/#258. It needs broader smooth,
feature, disconnected, cavity and graded examples; changing-input captured
producer/consumer checks; full-path memory/capture qualification; and a decision
about the measured quality tradeoff before any production policy is adopted.
Public power diagrams, Delaunay and VTM defaults are unchanged. The approximately
4.7% observed uniform complete-solve improvement is not a CPU speedup claim.

Run `python -m benchmarks.selective_topology_rollback --neighborhood endpoints
--budget 1 --report trial.json` followed by the usual `examples.vtm_gallery`
arguments. Use `baseline` or `star` for the other controls. The parser fix disables
abbreviated option matching so the forwarded `--n` reaches the gallery parser.
The failed earlier invocation and the cancelled prequalification queue are
retained with all raw results in the [evidence index](../experiments/selective-topology-rollback/).
