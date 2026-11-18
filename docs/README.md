# Documentation guide

Start with the [consolidated public API reference](api.md). It covers all names
exported by `warpower`, their defaults, input/output contracts, errors, capture
rules and numerical limits. The [repository README](../README.md) has installation
and a minimal example.

## User guides

| Topic | Guide |
|---|---|
| Device buffers, capacities, streams and CUDA capture | [Graph execution](graphs.md) |
| Certified unweighted Delaunay construction | [Delaunay recovery](delaunay-recovery.md) |
| Triangle boundaries, refits and queries | [Surface domains](domains.md) |
| Optional FP32 surface queries and refit limitations | [Native surface queries](native-surface-queries.md) |
| Sampled/automatic sizing and initial interior points | [Sizing and sampling](sizing.md) |
| Boundary fitting and feature roles | [Boundary workspace](boundary.md) |
| Uniform optimization with fixed hull vertices | [ODT workspace](odt.md) |
| Material selection, topology checks and measurements | [Extraction](extraction.md) |
| Complete variational fitting, growth and quality schedule | [Mesher workspace](mesher.md) |
| Reproducible inputs, mesh export and independent checks | [Examples](examples.md) |

Import supported APIs from `warpower`. Pages describing private VTM cores,
fragment reuse, selective updates or experimental kernels explain implementation
and research; they do not make those internal modules a supported public API.

## Benchmarks and correctness

[Benchmark reproduction](benchmarks.md) defines the measured outputs, timing,
allocator policy, CPU-quota checks and regression requirements.
[Numerical audits](numerical-audit.md) distinguish floating-point row checks
from exact rational facet checks and explain what remains unaudited.

The original [74-case L40 comparison](../results/summary/report.md) is a historical
measurement, not a rerun of the current tree. Subsequent evidence includes:

| Record | Scope |
|---|---|
| [Claim audit](../experiments/benchmark-claims-audit/README.md) | Corrects output/implementation comparisons and records the original seven wrong neighbor pairs |
| [Robust neighbor recovery](../experiments/robust-neighbor-recovery/README.md) | Numerical recovery and targeted correctness fixes |
| [Retry escalation](../experiments/retry-escalation/README.md) | Independent numerical, clipping-storage and queue requirements |
| [Cooperative capacity recovery](../experiments/cell-capacity-recovery/README.md) | Latest merged 74-case regression gate, live Bonsai comparison, exact facet audits and scoped sanitizer results |
| [Large native RadFoam comparison](../experiments/radfoam-fresh/README.md) | Fresh construction at larger scene sizes, with bounded/unbounded output qualifications |
| [Geometry updates](../experiments/geometry-updates/README.md) | Captured RadFoam/VTM transitions and update-method baselines |

Measurements apply to the revisions, inputs, output contracts and instrumentation
listed in each report. Test totals are snapshots at those revisions, not the
current suite size. Passing status flags, matching graphs and audited disagreement
sets do not establish exactness of every unexamined edge.
The broader [geometry qualification issue](https://github.com/alecjacobson/warpower/issues/300)
remains open, including recorded broader captured-workspace sanitizer findings.

## Implementation and research records

[Construction architecture](architecture.md) describes the current composition.
[VTM development](vtm.md) distinguishes implemented components from the remaining
comparable-quality CPU performance goal.
[VTM validation](vtm-validation.md), [tetrahedron validation](tetrahedra-validation.md),
[graph validation](graph-validation.md) and [CUDA validation](cuda-validation.md)
retain milestone evidence and instrumentation scope.

Most other pages in this directory are focused algorithm studies or experiment
records linked from those guides. Their tables, source hashes and unsuccessful
prototypes remain historical evidence. Read their revision and scope before
treating a statement about speed, default behavior or test coverage as current.
[Artifact handling](experiment-artifacts.md) describes retained raw evidence.
