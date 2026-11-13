# Cooperative capacity recovery

Large FP32 clipping retries now use 32 cooperating threads per cell, with 128,
256 or 512 slots. Traversal queues have a separate per-site requirement from
clipping storage. A queue-only retry retains the clipping limit and precision;
if clipping subsequently exhausts its limit, it can use spare allocated slots
without restarting the cell. Plane and vertex storage still share one capacity
class. No public tuning parameter is added.

Ordinary calls and captured replay share these policies. A larger physical queue
can serve a mixed batch while each cell retains its own logical limits. This
preserves GPU overlap between cells with different failure reasons. Numerical
checks still request FP64 and retain the existing terminal-error and consistency
repair behavior. FP32 cooperative clipping checks ambiguity conservatively and
requests FP64 directly; this is not an exactness certificate.

The tested library is `3f72982`, compared with merged main `8118e56`. The
earlier [precision-versus-storage separation](../retry-escalation/README.md)
is already present in both. The diagnostic and test-fixture fixes through `a377452` leave all 88
library source hashes unchanged. See [API](../../docs/api.md) and
[architecture](../../docs/architecture.md) for retry contracts and the untimed
per-stage diagnostic.

## Performance

L40, three warmups and twenty alternating paired synchronized wall-clock calls;
resident identical inputs, construction allocations/dispatch/recovery included.
Setup and JIT are excluded. The allocator release threshold is 32 GiB. The container has a
four-core CPU quota; accepted measured windows have no CPU throttling. GPU
process checks at timing-window endpoints found no competing process.

| Input | Main (ms) | Updated (ms) | Paired change, 95% CI |
|---|---:|---:|---:|
| live-bonsai-9216-voronoi | 80.762 | 37.686 | [-54.16%, -52.36%] |
| uniform-100000-voronoi | 12.648 | 12.723 | [-0.04%, +1.23%] |
| cluster5-100000-power | 22.500 | 19.388 | [-14.90%, -12.72%] |
| bonsai-1968522-voronoi | 215.057 | 211.197 | [-1.99%, -1.60%] |

All 74 historical inputs pass the existing gate: reject a paired 95% Student-t
interval wholly above +2%. The geometric-mean speed ratio is **1.0160×**.
Small positive overheads remain visible; passing this gate does not establish
zero overhead or an upper bound of 2% on every case. The complete per-case table,
raw timing samples and hashes are in [validation.json](validation.json).

The 100K uniform power case has a +2.28% mean difference with a broad interval
[-1.73%, +6.19%], dominated by one 20.03 ms candidate call. Its medians are
14.163 → 14.128 ms. The slow sample is retained. This passes the declared gate
but does not rule out a 2% overhead for that case.

The gain comes primarily from substituting the cooperative FP32 kernel, including
its clipping representation and numerical checks. The matched ablation compares the cooperative-only prototype with the final
independent queue policy; per-repetition results are retained in validation.json. An intermediate queue dispatcher took 44.1 ms because
it split difficult cells into sequential launch batches. Recombining eligible
cells recovered the overlap. Those failed screens are retained in the evidence.

| Input | Cooperative-only (ms) | Updated (ms) | Paired change, 95% CI |
|---|---:|---:|---:|
| bonsai-1968522-voronoi | 211.497 | 210.341 | [-0.72%, -0.38%] |
| live-bonsai-9216-voronoi | 38.049 | 37.690 | [-1.74%, -0.14%] |
| uniform-100000-voronoi | 12.657 | 12.527 | [-1.81%, -0.20%] |

Fresh calls in separate backend processes, three warmups and twenty samples:

| Backend | Live Bonsai mean (ms) | Median (ms) | Failed cells | Output |
|---|---:|---:|---:|---|
| warpower | 37.790 | 37.841 | 0 | bounded neighbors |
| paragram | 42.180 | 42.033 | 143 | bounded neighbors |
| radfoam | 21.320 | 19.437 | not reported | unbounded neighbors + tetrahedra |

Native RadFoam remains faster on this small input. It is a CUDA implementation
and returns an unbounded triangulation in internal sorted order, including
tetrahedra and an original-index permutation. Warp/Paragram return bounded
neighbor CSR in original order. Paragram's failed rows make its result incomplete.
These are qualified construction timings, not identical-output speed claims.
All samples are retained: the final native run contains two slow samples around
37–39 ms, which raise its mean while its median remains about 19.4 ms. No CPU
quota throttling or competing GPU process at the observation endpoints explains
those outliers; their cause is unresolved. The earlier native run is retained
as a separate measurement, not substituted for the final run.

Untimed observations show 130 vertex-capacity failures, one queue failure and
two uncertain predicates after the initial 9,216-cell pass. The cooperative
FP32/128 batch processes all 131 capacity failures together; 18 need 256 slots,
and only site 2021 needs 512. Per-stage precision recovery counts are recorded
by the diagnostic; they are not inferred from scheduled launch dimensions. In separate synchronized per-kernel profiling, FP32 128/256/512 stage durations are recorded in profile-final.json;
the last stage has one active cell and 32 threads.
These kernel timings are not the primary wall-clock measurements.

| Precision | Threads/cell | Clipping slots | Queue slots | Active cells | GPU kernel (ms) | Outcomes |
|---|---:|---:|---:|---:|---:|---|
| fp32 | 1 | 64 | 96 | 9216 | 9.307 | 9083 success, 130 vertex_capacity, 1 queue_capacity, 2 uncertain_predicate |
| fp64 | 32 | 128 | 128 | 2 | 1.331 | 2 success |
| fp32 | 32 | 128 | 256 | 131 | 2.882 | 110 success, 18 vertex_capacity, 3 uncertain_predicate |
| fp64 | 32 | 128 | 128 | 3 | 3.102 | 3 success |
| fp32 | 32 | 256 | 256 | 18 | 5.544 | 17 success, 1 vertex_capacity |
| fp64 | 32 | 128 | 128 | 0 | 0.013 | — |
| fp32 | 32 | 512 | 256 | 1 | 12.214 | 1 success |
| fp64 | 32 | 128 | 128 | 0 | 0.012 | — |


## Captured geometry and complete VTM solves

| Mode | Sites | Distribution | Weighted | Output | Main (ms) | Updated (ms) | 95% interval | Workspace MiB, before → after |
|---|---:|---|---|---|---:|---:|---:|---:|
| eager | 100,000 | uniform | False | neighbors | 12.682 | 12.573 | [-1.72%, +0.03%] | — |
| eager | 100,000 | cluster5 | False | neighbors | 20.660 | 18.020 | [-14.59%, -10.83%] | — |
| eager | 1,000,000 | uniform | False | neighbors | 104.538 | 104.826 | [-0.05%, +0.61%] | — |
| eager | 1,000,000 | uniform | True | neighbors | 110.535 | 109.466 | [-1.41%, -0.51%] | — |
| graph | 100,000 | uniform | False | neighbors | 11.079 | 11.075 | [-0.78%, +0.72%] | 796.62 → 829.05 |
| graph | 100,000 | uniform | True | neighbors | 12.427 | 12.547 | [+0.08%, +1.87%] | 796.17 → 828.67 |
| graph | 100,000 | uniform | False | tetrahedra | 473.247 | 473.222 | [-0.03%, +0.02%] | 402.95 → 403.02 |
| graph | 100,000 | uniform | True | tetrahedra | 477.660 | 477.642 | [-0.02%, +0.01%] | 402.57 → 402.64 |
| graph | 100,000 | uniform | False | neighbors + tetrahedra | 487.327 | 485.965 | [-0.35%, -0.21%] | 974.18 → 1006.67 |
| graph | 100,000 | uniform | True | neighbors + tetrahedra | 492.336 | 490.966 | [-0.30%, -0.25%] | 973.79 → 1006.29 |

All neighbor and tetrahedron sets are unchanged. Retry state grows from eight
to twelve bytes per site. Cooperative FP32 scratch keeps homogeneous and
Cartesian coordinates, increasing the captured 100K adjacency workspace by
about 32 MiB. Setup/allocation are excluded from captured replay timing.

Complete 4K-site VTM solves include fresh sampling, fitting, growth, quality,
extraction and diagnostics, with the existing hull-proof cache enabled in both
versions. Three warmups and ten alternating pairs; setup/JIT/downloads excluded.
Positions and vertex roles are bitwise identical; tetrahedron sets are identical.
The isolated package namespaces can hit different JIT caches. Setup durations
are not matched cold-compilation measurements. The comparison establishes
warmed solve performance only; raw setup durations are retained in the evidence.

| Case | Main median (ms) | Updated median (ms) | Paired change, 95% CI |
|---|---:|---:|---:|
| torus | 921.809 | 917.881 | [-0.49%, -0.34%] |
| graded-torus | 824.058 | 821.775 | [-0.40%, -0.23%] |

## Correctness and sanitizer scope

All **210 targeted tests pass**, including 25 new cases covering both precisions,
weighted high-degree cells with independent facet witnesses, actual cooperative
default dispatch, queue exhaustion and terminal limits, mixed scratch rows,
tiny initial queues, and conditional captured dispatch. Negative controls show
the observed prototype bugs fail their regression tests. The mixed-batch test
asserts the FP32/512 path actually ran: a final-graph-only test would miss a skipped
stage rescued by later consistency repair.

A weighted five-cluster regression also verifies an exact rational infeasibility
certificate using Python Fraction arithmetic. The certificate proves that the
disputed pair cannot share a facet. An intermediate FP32 cooperative formulation
created that facet temporarily, requiring two expensive consistency repairs.
Sharing the ordinary FP32 minor expansion and Cartesian side expression removes
that repair, while preserving the complete final graph. The previous formulation
fails the new path assertion; the oracle was checked on ten analytic controls.

Exact rational facet audits cover every CGAL disagreement, every baseline change,
and all fourteen directions of the original seven wrong pairs. All five scene
graphs are unchanged from main. Shared edges outside the audited set are not
individually certified.

| Scene | Directions audited | Main errors | Updated errors |
|---|---:|---:|---:|
| live-bonsai | 172 | 0 | 0 |
| bonsai | 368 | 0 | 0 |
| garden | 1182 | 0 | 0 |
| bicycle | 1566 | 0 | 0 |
| playroom | 474 | 0 | 0 |

**memcheck: 84 tests**, **initcheck: 84 tests**, **racecheck: 41 tests**, **synccheck: 41 tests** pass.
Memcheck/initcheck cover all 84 recovery tests; racecheck/synccheck cover the focused
retry/cooperative subset, including captured individual retry kernels.

Initcheck caught a read of unused retry-job entries in the new untimed diagnostic.
The observer now copies only populated entries and skips empty batches. All 25
recovery tests were rerun after this change; all recorded stage outcomes and
per-site requirements are unchanged after accounting for atomic collection order.
The 84-case memcheck run covers the unchanged geometry library; all seven affected
observer/captured tests were additionally rerun under memcheck after the fixes.
The full 84-case initcheck run passes after the fix. A second finding was in the mixed-batch test: its simulated queue failure
left retry-seed slots uninitialized. The fixture now supplies a fully initialized
empty seed row, matching the production failure contract. Captured tests also now copy only populated adjacency ranges and retry rows.
All failing logs are retained. These fixes change no measured library source.

Broader captured-workspace sanitizer findings reproduced on previous baseline
and candidate remain unresolved in [issue #300](https://github.com/alecjacobson/warpower/issues/300):
synccheck divergence in unchanged `block_bounds`, and cross-kernel shared-memory
racecheck diagnostics involving cooperative construction and `collect_failures`.
They are not declared false positives or suppressed. This report does not claim
all captured-workspace sanitizer configurations are clean.

## Evidence

[Verified evidence archive](https://github.com/alecjacobson/warpower/releases/download/evidence-geometry-updates-4ceb748/capacity-recovery-20261113.tar.gz); SHA-256 `24df0ce6a2104612efbae81f0fb76a4876464936e158e4643d70ce44ecab092e`.

The archive retains raw repetitions, source/input hashes, implementation and
ablation patches, exact queries and controls, diagnostic/profile records, tests,
sanitizer logs and the failed development screens. The initial exact-audit runner
stopped on an import-path error before auditing geometry; that log is retained,
and the corrected runner completes all five scenes. [Manifest](manifest.json)
and [download verification](remote-verification.json) record artifact integrity.
RadFoam/VoroTracing fitting runs remain deferred.
