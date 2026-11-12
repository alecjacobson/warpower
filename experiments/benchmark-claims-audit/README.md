# Geometry benchmark claim audit — 2026-11-12

The historical Warpower-versus-Paragram timings remain valid **for their measured
box-clipped neighbor-construction workloads**. They do not establish a universal
ranking, equally exact output, or the speed of our separate Delaunay tetrahedron
constructor. New matched-input measurements confirm both a small-input loss and
a large-input win. [Issue #300](https://github.com/alecjacobson/warpower/issues/300)
tracks the corrections, missing comparisons, correctness regressions and
optimization acceptance requirements. RadFoam and VoroTracing fitting runs are
**deferred until the geometry work is settled**.

## What the evidence supports

| Evidence | Retained result | Qualification or correction |
|---|---|---|
| Original L40 matrix | Faster than Paragram in 62/74 cases; 1.220× geometric mean | Fresh box-clipped Voronoi/power neighbor CSR, not tetrahedron construction or incremental updates. Twelve cases favor Paragram. |
| Original 100K five-cluster Voronoi case | 30.40 vs 65.99 ms, 2.17× | Floating-point audit of all 359 disputed rows; shared rows were not audited. This is not an exact proof of the entire graph. |
| Later four-checkpoint comparison | 1.30–1.87× over Paragram | Exact facet audit found seven incorrect Warpower undirected pairs: four spurious and three missing. Timings stand; an inference of equally exact output is false. |
| Live Bonsai update corpus, 9,216–17,274 sites | 286.970 ms cell rebuild; 22.890 native fresh; 6.725 native incremental; 19.969 VTM fragment reuse | Median of per-transition medians across 11 transitions. Native wins nine. The cell rebuild uses `DelaunayWorkspace(cache_strategy=False)`, not the neighbor builder. Forced-fresh VTM wall construction was not separately measured, so this is not a best-Warp-constructor result. |
| VTM hull-proof cache | Complete 4K torus solves 987.230→914.652 ms; graded torus 881.099→824.910 ms | Retain 7.35% and 6.38% for those three-repeat complete solves, with identical positions and tetrahedra/roles. It does not meaningfully improve the measured Bonsai updates. |
| Existing full-budget RadFoam fits | 89.234→54.804 min | Endpoint 32.4136→32.2511 dB: 0.1625 dB lower, one fit per backend, no measured seed variance. Sampled time-to-quality is a separate result. No new fits in this audit. |
| Existing VoroTracing fits/inference | 86.69→73.85 min; separately fitted models 182.9→200.7 camera-to-image FPS | Whole-pipeline changes and different fitted models do not isolate geometry or renderer acceleration. Same-checkpoint renderer comparisons are near parity. No new fits in this audit. |

The paper's fresh reconstruction comparison explicitly excludes incremental
reuse, uses RTX 5090/H200, and includes million-site checkpoints. Appendix D
acknowledges FP32 adjacency discrepancies. Its ranking cannot be combined with
our L40 ratios and 9–17K-site updates as a transitive claim. These are separate
workloads and accuracy contracts. [Paragram paper, §5.2–5.3 and Appendix D](https://arxiv.org/html/2605.06408v1).

## Seven silent neighbor-pair errors: targeted diagnosis

These seven undirected pairs are confirmed incorrect for the original float32
inputs and Warpower's clipping box. The audit tests whether the two cells share
a positive-area facet, using an exact rational LP and the constraints from a
validated CGAL neighbor star. It therefore distinguishes actual errors from
legitimate box-clipping differences and zero-area contacts.

| Scene | Spurious pairs | Missing pairs |
|---|---:|---:|
| Bonsai | 1 | 1 |
| Garden | 0 | 1 |
| Bicycle | 0 | 1 |
| Playroom | 3 | 0 |

A follow-up recheck reproduced all 14 directed exact-LP answers. The current
public API also reproduced all 14 incorrect directed edges, with successful
statuses at every affected endpoint. Each error occurs in both directions, so
the graph-symmetry check cannot expose it: both cells either include the same
spurious neighbor relation or omit the same real one.

Targeted reconstruction retained **every original input site**, rebuilding the
hierarchy for each scene and evaluating only the affected cells. At matched
cell/queue capacities of 512 and the same one-active-lane-per-warp dispatch:

| Diagnostic | Incorrect selected directed edges | Nonzero selected statuses |
|---|---:|---:|
| Public API | 14 / 14 | 0 |
| Targeted fresh cells, FP32 | 14 / 14 | 0 |
| Targeted fresh cells, FP64 | 0 / 14 | 0 |

This identifies **precision-sensitive construction errors that survive larger
buffers**. It does not yet isolate the unstable predicate, clipping operation or
traversal bound. The FP64 experiment fixes the selected pair decisions only;
it is not an automatic recovery mechanism, a production fix or an exact
certificate for the whole returned graph. The current kernel's success status
means its internal checks passed, not that an independent geometric certificate
was obtained. Unaffected shared edges were only sampled in the original audit,
so seven is the confirmed count, not a proved total error count.

These failures differ from the small-Bonsai capacity overflows discussed below:
they do not report a failure that the retry dispatcher can act on. Correctness
work must provide reliable detection/recovery or robust predicates, with these
pairs as regressions, before qualifying performance changes. No rendering impact
or causal connection to fitted PSNR has been measured.

The [machine-readable diagnosis](seven-pairs.json) pins all site IDs, input hashes
and outcomes. `seven-pairs-diagnosis-20261112.tar.gz` on the
[evidence prerelease](https://github.com/alecjacobson/warpower/releases/tag/evidence-geometry-updates-4ceb748)
contains the selected exact queries, prior-audit provenance, oracle source/binary,
replay scripts and selected output rows. It excludes the large original point
arrays, which remain the same public checkpoint inputs used in the comparison.
Verify against the [manifest](seven-pairs-manifest.json);
[remote verification](seven-pairs-remote-verification.json) checks a fresh download.
These are untimed correctness diagnostics, with no production algorithm changes
or fitting runs.

## New matched-input fresh-call probes

All times below are means in milliseconds on the same L40. Each row uses
identical float32 coordinates **and input order** across backends. There are
three warm-ups and five measured calls for Bonsai; the flagship uses ten
measured calls. Processes run serially. Inputs reside on the GPU; allocation,
construction and required neighbor export are included; upload, JIT compilation,
readback and correctness checks are excluded. Warp retains up to 32 GiB;
Paragram uses its normal Torch allocator. Native uses its own constructor and
allocation policy, unchanged. These are timing/status probes, not independently
certified geometric outputs.

| Input | Sites | Warpower neighbor CSR | Paragram neighbor CSR | Native RadFoam fresh |
|---|---:|---:|---:|---:|
| Live Bonsai, captured frame 3 | 9,216 | 141.241 | 42.150 | 19.579 |
| Public Bonsai checkpoint | 1,968,522 | 224.804 | 421.254 | excluded¹ |
| Five-cluster Voronoi, seed 42 | 100,000 | 29.336 | 65.310 | not measured |

¹ Native's large run measured 15,956.053 ms but its measurement window was
CPU-throttled, so it is excluded from accepted comparisons. Its raw samples and
quota deltas are retained. All seven accepted series passed the quota check.
Two initial native setup attempts failed before collecting timing samples;
`harness-correction.json` records their exclusion and the corrected sampler
lifecycle/main guard. No incomplete sample is used in this table.

Warp/Paragram return box-clipped neighbor CSR. Native also constructs Delaunay
tetrahedra and returns unbounded neighbor CSR; no box restriction or common
exact certificate was added to the native probe. The table compares practical
fresh routes to neighbor connectivity while explicitly disclosing different
outputs. It is not an equal-output correctness comparison. Native revision:
`3e7b52cf74e37ab2ab5e695f53570f515f537e3d`; Paragram revision:
`4f36b1f2c8738912c5efc056a06023d4c1d2da07`; Warpower revision:
`51487623858210b17827e09d52e0e266fd305af7`.

The small case favors Paragram by **3.35×** and native by **7.21×** over the
neighbor builder. Thus the separate tetrahedron driver is only part of the
explanation for the slow small-Bonsai result. A subsequent profile below
attributes the small-input time primarily to capacity retries. The large checkpoint still favors Warpower over
Paragram by **1.874×**, and the flagship by **2.226×**. These ratios do not prove a
crossover rule, general speedup, or all-case regression protection.

Warp reports zero failed statuses on these probes; Paragram reports nonzero
statuses. Neither fact proves every edge correct. For example, the small graph
has 131,678 directed edges from Warp, 131,538 from Paragram and 131,850 from
native; clipping and numerical differences must be audited before interpreting
these differences. Raw statuses, counts, dispersions, memory observations,
source hashes and input hashes are in [probe-summary.json](probe-summary.json).

### Why 9,216 sites can take almost as long as 1.97 million

Both neighbor timings use the same `voronoi_diagram` driver. The smaller cloud
contains a few much more complex cells, and the retry policy handles them
expensively. A subsequent Nsight profile of one call per input, after three
warm-ups, gives this breakdown. These instrumented totals explain the earlier
clean averages; they do not replace them.

| Work | 9,216 sites (ms) | 1,968,522 sites (ms) |
|---|---:|---:|
| Initial FP32 cell construction | 8.964 | 199.329 |
| Retry and consistency-repair construction | 127.892 | 11.122 |
| Other kernels, transfers and host work | 4.296 | 16.426 |
| Profiled wall total | 141.151 | 226.878 |

In the small case, 131 cells exceed the initial 64-vertex capacity; 18 still
exceed capacity 128 after the FP32 retry. The policy then switches all unresolved
cells to FP64, including these capacity failures. The FP64 passes cost 3.647 ms
at capacity 128, 46.799 ms at 256, and **69.042 ms at 512**. The last pass has
**one active cell**, site 2021, which ends with **135 neighbors**. The kernel
assigns one active CUDA thread per retry cell; this last expensive cell therefore
has very little parallelism. The million-site checkpoint has at most 51 neighbors
per cell and never reaches capacity 512.

This identifies a capacity-recovery performance problem. More sites do not
necessarily mean more difficult individual cells, and a few expensive serial
retries can dominate. The 128-capacity FP64 pass resolves none of the 18 small-case
failures. Optimizations to capacity selection, precision selection, reuse or
parallelism still need correctness validation; removing recovery would change
the output contract. This diagnosis does not certify the graphs or establish a
general small-versus-large scaling rule.

Kernel durations come from Nsight; active counts/statuses come from a separate
untimed run with synchronous readbacks, using unchanged production kernels.
The [stage summary](neighbor-scaling.json) records both sources. Raw traces,
scripts and per-pass site IDs are in `neighbor-scaling-diagnosis-20261112.tar.gz`
on the same [evidence prerelease](https://github.com/alecjacobson/warpower/releases/tag/evidence-geometry-updates-4ceb748),
with a [manifest](neighbor-scaling-manifest.json) and
[fresh download verification](neighbor-scaling-remote-verification.json).

## Shared clipper, different construction drivers

The published neighbor API and the cell-based tetrahedron API share clipping
primitives and hierarchy machinery. They do not share the complete traversal
and recovery policy:

- `voronoi_diagram`: FP32 construction, nearest-leaf initialization, directional
  traversal bounds, selected retries/FP64 recovery, box-clipped neighbor export.
- The measured `DelaunayWorkspace(cache_strategy=False)` path: root-based
  distance-queue cell traversal with larger FP64 scratch capacities on these
  small inputs, cell-incidence dual extraction, hull handling and certification.
  Its FP32 candidate gates start at 32,768 sites.
- VTM wall/fragment construction: direct tetrahedron/frontier construction,
  reuse and repair, with validation/fallback. A forced-fresh wall baseline is
  still missing from the comparison.
- Native RadFoam: direct tetrahedron/frontier construction; fresh and incremental
  operations are separate benchmark entries.

For generic unbounded Voronoi cells, shared facets correspond to Delaunay edges.
The clipper also already retains vertex/plane incidence suitable for emitting
tetrahedral candidates. **Emitting tetrahedra does not inherently require a
separate FP64 reconstruction of every cell.** Finite clipping can omit incidence
outside the box, while ties, ambiguous cells and validation still require
explicit handling. These are reasons to specify outputs and correctness, not
an explanation that makes the current slower traversal unavoidable.

On one warmed 9,216-site transition, cell construction accounts for
282.773/286.592 ms of GPU kernel time (98.7%); validation and other kernels total
about 3.82 ms. No normalized/symbolic recovery or reallocation occurs there.
This supports optimizing construction, not removing validation. Simply lowering
the gate is not validated: [#270](https://github.com/alecjacobson/warpower/issues/270)
records an 8K Armadillo attempt that failed after repair. See the
[profile and update report](../geometry-updates/README.md).

## Retrospective verification and remaining gates

[report.json](report.json) records the retrospective checks:

- Verified SHA-256 identities of all **148** raw files in the historical
  74-case matrix; recomputed every ten-sample mean and all paired aggregates.
  Input/hardware/harness/allocator compatibility checks pass. Historical rows,
  numeric values and raw measurements are unchanged.
- Six shared construction modules are byte-identical to the historical source;
  five input/build functions are AST-identical. The API has additional optional
  tetrahedron output and lazy entrypoints. Source continuity does not establish
  a timing regression gate or prove geometric correctness.
- Recomputed the later four-scene ratios and checked the exact-audit accounting:
  Bonsai one spurious/one missing pair, Garden one missing, Bicycle one missing,
  Playroom three spurious. All Warp-versus-CGAL disagreements were examined;
  shared edges were sampled, so this is not a complete exact graph certificate.
  [Pinned checkpoint summary](https://github.com/alecjacobson/warpower/blob/e37e00f693a26bee20928ba01f64a78d7906ed8a/results/validation/radfoam/checkpoint-comparison/summary.json).
  The four-thread CGAL series was CPU-throttled; do not advertise it as a clean
  four-core reference. CGAL's constructor plus neighbor export also has a
  different output/work contract from bounded GPU adjacency.

The full 74-case matrix has **not** been rerun on the current branch. Three
probes and unchanged kernels cannot replace paired baseline/candidate testing.
Issue #300 requires preserving the original weighted/unweighted inputs,
per-case timing and memory checks, CPU-quota rejection and independent
correctness regressions. Repeatable >2% per-case slowdowns beyond observed noise
must block adoption or receive an explicit documented policy decision; aggregate
improvements must not conceal them. No automated CI timing gate is claimed.

Remaining geometry work includes a forced-fresh VTM wall baseline, candidate
emission through the optimized clipper, repair/certification costs, the seven
exact facet discrepancies, update identity changes and complete VTM solves.
Fitting integration is a deferred follow-up, not a prerequisite for publishing
this audit. No production geometry algorithm is changed by this audit.

## Evidence and reproduction

The [evidence prerelease](https://github.com/alecjacobson/warpower/releases/tag/evidence-geometry-updates-4ceb748)
contains `benchmark-claims-audit-20261112.tar.gz`. Its committed
[manifest](manifest.json) pins raw reports, failed setup reports, command logs,
probe scripts, selected source snapshots and the two Bonsai point arrays. The
[remote verification](remote-verification.json) records a fresh download check.
The original 74-case raw measurements remain pinned in Git and are not replaced
by the new probes. The paper text is not redistributed in the bundle.

```sh
python -m benchmarks.artifacts verify \
  --archive benchmark-claims-audit-20261112.tar.gz \
  --manifest experiments/benchmark-claims-audit/manifest.json
# After extracting verified evidence into a fresh persistent directory:
python -m benchmarks.run --backend warpower \
  --input "$AUDIT_EVIDENCE/bonsai-live-9216.npz" \
  --distribution bonsai-live-9216 --warmup 3 --repeats 5 \
  --mempool-retain-mb 32768 --output "$AUDIT_RESULTS/small-warp.json"
python -m benchmarks.run --backend paragram \
  --input "$AUDIT_EVIDENCE/bonsai-live-9216.npz" \
  --distribution bonsai-live-9216 --warmup 3 --repeats 5 \
  --output "$AUDIT_RESULTS/small-paragram.json"
python "$AUDIT_EVIDENCE/native_fresh_v2.py" \
  --input "$AUDIT_EVIDENCE/bonsai-live-9216.npz" \
  --module-dir "$NATIVE_REFERENCE" --warmups 3 --repeats 5 \
  --output "$AUDIT_RESULTS/small-native.json"
```

For the large probe substitute `bonsai-checkpoint-1968522.npz`; for the flagship
use `--distribution cluster5 --n 100000 --seed 42` without `--input` and with
`--repeats 10`. Set `PYTHONPATH` to the pinned checkout, run one GPU process at a
time and limit BLAS/OMP threads as recorded in `run_probes.py`. The optional
native reference build is described in the [update report](../geometry-updates/README.md).
The one-off audit/supervisor scripts retain their actual original local paths as
provenance; adapt those paths if rerunning them elsewhere.
