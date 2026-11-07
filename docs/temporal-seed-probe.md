# Temporal retained-cell sphere-proof probe

The private #130 sphere-proof experiments preserve the tested meshes but improve
complete 1k/4k torus solves by less than 1%. They remain unadopted. Production
keeps the [current-input flat-group checker](grouped-seed-queries.md), without
temporal snapshots. No standalone API, default kernel or README result changes.

## Proof and provenance

After a successful complete constructor call, the prototype snapshots original
float32 positions, canonical owners, the exact retained tetrahedron rows and
their count. Snapshot availability is published after the copies. Failure or a
fresh-trajectory reset invalidates it. The original hierarchy's owner buffer
aliases the base constructor's current owner buffer; setup checks this identity.
Each routed constructor owns its own snapshots. All buffers are allocated at
setup, and copies, comparisons and availability decisions remain device work.

Before canonicalizing a retained source cell, every corner must match its saved
row ID, remain canonical under that same ID, and retain its saved exact coordinate
value. The winning source carries this unchanged-cell flag through deduplication.
Checking only the resulting unique corners would be insufficient: remapping a
moved old corner onto an unchanged existing site can create a different cell.
The saved row comparison also prevents altered retained connectivity from
inheriting an unrelated proof. Coordinate comparisons use exact component
equality, with no tolerance; signed zeros represent the same geometric value.

For such an unchanged previously certified cell, an unchanged previously
canonical query ID has the same physical and symbolic sphere result. The checker
therefore visits only current canonical queries that moved or were previously
noncanonical. The latter includes unchanged duplicates promoted when an old owner
moves. New, changed, remapped or unproved cells retain complete ordinary queries.
Every retained query keeps the existing conservative point-box, Ball64 and exact
symbolic tests. Topology, coverage, hull support and all new-cell certificates
remain mandatory. This probe does not cache hull-support results.

## Complete solves

Fresh controls use main 651e35f, with runtime b5d369c, on NVIDIA L40 / Warp 1.19.0.
All first and timed quota windows are unthrottled. Measurements include complete
resident-device refit, sampling, growth, fitting, quality stages, extraction and
diagnostics; allocation, compilation, transfers and offline checks are separate.
The 1k case uses five timed replays and the 4k case three.

| Private variant | Sites | Control, ms | Candidate, ms | Reduction |
|---|---:|---:|---:|---:|
| v1: compact changed-query list | 1,000 | 703.477 | 702.030 | 0.21% |
| v1: compact changed-query list | 4,096 | 2,401.547 | 2,387.517 | 0.58% |
| v2: production launch sizes restored | 1,000 | 704.868 | 700.982 | 0.55% |
| v2: production launch sizes restored | 4,096 | 2,402.247 | 2,386.221 | 0.67% |
| v3: compact/grouped hybrid | 1,000 | 702.864 | 700.602 | 0.32% |
| v3: compact/grouped hybrid | 4,096 | 2,399.568 | 2,394.045 | 0.23% |

The initial copied seed kernel inherited a 256-thread module default instead of
the production module's 32-thread default. v2 restores that launch size, with
identical device-function/kernel syntax trees. All initial results are retained.
v3 uses the compact list only for at most 32 changed queries; otherwise it keeps
Morton-group pruning and skips unchanged query IDs with freshly written flags.
It does not improve on v2. No variant has undergone broad adoption validation.

All six candidate meshes are bitwise equal to their respective controls in
positions, with matching full/material cell sets, oriented boundaries, site roles,
inputs, parameters and work counters. Each passes independent CGAL exact global
full-hull checks and NumPy/AABB material quality checks, including single-cycle
boundary-vertex links.

v1/v2 add 212,012 bytes of persistent allocator-pool storage at 1k and 1,519,652
bytes at 4k. v3 adds 216,012 and 1,548,324 bytes. These include routed workspaces,
not process-wide peak memory. The snapshots add four device copies per completed
construction plus fixed metadata kernels. Raw records retain every setup and
first-execution measurement; these are single observations, not startup speedup
evidence.

## Diagnostic and tests

An instrumented v2 4k run records three complete graph executions: 624 checker
calls, 12,882,912 cell instances and 4,323,564 unchanged proved cells (33.56%).
It records 1,054,374 changed-query entries across those calls. The numbers of
calls with zero, one, 32–255, 256–1023 and at least 1,024 changed queries are
27, 21, 6, 66 and 504, respectively; none have 2–31. A compact list discards the
flat groups' spatial ordering and pruning, motivating v3. The diagnostic output
is bitwise equal to the uninstrumented mesh and work counters match. Instrumented
timings are excluded from performance evidence; these counts do not measure
the exact number of eliminated Ball64 evaluations.

Each executable variant passes 69 existing reconstruction tests. v1 and v2
also pass 18 parameterized provenance/mask tests; v3 passes an expanded 24 cases.
The latter compare 15 device-produced phases, replayed twice, with exhaustive
flat-group checks and independent rational symbolic masks on small inputs.
Cases include unchanged/one/all moved sites, duplicate-owner promotion, canonical
corner remapping, altered valid-looking and invalid source rows, unavailable
snapshots, shrinking/empty/restored prefixes and invalid counts. They span cube,
octahedron and random inputs, scales 2^-60 through 2^60, and blocks 32/256.
Explicit 31/32/33-query cases put changed sites strictly inside a cached cell's
sphere, requiring rejection on both sides of the hybrid dispatch threshold.
Old snapshot/proof/query buffers are poisoned before use.

The first v0 compile attempt used unsupported vector inequality: 54 existing
tests passed and 15 stopped at compilation. Its source and failure log are
retained. Replacing it with explicit component comparisons yields the executable
variants above. No failure is counted as a correctness pass.

## Evidence and remaining scope

The [manifest](../results/validation/vtm/temporal-seed-probe/manifest.json)
records every variant, builder, command, log, timing sample, output, independent
check and diagnostic, with verified hashes. For reproduction in a fresh checkout
of the recorded baseline, materialize the selected archived variant under its
original import name, temporal_seed_queries.py, before running that variant's
recorded entry point. The v2 diagnostic uses its archived v2 adapter, while the
latest adapter targets v3. Workspace-specific paths identify the measured run.

There is no new full-suite, broad gallery, sanitizer, native-capture or standalone
timing claim for these unadopted prototypes. The tested unchanged-cell premise
does not by itself establish a worthwhile complete-solve gain. Broader temporal
reuse, including hull-support proofs, remains under
[#130](https://github.com/alecjacobson/warpower/issues/130). The wide CPU
outperformance goal remains unmet.
