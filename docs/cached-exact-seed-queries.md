# Cached exact fallback for VTM seed queries

At VTM capacities 4,097–16,384, strict sphere tests and exact fallback now run
in separate kernels. Each lane caches its first unresolved point. A second
unresolved point marks overflow and preserves the complete lane rescan. The
existing predicates, symbolic tie policy, hierarchy and final certificate stay
in force. Independently callable diagram/Delaunay functions and smaller VTM
routes retain their existing implementations.

## Complete resident solves

Production `17c6361` is compared with main `7c3dfc7` on NVIDIA L40 / Warp 1.19.0.
Seven fresh interleaved pairs each record three complete measured replays,
including domain refit, sampling, fitting, quality/refinement and extraction.
Setup, compilation, transfers and offline validation are excluded. All initial
and measured windows have zero CPU quota throttling. Raw samples and setup costs
are retained in the [comparison](../results/validation/vtm/cached-exact-seed-queries/hierarchical-cache-final-v1-independent.json).

| Sites and schedule | Before (ms) | After (ms) | Change |
| --- | ---: | ---: | ---: |
| Uniform torus 16,384, fixed 32 | 6,263.809 | 5,260.133 | -16.02% |
| Uniform torus 8,192, progressive | 3,545.306 | 3,225.367 | -9.02% |
| Uniform torus 8,192, fixed 32 | 1,817.195 | 1,602.822 | -11.80% |
| Torus target 4,096, reserved 16,384 | 7,400.578 | 7,306.362 | -1.27% |
| Graded torus 16,384, progressive | 8,138.298 | 7,325.660 | -9.99% |
| Uniform torus 4,096 | 1,508.394 | 1,498.461 | -0.66% |
| Uniform torus 1,000 | 513.973 | 513.319 | -0.13% |

All fourteen meshes pass independent exact full-hull/orientation/empty-sphere,
material topology/quality, boundary vertex-link and boundary self-intersection
checks. Every pair has identical coordinates, canonical full/material cells,
oriented boundary, roles and work counters. The graded VTU exports round-trip.
These are GPU improvements; the wide comparable-quality native CPU goal remains
open. The [CPU study](fixed-site-current.md) records count/quality and timing
limitations. README retains its previously earned Paragram flagship result.

## Why cache the unresolved point

The first private split-only revision improved the fixed and graded 16K cases
by 16.2% and 9.9%, but regressed a structured-grid retained-mesh test by 7.13%
(4.363 to 4.674 ms). Its exact pass repeated the tree traversal. That negative
result, all private sources and all fourteen private normal-case output checks
are preserved. The earlier [small-capacity experiment](prepared-split-probe.md)
also remains unadopted.

On the 17³ integer grid, diagnostic instrumentation observes 50,341 queried cell
instances across three invocations, all requiring exact fallback. Of 199,236
flagged lanes, the revised cache resolves 197,114 directly and rescans 2,122.
The instrumented output matches the independently checked clean mesh. On the
fixed-16K torus, three graph executions query 67,017,258 cell instances across
681 calls without exact fallback. Instrumentation times are not benchmark samples.

The newly interleaved production grid guard measures **4.357 to 2.685 ms
(-38.38%)** over nine retained-mesh replays. Both 24,034-cell outputs
pass independent exact full-hull and physical quality checks and agree exactly.
Initial full construction is outside this guard's timing: it is neither a fresh
VTM solve nor the standalone public Delaunay benchmark.

## Current-invocation evidence and overflow

The strict pass retains all owner/incidence, hierarchy/point-box, bounded
binary32 and prepared Ball64 checks. Certified positive signs reject the cell;
negative signs continue. A rejected cell still visits its other queries.
Each lane saves its first unresolved point index. Further unresolved points
replace it with -2, which selects the original complete lane traversal in the
exact pass. Thus fixed cache capacity never limits the number of exact checks.
A single saved index invokes the unchanged exact/symbolic predicate directly.
Malformed saved indices fail closed.

The exact kernel follows the strict kernel in stream order, against unchanged
input within that invocation. Every invocation clears all pending masks; every
flagged lane writes its saved index before publishing its bit. Unflagged entries
are never read. No cached proof survives a later input update. All storage is
allocated during setup; replay adds a mask reset and ordered exact-query launch.
Replay has no host decision, readback, allocation or dynamic queue.

At 131,072 candidate cells, the 32-int cache adds 16 MiB and the masks add
0.5 MiB. The exact total is 17,301,504 bytes. Pool measurements are persistent
workspace allocator allocations, not process-wide peak memory:

| Reserved sites | Before pool bytes | After pool bytes |
| --- | ---: | ---: |
| 8,192 | 2,036,011,260 | 2,044,662,012 |
| 16,384 | 2,520,826,939 | 2,538,128,443 |

## Validation and native execution

The complete suite passes **1,237 tests**, including 131 focused query/traversal
and integration cases. New tests cover unsigned lane 31, all-lane overflow,
poisoned storage, zero/restored counts, both block sizes, and a decisive overflow
case: its first uncertain point permits the cell but its second rejects it.
Independent rational oracles and the existing flat checker establish the answers.
Captured producer/consumer tests include input updates, duplicates, invalid boxes,
thin clouds and extreme scales through 16,384 sites. Both promoted query-kernel
ASTs match the measured private cache revision. All 141 other runtime, benchmark
and example source files are byte-identical to the control.

Fourteen selected cases each pass isolated padded and pool-disabled lifetime
memory checks: **28 jobs, zero reported errors**. These include cache overflow
and large-route failure/recovery. The older combined-process sanitizer issue
[#155](https://github.com/alecjacobson/warpower/issues/155) remains separate.

Native traces cover a warmed full 16K solve and one device-controlled five-phase
loop: 4,097 sites, a deformed 8,192-site case, invalid target 8,193, collapsed
domain and recovery. The three valid frames pass independent geometry checks;
invalid frames expose zero counts and consumed volume. Output pointers stay
stable and recovery matches the initial mesh. Native graph nodes contain device
kernels, memsets, device-to-device copies and conditional control, with no host
callbacks, host transfers or explicit allocation/free calls. Each trace has one
external final synchronization. The first loop launch includes deferred graph
instantiation/upload before graph launch; this does not claim zero driver-internal
setup work or memory.

The full trace records 68,542 kernel activities and the
loop 207,900. Selected full-trace costs are diagnostic,
not additional clean timing samples:

| Kernel | Calls | Registers/thread | Activity (ms) |
| --- | ---: | ---: | ---: |
| strict_check | 227 | 80 | 1,162.000 |
| retained_cached | 194 | 255 | 930.882 |
| check_unproved_spheres | 554 | 255 | 825.750 |
| search_unproved_faces | 327 | 255 | 721.285 |
| record_face_support | 554 | 255 | 444.350 |
| exact_check | 227 | 255 | 16.895 |

## Independent API controls

| Independent operation | Before (ms) | After (ms) | Change |
| --- | ---: | ---: | ---: |
| delaunay1000 | 9.177 | 9.172 | -0.06% |
| delaunay100000 | 182.756 | 182.934 | +0.10% |
| delaunay-grid | 2.210 | 2.188 | -0.96% |
| delaunay-fitted-hand | 13.391 | 13.423 | +0.24% |
| diagram100000 | 18.893 | 18.021 | -4.62% |

These fresh pairs use nine complete Delaunay replays and thirty diagram runs,
with matching inputs, output counts/status/diagnostics and clean quota windows.
The public APIs remain independently callable and their code paths are unchanged.

Tracked in [#175](https://github.com/alecjacobson/warpower/issues/175). The
[manifest](../results/validation/vtm/cached-exact-seed-queries/manifest.json)
pins sources, every positive/negative experiment, complete timings, independent
checks, memory logs and native traces. The later [new-cell checker](prepared-new-cell-queries.md)
shares current geometry and traverses existing bounds. Larger active-prefix
routing [#230](https://github.com/alecjacobson/warpower/issues/230), scratch sharing
[#122](https://github.com/alecjacobson/warpower/issues/122), and known Armadillo
extraction failures [#228](https://github.com/alecjacobson/warpower/issues/228)
remain separate work. No extraction guard or quality criterion is weakened.
