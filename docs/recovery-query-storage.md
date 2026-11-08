# Recovery query storage

Complete tangential topology recovery now omits the approximate retained-cell
query cache and five scratch arrays inherited from ordinary quality moves.
Recovery always extracts and validates a complete candidate mesh, so it never
uses that approximate path. Ordinary boundary jitter and interior moves retain
their arrays, as does the recovery candidate extractor's separate, active cache.
Checkpoints and rollback storage are retained. There is no public API change.

The private `QualityMove(..., _approximate_queries=True)` constructor choice
defaults to the existing behavior; only `TopologyRecovery` passes `False`.
An AST audit resolves 18 current methods and callbacks, and runtime guards forbid
the unused callback during gallery capture. This is an audit of the current
explicit call structure, not a general Python alias proof. A production source
audit pins 262 files: 259 are unchanged, with only two constructors and the new
focused test differing. All device kernels and replay methods are unchanged.

## Storage and replay measurements

Baseline `5035fa5895a3`, runtime `cbaa691a55db`, NVIDIA L40, Warp 1.19.0.
Each full solve uses three warmed fresh captured replays, including sampling,
growth, fitting, quality moves and extraction. Setup/upload/readback/export are
excluded. All 20 before/after timing jobs have clean CPU-quota windows.

| Complete solve | Before, ms | After, ms | Change | Saved pool bytes |
|---|---:|---:|---:|---:|
| Torus, 16,384 sites | 4893.938 | 4896.811 | +0.06% | 25,690,136 |
| Thin hand, 1,000 sites, seed 7 | 1251.502 | 1247.882 | -0.29% | 1,571,096 |
| Graded Armadillo, 4,096 sites | 2579.089 | 2582.116 | +0.12% | 6,422,552 |
| Graded Armadillo, 8,192 sites | 4313.938 | 4315.577 | +0.04% | 12,845,080 |
| Uniform Armadillo, 8,192 sites | 3944.983 | 3958.417 | +0.34% | 12,845,080 |

The saved payload is `180*m + 4*next_power_of_two(4*m) + 24` bytes, where `m`
is tetrahedron capacity (eight times site capacity here). Observed allocator
savings match this expression. These are savings from one unused portion of the
recovery workspace, not its entire storage cost. The 16K torus workspace changes
from 4,582,747,092 to 4,557,056,956 pool bytes; setup in this pair changes from
4.976 to 4.838 seconds. Raw constructor/setup measurements for every case are in
the [comparison](../results/validation/vtm/recovery-query-storage/quality-query-storage-final-v2-independent.json).
No replay speedup is claimed for this construction-only change.

The independent geometry functions keep their source and allocations:

| Standalone case | Before, ms | After, ms | Change |
|---|---:|---:|---:|
| Delaunay, 1K | 9.179 | 9.199 | +0.22% |
| Delaunay, 100K | 183.280 | 183.354 | +0.04% |
| Delaunay, grid | 2.202 | 2.194 | -0.37% |
| Delaunay, fitted hand | 13.438 | 13.407 | -0.23% |
| Power diagram, 100K | 18.126 | 18.160 | +0.19% |

These short paired measurements show no material regression in the measured
cases; they are not a statistical bound for every workload. They do not establish
the outstanding comparable-quality CPU speedup goal. The README flagship is
unchanged.

## Geometry and graph checks

All 1,327 tests pass, including 11 focused cases with the complete recovery loop.
Ten final before/after meshes pass independent complete-hull/Delaunay, material,
boundary topology, vertex-link and exact self-intersection checks. The eight
gallery exports also pass VTU round trips. Two already completed torus proofs are
reused by exact mesh and report hashes; the remaining eight meshes are checked
by the final v2 validator. The private pilot separately validates eight meshes.
Material cells, oriented boundaries, roles, scheduler work and recovery decisions
agree. Four of the five final pairs have bitwise equal points and identical full
connectivity.

The hand pair changes two near-zero coordinates, by at most `3.47e-18`, and
replaces four exterior full-hull tetrahedra with four others. Both complete hulls
independently validate; material cells and oriented boundary are unchanged.
Two unchanged-baseline runs exhibit exactly the same full-cell variation, and
the production hull matches the other certified baseline. This demonstrates
baseline variation without identifying its precise cause. The initial strict
full-connectivity checker failed and is retained. Its replacement preserves
`full_tets_equal=false` for this case and retains exact connectivity checks for
all other cases. See the [integer-ID variation report](../results/validation/vtm/recovery-query-storage/quality-query-storage-hand-variation-v2.json)
and [#208](https://github.com/alecjacobson/warpower/issues/208). The first diagnostic
JSON accidentally serialized NumPy integer cell IDs as booleans; it is retained
as an invalid connectivity diagnostic, and v2 corrects that encoding explicitly.

The unchanged full 8K nondefault-stream loop exercises one-attempt exhaustion,
successful recovery, invalid target, collapsed domain, then successful recovery.
Output pointers stay stable, invalid outputs have zero counts, and diagnostic
resets agree. Ordinary, native-profiled and both sanitizer loops match the
independently certified reference's bitwise coordinates, roles, inputs and
oriented connectivity. Exhaustion restores points and the complete hull. This
transfers the reference geometry proof by exact comparison; it is not another
independent predicate run on each identical frame.

Native traces of the full solve and loop contain respectively 90,971 and 156,956
kernel activities, matching the prior recovery implementation. Captured nodes
are kernels, conditional nodes, memsets and device-to-device copies. Each replay
window has one graph launch and one final external synchronization, without
host transfers, callbacks or allocation/free calls. First-launch instantiation
and upload, where present, remain disclosed in the raw audit.

The new storage contract and the unchanged full loop each run in two isolated
sanitizer modes: 32-byte padded memcheck, and pool-disabled API/lifetime checking.
All four jobs pass with zero errors. Diagnostic stack size is 131,072 bytes;
periodic 60-second stack dumps are observations, not deadlines. The storage jobs
have 600-second limits and full loops 1,200-second limits. Previous combined
workspace instrumentation limitations remain documented in the
[recovery qualification](tangential-topology-recovery.md); this change makes no
claim to fix those instrumentation limits.

The [manifest](../results/validation/vtm/recovery-query-storage/manifest.json)
preserves source bindings, commands, raw meshes/exports, compressed native
traces, failed diagnostics and prior proof references. Scan input attribution
and terms remain those of the original Stanford/libigl asset.
