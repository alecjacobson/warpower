# Certified VTM fragment reuse through 16,384 sites

The active VTM reconstruction route now uses the existing certified fragment and
fresh-wall strategies for reserved capacities from 2,049 through 16,384 sites.
The implementation changes one setup bound. The original-coordinate proof
kernels, ordinary fallback constructor, independently callable power-diagram and
Delaunay paths, and meshing schedules are unchanged.

This extends the earlier [4,096-site integration](large-fragment-reuse.md).
Every reconstruction checks retained cells against the current original input.
Those proofs survive completion within that reconstruction; no proof survives an
input change. Fresh solves clear retained fragments. Failed fragment/wall
completion still uses the ordinary certified constructor, and finite output
exhaustion still returns an error with zero counts. Reserved capacities above
16,384 retain the previous full-construction strategy.

## Complete resident solves

Production `f8e2834` is compared with control `689fd31` on NVIDIA L40 / Warp 1.19.0.
The four ordinary large-torus controls were recorded at `b44ab9e`, whose runtime
is identical to `689fd31`. The graded control comes from the preceding private
study; the small/reserved and standalone pairs are newly interleaved runs.
Three measured complete fresh replays include refit, sampling, fitting,
refinement, quality stages and extraction. Setup, compilation, transfers and
offline checks are excluded. All reported successful timing windows have zero
CPU quota throttling. Separate setup/initial execution and all raw samples are
retained.

| Sites and schedule | Control (ms) | Production (ms) | Ratio |
| --- | ---: | ---: | ---: |
| Uniform torus 8,192, progressive | 21,649.395 | 3,675.246 | 5.89× |
| Uniform torus 8,192, fixed 32 | 13,933.646 | 1,889.039 | 7.38× |
| Uniform torus 16,384, progressive | 45,102.121 | 7,443.444 | 6.06× |
| Uniform torus 16,384, fixed 32 | 60,049.695 | 7,122.329 | 8.43× |
| Graded torus 16,384, progressive | 51,060.051 | 8,842.355 | 5.77× |
| Torus target 4,096, reserved 16,384 | 24,222.109 | 7,566.430 | 3.20× |

All 9 successful control/production pairs preserve canonical full/material
connectivity, oriented boundaries, roles and recorded iteration work; 9 have
bitwise identical positions. Each new successful output passes independent CGAL
exact full-hull/orientation/empty-sphere checks, material topology and quality
measurements, single-cycle boundary vertex links, and a boundary self-intersection
check with zero pairs. Retained earlier controls have the same independent checks.
Gallery VTU exports round-trip exactly. See the [complete comparison record](../results/validation/vtm/extended-fragment-reuse/extended-reuse-final-v1-independent.json).

The five earlier private uniform/graded torus pairs also preserve bitwise
positions and canonical connectivity, with 5.77–8.43× gains. The production change
has exactly the private prototype's substituted class source; all other runtime
files and predicate kernels are unchanged.

## Small cases and independent APIs

| Complete torus solve | Control (ms) | Production (ms) |
| --- | ---: | ---: |
| Target/capacity 1,000 | 512.634 | 512.920 |
| Target/capacity 4,096 | 1,509.285 | 1,511.032 |
| Target 1,000, capacity 8,192 | 561.352 | 561.606 |

Standalone timings cover the entire existing operation, with nine Delaunay
replays and thirty power-diagram runs. Their source paths are unchanged.

| Independent operation | Control (ms) | Production (ms) | Change |
| --- | ---: | ---: | ---: |
| delaunay1000 | 9.190 | 9.175 | -0.16% |
| delaunay100000 | 181.898 | 181.232 | -0.37% |
| delaunay-grid | 2.200 | 2.215 | +0.65% |
| delaunay-fitted-hand | 13.417 | 13.391 | -0.20% |
| diagram100000 | 18.117 | 18.136 | +0.11% |

## Persistent storage and capture

| Reserved sites, progressive torus | Control pool bytes | Production pool bytes | Additional bytes |
| --- | ---: | ---: | ---: |
| 8,192 | 1,967,118,853 | 2,036,017,404 | 68,898,551 |
| 16,384 | 2,396,502,629 | 2,534,289,772 | 137,787,143 |

These measure persistent workspace allocator-pool allocations, not process-wide
peaks. A reservation pays the extra storage even when a smaller active target
only uses child routes. No allocations occur during replay.

The full suite passes **1,121 tests**. The 15 expanded large-route cases cover
capacities 2,050/4,096/4,097/8,192/16,384; active prefixes across 1,024/2,048;
captured producer/consumer work; nonfinite input and recovery; fresh resets;
corruption/fallback and output exhaustion at 2,050/8,192; and cospherical duplicate
ownership at scales 2^-60, 1 and 2^60. All 15 pass both padded memory checking and
pool-disabled allocation-lifetime checking with zero reported errors, in fresh
processes for each parameterized case. The first combined sanitizer attempt
stalled at setup synchronization after two test markers; its interrupted log and
failed qualification record are retained, not counted as a pass. The unresolved
combined-process behavior is tracked in
[#155](https://github.com/alecjacobson/warpower/issues/155).

A native 16k fixed-site full solve and a single device-controlled five-phase loop
show kernel, memset, device-to-device copy and conditional nodes only. There are
no host callbacks, host copies or allocation/free nodes/APIs. The loop reserves
8,192 sites and exercises 4,097 sites, deformed geometry at 8,192, an invalid
8,193 target, a collapsed domain, and recovery to 4,097. Both invalid phases expose
zero counts and zero downstream consumed volume; all three successful snapshots
pass independent hull/material/boundary checks. Output pointers stay fixed.
Each profile contains one final external synchronization. Explicit array/workspace
setup and final readback are outside the trace. The first loop launch also records
Warp’s deferred executable-graph instantiation and upload before its one graph
launch; these are initial preparation, not per-phase host decisions. The warmed
full-solve trace has no such calls. Driver-internal preparation is not claimed to
require zero host work or internal storage. Profiling durations are diagnostic,
not clean benchmark timings.

## Remaining work

Uniform and graded Armadillo at 8,192 sites fail extraction in both control and
extension. The graded case also fails with doubled surface quadrature. These
failed outputs and timings are retained without a successful-mesh speedup claim;
one denser control has CPU quota throttling. In the graded diagnostic, removing
six exterior-center cells repairs four bad edges but leaves two disconnected
boundary fans at vertex 6,598. The existing guard correctly rolls back and
publishes zero counts. [Issue #228](https://github.com/alecjacobson/warpower/issues/228)
tracks the recovery problem. This PR changes no extraction guard.

These are gains over the GPU implementation. The 16k fixed-site solve remains
slower than the separately measured native parallel CPU result of about 2.44 s,
which has differing counts/quality and includes throttled samples. The
[comparable-quality CPU speed goal](fixed-site-current.md) remains open.

The earlier native diagnostic attributes 3.03 of 7.08 seconds of kernel activity
to grouped seed checks. [Issue #177](https://github.com/alecjacobson/warpower/issues/177)
is now implemented by the separately measured [hierarchical seed queries](hierarchical-seed-queries.md).
The timings in this report predate that follow-up. Larger active prefixes still pay for the full
reservation after 2,048 sites; [#230](https://github.com/alecjacobson/warpower/issues/230)
tracks additional routes, with scratch sharing in
[#122](https://github.com/alecjacobson/warpower/issues/122). These are separate
follow-ups so their timing and storage costs can be measured independently.

Reproduction commands, pinned source and native binary hashes, raw results,
failed cases, and lossless native traces are linked in the
[provenance manifest](../results/validation/vtm/extended-fragment-reuse/manifest.json).
