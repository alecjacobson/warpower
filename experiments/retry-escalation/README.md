# Independent retry escalation

The follow-up to [predicate recovery](../robust-neighbor-recovery/README.md)
finishes the separation of numerical recovery and geometric storage growth.
The earlier `12f5444` implementation skipped numerical failures in ordinary
FP32 retries, but captured retries lacked that guard and cooperative FP64
capacity failures could lose their precision history. Both gaps are corrected.

Each original site has independent precision requirements and a storage floor,
shared by the ordinary and captured retry kernels. A numerical failure promotes
to FP64 and preserves the size already reached; later capacity failure cannot
downgrade that cell. Capacity-only failures never promote precision, including
at the supported size limit. Captured replay resets this state for new inputs.

| Failure | Next action |
|---|---|
| Capacity overflow at FP32/128 | FP32/256 |
| Numerical failure at FP32/256 | FP64/256, bypassing the 128-slot cooperative kernel |
| Capacity overflow at FP64/128 | FP64/256 |
| Numerical failure in serial FP64 | Explicit numerical failure; no fictitious size escalation |
| Capacity overflow at FP32/512 | Explicit capacity failure; no automatic FP64 promotion |
| Output row fills but clipping scratch fits | Widen the row while retaining precision and scratch size |

Eligible small cells get at most one 128-slot cooperative FP64 attempt. Larger
requirements bypass that queue. Failed cooperative numerical checks may use
the serial FP64 implementation at an adequate size; if that also fails, the
numerical status stays explicit. Completed rows remain intact. The original
cancellation screen and floating-point clipping are still not a global exactness
certificate. See [API](../../docs/api.md) and [architecture](../../docs/architecture.md).

## Performance and memory

All 74 historical inputs pass the existing per-case gate against `12f5444`:
reject a paired 95% Student-t interval wholly above +2%. The geometric-mean
speed ratio is **0.9986×**. This incremental
comparison is separate from the earlier 1.097× improvement over `655412e`.
Small positive changes remain visible; passing the gate does not mean every
case has zero overhead.

L40; three warmups; twenty alternating paired synchronized wall-clock calls;
resident identical inputs; construction allocations included; 32 GiB allocator
retention. No accepted timing window is CPU-throttled. Live Bonsai uses thirty pairs.

| Input | Baseline (ms) | Updated (ms) | Paired change | 95% interval |
|---|---:|---:|---:|---:|
| uniform-100000-power | 14.269 | 14.211 | -0.38% | [-1.29%, +0.54%] |
| gradient-100000-voronoi | 13.107 | 13.119 | +0.11% | [-0.54%, +0.75%] |
| cluster10-100000-power | 25.388 | 25.482 | +0.39% | [-0.59%, +1.37%] |
| live-bonsai-9216-voronoi | 83.002 | 81.346 | -1.54% | [-4.07%, +0.98%] |
| bonsai-1968522-voronoi | 223.638 | 224.219 | +0.26% | [+0.02%, +0.50%] |
| uniform-15000000-voronoi | 1725.438 | 1729.979 | +0.27% | [+0.05%, +0.48%] |

Four previously marginal cases also pass a direct comparison against original
baseline `655412e`, with thirty paired repetitions:

| Input | Baseline (ms) | Updated (ms) | Paired change | 95% interval |
|---|---:|---:|---:|---:|
| gradient-100000-power | 14.059 | 14.153 | +0.73% | [-0.03%, +1.49%] |
| gradient-100000-voronoi | 12.820 | 13.070 | +1.95% | [+1.46%, +2.44%] |
| uniform-100000-power | 14.009 | 14.186 | +1.27% | [+0.82%, +1.73%] |
| uniform-100000-voronoi | 12.755 | 12.681 | -0.56% | [-1.16%, +0.04%] |

Separate eager/captured geometry timings and measured persistent workspace memory:

| Mode | Sites | Input | Output | Baseline (ms) | Updated (ms) | Paired change | 95% interval | Workspace MiB, before → after |
|---|---:|---|---|---:|---:|---:|---:|---:|
| eager | 100,000 | uniform Voronoi | neighbors | 12.666 | 12.721 | +0.44% | [-0.22%, +1.11%] | — |
| eager | 100,000 | cluster5 Voronoi | neighbors | 20.871 | 21.078 | +1.05% | [-0.37%, +2.46%] | — |
| eager | 1,000,000 | uniform Voronoi | neighbors | 100.296 | 100.718 | +0.42% | [+0.26%, +0.58%] | — |
| eager | 1,000,000 | uniform power | neighbors | 107.088 | 106.999 | -0.08% | [-0.29%, +0.13%] | — |
| graph | 100,000 | uniform Voronoi | neighbors | 10.973 | 10.938 | -0.32% | [-0.94%, +0.31%] | 735.86 → 796.62 |
| graph | 100,000 | uniform power | neighbors | 12.554 | 12.445 | -0.86% | [-1.70%, -0.01%] | 735.41 → 796.24 |
| graph | 100,000 | uniform Voronoi | tetrahedra | 473.244 | 473.231 | -0.00% | [-0.02%, +0.01%] | 402.95 → 403.02 |
| graph | 100,000 | uniform power | tetrahedra | 477.757 | 477.785 | +0.01% | [-0.01%, +0.03%] | 402.57 → 402.64 |
| graph | 100,000 | uniform Voronoi | neighbors + tetrahedra | 487.540 | 487.730 | +0.04% | [+0.01%, +0.07%] | 913.41 → 974.24 |
| graph | 100,000 | uniform power | neighbors + tetrahedra | 492.543 | 492.591 | +0.01% | [-0.01%, +0.03%] | 913.03 → 973.86 |

Retry state adds eight bytes per site. Captured FP32 scratch now supports 512
slots, which costs additional persistent memory as shown above. Setup/allocation
are excluded from captured replay timing and included in eager timing.
The geometry gate returns identical neighbor sets and tetrahedron sets.

Complete VTM solves preserve bitwise positions, vertex roles and tetrahedron sets,
with the existing hull-proof cache enabled in both versions. Three warmups, ten
alternating pairs; setup/JIT/downloads excluded, synchronization included:

| Complete 4K VTM solve | Baseline median (ms) | Updated median (ms) | Paired change | 95% interval |
|---|---:|---:|---:|---:|
| torus | 921.714 | 918.144 | -0.43% | [-0.53%, -0.32%] |
| graded-torus | 824.254 | 822.647 | -0.21% | [-0.29%, -0.13%] |

## Correctness and sanitizer scope

All **185 targeted tests pass**, including 17 additional escalation regressions. They force
real 128→256→512 clipping-buffer overflow at both precisions, require precision
retention and storage floors in both dispatch paths, check numerical statuses
5/6 at larger capacities, distinguish output-row overflow, enforce the size limit,
and verify fresh state on captured replay.

Exact rational facet audits recheck every CGAL disagreement, every baseline change,
and all fourteen directions of the seven earlier wrong pairs. There are no candidate
errors or regressions in the audited set. Other shared unchanged pairs are not certified.

| Scene | Directed pairs audited | Baseline errors | Updated errors |
|---|---:|---:|---:|
| live-bonsai | 172 | 0 | 0 |
| bonsai | 368 | 0 | 0 |
| garden | 1182 | 0 | 0 |
| bicycle | 1566 | 0 | 0 |
| playroom | 474 | 0 | 0 |

**Sanitizer coverage is qualified:** memcheck and initcheck pass all 59 recovery/
escalation tests; racecheck and synccheck pass 21 focused retry/cooperative cases,
including captured individual retry kernels. Broader captured-workspace checks
also reproduce diagnostics on baseline `12f5444`: synccheck reports divergence in
unchanged `block_bounds`, and racecheck reports cross-kernel shared-memory hazards
between cooperative construction and `collect_failures`. Their root causes remain
unresolved; they are not labeled false positives. Baseline/candidate failure logs,
tool versions and isolation probes are retained. No sanitizer suppression is used.
The full workspace tests pass ordinary execution, memcheck and initcheck.

## Evidence

The [verified evidence archive](https://github.com/alecjacobson/warpower/releases/download/evidence-geometry-updates-4ceb748/retry-escalation-20261112.tar.gz) contains the complete implementation patch,
all per-repetition samples, input/source hashes, exact queries, test/sanitizer logs,
isolation probes and runners. SHA-256: `747f83b5340bae4a744cf577d81b97f616a3d3c974cc5c1c778194efeacad9d4`. A fresh download verified
all members, and applying the patch to `12f5444` reconstructed all recorded library
and changed-file hashes. See [manifest](manifest.json), [verification](remote-verification.json)
and [full tables](validation.json).

The earlier v1 screen and failed broad sanitizer runs are retained; acceptance
measurements use frozen v2. Full-capture sanitizer findings remain follow-up work
tracked with [issue #300](https://github.com/alecjacobson/warpower/issues/300).
Issue #301 also requests resource-specific growth and per-stage reason diagnostics.
Plane, vertex and traversal storage still use the existing bundled capacity
classes; the cooperative implementation remains an FP64 numerical-recovery
kernel. Those broader policy/diagnostic requests remain open. This follow-up
completes precision-versus-storage separation; it does not close all of #301.
RadFoam/VoroTracing fitting runs remain deferred.
