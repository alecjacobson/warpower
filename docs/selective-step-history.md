# Bounded selective step history

The complete mesher avoids repeating large selective proposals after a preceding
search found that only smaller proposals passed its quality guard. The policy is
eligible only after quality-only rejections; a non-quality warm rejection returns
to the ordinary search immediately. It adds 44 device bytes, with all decisions,
resets and retries recorded in Warp kernels and CUDA graphs.

This changes the order of trials and is not a bitwise trajectory guarantee.
Skipping a rejected reconstruction can change later reuse history and rounding.
Qualification therefore checks complete meshes, element quality, boundary fit,
work counts and convergence, rather than treating a rejected-prefix count as a
speedup. The original ungated prototype regressed uniform Armadillo by 4.98% and
was rejected; its records remain in the historical evidence bundle.

## Policy and resets

For an eligible selective search, the initial scale is twice the preceding
accepted scale, capped by the original damping. The halving counter starts at
that absolute level, preserving the original smallest trial scale. Only actual
rejected proposals contribute to `rejected_steps`.

A warm search retries the full original bracket once if it exhausts its levels,
encounters a non-quality failure, or appears converged because actual movement is
small. The convergence retry discards that small accepted trial without counting
it as an accepted iteration or a rejection. A default-scale search must then
confirm convergence. Rejection need not be monotone in scale.

History admission uses fresh selective rejection counters and quality reason 4,
not a sticky error from an earlier trial. Geometry, objective, material, boundary,
feature, minimum-quality, actual-displacement and transaction rollback checks
remain in place. A retry cannot loop indefinitely: each iteration gets at most
one additional ordinary bracket.

Every core invocation resets history, covering mode/stage transitions, new
active prefixes and fresh inputs. Every complete solve resets statistics. Global
searches retain the original scale. Optional local vertex rollback retains its
original scheduling. `VTMWorkspace` used independently defaults to the original
policy; the complete mesher selects history internally when eligible. Standalone
power-diagram and Delaunay routines do not use this module.

## Current-main qualification

The current integration is pinned separately from the historical pilot: baseline
`f3b35c9` and candidate `de9d5a2`. It retains the merged initial corner budget and
all boundary-query precision options. The ordinary suite passes 1,458 tests and
33 subtests. All original 37 memory contracts pass in each of padded and
allocation-lifetime modes, across 42 successful isolated processes, with a 32 KiB
diagnostic instrumentation stack. Production stack/allocator settings are unchanged. These memory contracts use the
reference surface backend; the native refit and descriptor-instrumentation
limitations in #267/#269 remain.

The combined local-retry batch completed four tests, then stalled in a later
constructor's setup synchronization. It was terminated incomplete after 373.71
seconds and is retained under #155; it is not counted as a pass. Separate runs
complete every remaining contract without dropping cases.

## Complete solves and quality

NVIDIA L40 / Warp 1.19.0; median five additional fresh captured solves, including
domain refit, sampling, growth, fitting, quality stages and final extraction.
Setup, transfers, export and independent CPU checks are outside the timing window.
Every timing window has full CPU quota. These compare GPU configurations, not
CPU implementations. Negative percentage means less time.

| Case | Original | History | Change |
|---|---:|---:|---:|
| torus16384-fixed32 | 4392.38 ms | 2911.10 ms | -33.72% |
| torus1000 | 488.53 ms | 488.07 ms | -0.09% |
| torus8192-fixed32 | 1362.58 ms | 1242.61 ms | -8.80% |
| torus8192-progressive | 2638.21 ms | 1975.68 ms | -25.11% |
| thin-hand-seed7 | 1218.13 ms | 1225.72 ms | +0.62% |
| cavity | 351.73 ms | 335.02 ms | -4.75% |
| disconnected | 478.15 ms | 478.38 ms | +0.05% |
| plate | 866.26 ms | 786.70 ms | -9.18% |
| armadillo-graded4096 | 2226.95 ms | 2076.84 ms | -6.74% |
| armadillo-graded8192-recovery | 3686.48 ms | 3332.95 ms | -9.59% |
| armadillo-uniform8192-recovery | 3454.52 ms | 3460.20 ms | +0.16% |
| armadillo-graded8192-native-mixed | 2511.72 ms | 2301.08 ms | -8.39% |
| armadillo-uniform8192-native-mixed | 2317.98 ms | 2326.39 ms | +0.36% |

All 26 saved meshes pass independent exact full-hull/original-coordinate
empty-sphere and hull-support checks, material incidence, boundary vertex links,
and exact boundary self-intersection checks. Gallery VTU round trips are exact.
Roles, canonical full/material connectivity, oriented boundary faces and accepted
work counts match across every pair. Most sites are bitwise identical; the
complete numerical comparisons retain the nonidentical cases below.

| Nonidentical sites | Maximum coordinate difference | Minimum radius ratio, before → after | Surface RMS (% bbox diagonal), before → after |
|---|---:|---:|---:|
| thin-hand-seed7 | 3.46944695e-18 | 0.110977485 → 0.110977485 | 0.00599947673 → 0.00599947673 |
| disconnected | 4.65661287e-10 | 0.0811127832 → 0.0811127832 | 0.0198386194 → 0.0198386194 |
| armadillo-graded4096 | 0.00011318922 | 0.0740833902 → 0.0740841834 | 0.275174752 → 0.275178731 |

The graded 4K case retains a small fit change rather than being described as
bitwise equivalent. All quality percentiles, mean radius ratio, dihedral extrema
and volume errors are published. The policy changes trial order; these measured
comparisons are not a universal identical-trajectory guarantee.

## Device execution and public routines

The exercised 16K history replay has 47,575 CUDA kernel activities, 21 warm
starts, 20 warm accepts and one default-scale convergence retry. Its sites,
canonical connectivity, oriented boundary, work counters and history statistics
match the independently validated timed output. A separate five-phase graph
produces changed inputs/targets, refits, meshes and consumes the output, including
invalid target, collapsed domain and valid recovery; it has 41,850 kernel activities.
Both traces contain only device-to-device copies and no host callbacks or replay
allocation APIs. All graph copy nodes are checked too, because Nsight may omit
some graph copy activity records. These audits use the reference surface backend;
they do not resolve the optional native-mesh refit limitation in #267.

The ten-case paired public power/regular-diagram matrix passes its 2% regression
gate with equal outputs and unchanged eager launch/allocation traces. Separate
Delaunay 1K/100K/grid/hand controls and 100K diagrams have unchanged diagnostic
counts; their measured changes are +0.51%, -0.15%, -0.18%, -0.11% and -0.03%.
Each eligible complete workspace adds exactly 44 bytes. No history arrays or
history launches are added to standalone power diagrams or Delaunay workspaces.

Source pins, commands, samples, numerical reports, memory logs' hashes and raw
inventories are in [experiments/selective-step-history](../experiments/selective-step-history/).
Raw meshes and traces remain outside ordinary Git. This closes the bounded
history study (#257/#255), not uniform-scan material recovery (#258), the default
input problem (#263), or comparable-quality CPU performance (#40).
