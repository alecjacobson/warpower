# Clipping workspace ownership and first-pass capacity

The active VTM constructor builds full-capacity and prefix Delaunay workspaces
before capture. Their clipping kernels execute serially, but previously each
original/normalized builder reserved a separate set of FP64 temporary cell arrays.
An explicit owner now allocates those arrays once per active-constructor family.
The largest builder reserves first; later builders must fit without resizing.
Independent families have separate owners and can execute on different streams.

Only transient clipping arrays are shared. Hierarchies, emitted tetrahedra,
retained fragments, exact certificates, strategy state and output pointers remain
separate. Standalone diagram allocation is unchanged. Standalone Delaunay also
keeps separate scratch unless it belongs to an active constructor family.

For allocated clouds below 32,768 sites, original-coordinate FP64 construction
starts at 256 slots to avoid restarting occasional cells that exceed 128 slots.
Large clouds retain 128 and their FP32 candidate pass. Normalized and symbolic
recovery retain their original schedules. Every input still receives the existing
bounded retries and original-coordinate certification. Historical experiment
drivers explicitly capture their original schedules, including the 128-slot
overflow fixtures.

All existing device-function and kernel bodies are unchanged. Ownership and
capture schedules changed, so the source audit complements execution checks;
it cannot establish graph ordering, output correctness or performance alone.

## Complete solves and storage

NVIDIA L40 / Warp 1.19.0; median five additional fresh captured complete solves.
Setup, transfers, exports and independent validation are separate. Accepted
windows have full CPU quota. Source `8a575f1` is compared with `bfd5448`.
Both include the earlier selective step-history improvement.

Storage is active Warp pool bytes added by the mesher workspace constructor,
excluding the existing domain, compiled code and CUDA graph/driver storage.
It is not total VRAM. Negative time change means less time.

| Case | Before (ms) | After (ms) | Change | Pool bytes saved |
|---|---:|---:|---:|---:|
| torus16384-fixed32 | 2911.16 | 2907.97 | -0.11% | 2,726,297,600 |
| torus1000 | 488.40 | 489.24 | +0.17% | 0 |
| torus8192-fixed32 | 1241.47 | 1240.83 | -0.05% | 1,887,436,800 |
| torus8192-progressive | 1972.48 | 1977.10 | +0.23% | 1,887,436,800 |
| thin-hand-seed7 | 1225.12 | 1223.13 | -0.16% | 0 |
| cavity | 335.57 | 335.50 | -0.02% | 0 |
| disconnected | 478.84 | 477.74 | -0.23% | 0 |
| plate | 786.35 | 786.21 | -0.02% | 0 |
| armadillo-graded4096 | 2072.07 | 2065.85 | -0.30% | 1,048,576,000 |
| armadillo-graded8192-recovery | 3344.30 | 3330.10 | -0.42% | 1,887,436,800 |
| armadillo-uniform8192-recovery | 3459.55 | 3458.17 | -0.04% | 1,887,436,800 |
| armadillo-graded8192-native-mixed | 2302.72 | 2304.76 | +0.09% | 1,887,436,800 |
| armadillo-uniform8192-native-mixed | 2318.74 | 2323.10 | +0.19% | 1,887,436,800 |

All 26 saved meshes pass independent original-coordinate exact CGAL full-hull,
material incidence, boundary vertex-link and boundary self-intersection checks.
Gallery VTU round trips are exact. Roles, canonical full/material connectivity,
oriented boundary and accepted work counts match. Sites are bitwise identical
except the thin hand (maximum coordinate difference 4.404571739312668e-18) and
disconnected components (4.656612873077393e-10). All quality/fit metrics and their
differences remain in the report. Complete-solve time is effectively unchanged;
these are primarily workspace savings, not a new complete-pipeline speedup.

## Independent public routines

A separate 25-case fresh-construction matrix uses actual production capacity
policies with `cache_strategy=False`, nine alternating timing pairs, unchanged
inputs and 50 independently validated outputs. Every pair has equal canonical
connectivity and full CPU quota. The four 100K cases use strict exact CGAL
connectivity comparison; smaller/degenerate cases use exhaustive original-point
empty-sphere and hull-support checks, allowing valid cospherical alternatives.
The symbolic schedule stays at 128/256/512/1024. Large clouds retain the FP32
candidate first pass and the original schedule.

| Fresh Delaunay case | Before (ms) | After (ms) | Change |
|---|---:|---:|---:|
| sphere-8192 | 1654.003 | 1182.374 | -28.51% |
| near-plane-1000 | 30.100 | 20.154 | -33.04% |
| scan-uniform | 107.291 | 91.363 | -14.85% |
| scan-graded | 109.488 | 92.672 | -15.36% |
| uniform-100000 | 182.003 | 182.495 | +0.27% |
| gradient-100000 | 184.671 | 186.056 | +0.75% |
| cluster5-100000 | 265.324 | 263.720 | -0.60% |
| cluster10-100000 | 871.156 | 871.454 | +0.03% |

No case triggers the paired 95% confidence-interval 2% regression gate. The
initial lattice-4096 interval was wide (-3.87% to +5.56%), so a focused follow-up
uses 63 additional alternating pairs with the same source and inputs. Its
independent checks also pass. All original samples and both intervals are kept;
the follow-up does not replace the first run. Constructor times include differing
module warmup and are not setup-speed comparisons. Two earlier matrix launches
failed during alias-module preload before any timing; the final driver preloads
both reachable helper modules in both namespaces.

Five separate controls keep the ordinary cached Delaunay behavior: 1K, 100K,
grid and fitted hand, plus 100K diagrams. Diagnostic counts match; median changes
are +0.29%, +0.38%, +0.23%, -0.07% and -0.21%. The ten-case power/regular-diagram
matrix also passes its 2% gate with equal outputs and unchanged eager allocation
and launch traces. A separate synchronized constructor check covers all six
weighted/unweighted adjacency/tetrahedra combinations at 100K: recorded allocation
requests and pool-byte deltas match exactly. Earlier unsynchronized pool counters
are retained, not used to claim a diagram storage change.

## Captured execution

The 16K complete replay has 47,575 kernel activities. Its inputs, sites, roles,
canonical connectivity, oriented boundary, work counts and history statistics
match the independently validated timed output. The five-phase producer/consumer
loop has 41,714 activities, including changed inputs/targets, invalid input and
valid recovery. Both native traces show only device-to-device transfers, no host
callbacks and no replay allocation APIs. Graph DOT copy nodes are audited because
CUPTI omits many graph copy activities. These checks use the reference surface
backend; optional native-mesh refit/instrumentation remains tracked separately.

Source pins, commands, every timing sample, numerical comparisons, native audit,
memory scopes and all failed/cancelled attempts are indexed in
[experiments/clipping-workspace](../experiments/clipping-workspace/).
Raw meshes and traces are published as evidence assets, not ordinary Git blobs.

The lattice follow-up median is 2531.580 → 2494.925 ms; its paired relative-change 95% interval is -3.93% to -0.33%.


## Memory-check scope

All eight selected contracts pass padded memory checking and use-after-free
tracking: concurrent independent families at both cell budgets, changed 16K
prefixes, fallback, capacity exhaustion, one-ULP inputs, rollback and symbolic
duplicates. Four contracts also pass full allocation-order tracking.

Four larger conditional-sort tests retain native CUB access-before-allocation
reports under full tracking, followed by an instrumented crash. Each reproduces
on the unchanged baseline. A standalone Warp sort without Warpower geometry
reproduces the conditional-capture report; top-level full tracking and
conditional use-after-free tracking pass. This is the existing #91 limitation,
not a green full-tracking result or a demonstrated explanation of the diagnostic
internals. Failed logs, exact scopes, baseline controls and SDK hashes are retained.
The runtime allocator and geometry checks are unchanged.

The ordinary suite passes 1,466 tests and 33 subtests. Instrumentation uses an
explicit diagnostic stack allowance; production does not change the CUDA stack.
The optional native-surface refit limitations in #267/#269 remain separate.

The isolated allocator-hook pilot is documented in
[shared clipping scratch](shared-clipping-scratch-pilot.md). The earlier capacity
experiment and rejected large-cloud variant are documented in
[capacity schedules](capacity-schedule.md). Their measurements are distinct from
this production qualification. This work does not establish a CPU meshing speedup.
