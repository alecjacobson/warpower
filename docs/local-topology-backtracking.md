# Optional local rollback for material topology failures

`VariationalMeshingWorkspace(..., local_topology_retries=1)` enables a bounded
retry that freezes moved endpoints of current invalid material edges. The default
is zero. It adds no arrays or graph operations when disabled, and standalone power
diagrams and Delaunay workspaces do not import the helper.

Every proposal still reconstructs the current full hull and runs the ordinary
objective, boundary, material-selection, topology and quality checks. A fresh
recovered-selection rejection supplies the current edge witnesses; prior error
flags cannot trigger retries. Only moved, active, selection-eligible noncorner
sites can be frozen. No material labels change. The budget applies per selective
iteration. Exhaustion, no new eligible endpoint or another rejection restores the
ordinary global search and eventual full rollback. Topology retries invalidate
the quality-only step-history prediction.

The option requires positive quality rounds and material recovery. This initial
endpoint policy cannot be combined with `local_quality_retries`; specifying both
raises an error rather than silently disabling one. The broader endpoint vertex
star remains a benchmark option. Use `--local-topology-retries 1` in the gallery
or complete-mesher benchmark to select the optional policy.

Four device counters at `mesh.core.local_backtracking.statistics` record local
retries, blocked-site events, fresh eligible topology obstructions and accepts
immediately following a local retry. The complete mesher resets them at the start
of every captured solve, including invalid-input replays. Core calls reset the
per-iteration retry state without discarding those complete-solve totals.

The [benchmark study](selective-topology-rollback.md) found a modest uniform-scan
speed improvement and a small worst-element-quality decrease, while mean quality
and sampled fit improved slightly. It did not help the graded scan. This is why
the policy is optional. The production qualification below measures this integration separately from
the historical experimental timings.

## Production qualification

Runtime `c35583b` is compared with `d7d6675` on NVIDIA L40 / Warp 1.19.0.
Each complete solve includes fresh sampling, growth, fitting, quality work and
extraction. Numbers are medians of five additional captured solves; setup,
transfers, export and independent checks are outside timing. Each accepted
timing window has full CPU quota. Historical pilot data are not substituted.
Three affected triplets were repeated after CPU quota interruptions in the first
batch (hand setup, native/mixed uniform first replay and graded timed replay).
The original runs and the first incomplete validation attempt remain in the
archive; the table uses the clean replacement triplets. The four completed
independent torus triplet checks are retained with exact mesh-file hash binding.

| Case | Baseline (ms) | Disabled (ms) | Enabled (ms) | Enabled change | Local retries |
|---|---:|---:|---:|---:|---:|
| torus16384-fixed32 | 2909.773 | 2917.769 | 2914.429 | +0.160% | 0 |
| torus1000 | 488.836 | 487.419 | 490.170 | +0.273% | 0 |
| torus8192-fixed32 | 1241.249 | 1241.429 | 1244.319 | +0.247% | 0 |
| torus8192-progressive | 1975.835 | 1978.337 | 1977.806 | +0.100% | 0 |
| thin-hand-seed7 | 1224.475 | 1225.168 | 1227.837 | +0.275% | 0 |
| cavity | 335.175 | 335.826 | 335.947 | +0.230% | 0 |
| disconnected | 478.582 | 478.559 | 480.311 | +0.361% | 0 |
| plate | 785.886 | 786.182 | 788.691 | +0.357% | 0 |
| armadillo-graded4096 | 2065.481 | 2067.114 | 2072.417 | +0.336% | 0 |
| armadillo-graded8192-recovery | 3337.595 | 3338.073 | 3337.529 | -0.002% | 0 |
| armadillo-uniform8192-recovery | 3458.755 | 3458.823 | 3350.605 | -3.127% | 6 |
| armadillo-graded8192-native-mixed | 2305.905 | 2306.291 | 2310.986 | +0.220% | 0 |
| armadillo-uniform8192-native-mixed | 2325.655 | 2322.646 | 2218.962 | -4.588% | 6 |

Disabled-path timing changes range from -0.290% to
+0.275% in this gallery. Synchronized constructor accounting
finds exactly equal baseline/disabled workspace pool bytes in all thirteen cases.
Enabling the option adds `8 * capacity + 36` device-array bytes in these cases
(65,572 bytes at 8K). These are active Warp pool bytes, not total process VRAM.
The ten-case public power/regular-diagram regression gate passes its paired 2%
criterion, with equal outputs and unchanged eager launch/allocation traces.

All 39 final meshes pass independent original-coordinate exact CGAL full-hull
checks, material incidence, boundary vertex links and exact boundary intersection
checks. Gallery VTU exports round-trip exactly. The evidence retains full
coordinate/connectivity comparisons and quality/fit distributions. The disabled
disconnected control differs in one coordinate by 4.656612873077393e-10, with
identical roles and canonical/oriented connectivity. The enabled hand differs
by at most 3.469446951953614e-18 with equal connectivity. Other inactive-policy
cases are bitwise identical. Floating-point atomic reduction order is not
promised to be identical across runs.

For armadillo-uniform8192-recovery, minimum radius ratio is 0.0310191823605 → 0.0310047215082 (-0.0466% relative), mean radius ratio 0.874565022 → 0.874619693, and sampled two-sided RMS fit is 0.147990386% → 0.147975880% of the input bounding-box diagonal. Selective step counts are 2 → 4.

For armadillo-uniform8192-native-mixed, minimum radius ratio is 0.0310191823605 → 0.0310047215082 (-0.0466% relative), mean radius ratio 0.874565024 → 0.874619693, and sampled two-sided RMS fit is 0.147990384% → 0.147975880% of the input bounding-box diagonal. Selective step counts are 3 → 4.

The full ordinary suite passes 1,498 tests and 33 subtests. A focused integration
check also passes after merging the two benchmark studies; runtime, examples and
the main README still match the tested runtime revision exactly. All six isolated
production memory processes pass with zero errors: fresh-witness contracts,
forced budget-two rollback/recovery and a complete five-phase changing-input
producer/mesher/consumer cycle, each under padded and full allocation-order
tracking. Diagnostic stack/pool settings are recorded and do not change runtime
defaults. The historical experimental ten-replay timeout remains incomplete;
these specifically scoped production passes do not relabel it as passed.
The actual 8K reference scan additionally passes padded and use-after-free
checking, with six local retries and exact output/work binding to the independently
validated timed mesh. This larger case is not claimed to pass full allocation-order
tracking; the existing conditional-sort limitation remains separately documented.

Native production audits cover a reference-backend 8K scan that actually retries
locally and the changing-input loop. They contain only device-to-device copies,
with no host callbacks or replay allocation APIs. The scan trace is bound by
exact inputs/sites/roles, canonical connectivity, oriented boundary and work
counts to its independently validated timed output. Optional native Mesh refit
limitations (#267/#269) remain separately tracked.

This option does not fix the 45-degree default Armadillo case: all six current
1K/8K seed diagnostics still reject final material topology before selective
optimization. There is no comparable-quality CPU speedup claim. Detailed
measurements and raw checksums are in the
[evidence index](../experiments/local-topology-backtracking/).
