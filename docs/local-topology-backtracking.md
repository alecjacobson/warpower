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
or complete-mesher benchmark to select the production candidate.

Four device counters at `mesh.core.local_backtracking.statistics` record local
retries, blocked-site events, fresh eligible topology obstructions and accepts
immediately following a local retry. The complete mesher resets them at the start
of every captured solve, including invalid-input replays. Core calls reset the
per-iteration retry state without discarding those complete-solve totals.

The [benchmark study](selective-topology-rollback.md) found a modest uniform-scan
speed improvement and a small worst-element-quality decrease, while mean quality
and sampled fit improved slightly. It did not help the graded scan. This is why
the policy is optional. Historical experimental timings are not qualification of
this production integration. Production validation, source audits and paired
performance measurements are pending; do not promote this candidate yet.
