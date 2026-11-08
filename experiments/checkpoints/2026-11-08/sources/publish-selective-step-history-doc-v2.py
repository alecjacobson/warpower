"""Write the selective-history report from completed production qualification."""
import json
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm'
read=lambda n:json.loads((w/n).read_text())
s=read('selective-step-history-production-source-v1.json');proof=read('selective-step-history-final-v1-independent.json');assert proof['complete']
q=read('selective-step-history-qualified-v1.json');assert len(q)==6 and all(r['exit_code']==0 for r in q)
f=read('selective-step-history-feature-fit-v1.json');assert f['complete']
text=f'''# Bounded selective step history

The complete mesher uses the previous accepted selective step size to avoid
repeating large proposals after quality-only rejections. Every actual proposal
still passes the original Delaunay, objective, domain and minimum-quality checks.
The policy changes the order of trials; it does not guarantee an identical
trajectory or a globally optimal step.

## Search and safeguards

Each core invocation starts without history. An eligible later selective step
starts at `min(damping, 2 * previous_accepted_alpha)`, using the same absolute
halving levels and minimum trial scale as the original search. A search containing
an objective or geometry failure cannot seed the next step. Rejection counters
must advance before a sticky quality reason is treated as fresh evidence.

An apparently converged warm trial is discarded and the complete default bracket
is searched before convergence can be accepted. Exhausting a warm bracket also
retries the default bracket once. A non-quality warm rejection triggers that
retry immediately. The default retry cannot restart again. This retains a finite
search even if acceptance is not monotone in step size. Counts include actual
rejected proposals, not skipped scales; the convergence-verification trial is
neither an accepted iteration nor a rejected proposal.

Six history integers and five statistics integers add 44 device bytes. History
resets for each core invocation, statistics for each fresh complete solve. The
complete mesher enables the policy only when selective stages are configured and
local vertex rollback is disabled. Independently callable power-diagram,
Delaunay and default VTM-core entry points keep their existing paths. Optional
local rollback retains its original scheduling. All replay decisions are device
kernels and conditional graph nodes; setup allocates the fixed buffers.

The source audit covers {len(s['source_sha256'])} files: {s['unchanged_files']} are
unchanged. The original scalar start/finish bodies were factored into shared Warp
functions without changing their AST. Directions, proposals and acceptance remain
unchanged; the new policy AST matches the independently validated v2 prototype.

## Complete measurements

NVIDIA L40, Warp 1.19.0+cu12, Python 3.10. Baseline `{s['baseline']}`;
runtime `{s['runtime']}`. Each warmed replay performs a fresh complete solve,
including domain refit, sampling, growth, optimization, quality work, recovery and
extraction. Setup, transfers and offline validation are outside replay. All 32
complete/standalone jobs have clean recorded CPU-quota windows. These compare GPU
revisions; they do not establish the wide comparable-quality CPU speedup goal.

| Complete example | Before, ms | After, ms | Time reduction |
|---|---:|---:|---:|
'''
for name,r in proof['cases'].items():
 t=r['timing'];text+=f"| {name} | {t['before_ms']:.3f} | {t['after_ms']:.3f} | {100*t['reduction_fraction']:.2f}% |\n"
text+='\nPositive reduction means faster; negative values retain observed regressions.\nEach paired workspace adds exactly 44 device bytes.\n\n'
text+='''## Geometry and quality

All 22 final meshes pass independent full-hull orientation, empty-sphere and
coverage checks, material orientation/manifold checks, single-cycle boundary
vertex links and exact boundary self-intersection checks. Prior proofs transfer
only through exact input/coordinate/oriented-connectivity/aligned-quality
comparisons. Otherwise the mesh receives fresh validation. All fourteen gallery
VTU/OFF exports round-trip exactly. Surface fit uses sampled quadrature, not a
Hausdorff bound. The scan retains its [Stanford/libigl provenance](scanned-armadillo.md).

| Changed coordinates | Maximum coordinate difference | Same full-hull cells | Minimum quality difference | Surface RMS difference |
|---|---:|---|---:|---:|
'''
for name,r in proof['cases'].items():
 c=r['comparison'];quality=r['quality_comparison']
 if not c['points_bitwise_equal'] or not c['full_tets_equal']:
  text+=f"| {name} | {c['maximum_coordinate_difference']:.8g} | {c['full_tets_equal']} | {quality['radius_ratio_percentiles']['0']['difference']:+.8g} | {quality['two_sided_area_weighted_rms']['difference']:+.8g} |\n"
text+='\nThe report retains every quality percentile, dihedral extreme, mean quality,\nvolume error, work count and fit difference for both outputs. Geometry variation\nis observed; no causal attribution to a particular floating-point kernel is made.\n\n'
for name,r in f['cases'].items():
 text+=f"For {name}, exact corner records match: {r['exact_corners_equal']}; chain diagnostics match: {r['chain_diagnostics_equal']}. Feature-to-boundary sampled RMS changes by {r['rms_difference']:+.8g}, maximum sample distance by {r['maximum_sample_distance_difference']:+.8g}.\n\n"
text+='''Nearest-chain assignment and connectivity gaps are diagnostics, not a guarantee
of curve-conforming mesh edges. They use the existing offline feature measurement.

| Independent routine/input | Before, ms | After, ms | Change |
|---|---:|---:|---:|
'''
for name,r in proof['standalone'].items():text+=f"| {name} | {r['before_ms']:.3f} | {r['after_ms']:.3f} | {100*r['relative_change']:+.2f}% |\n"
text+='''
All recorded standalone inputs, counts, errors and diagnostics match. Timing
variation does not establish a standalone speedup. README keeps installation
instructions and the earned Paragram flagship.

## Rejected version and qualification

The first prototype used recent alpha regardless of failure type. Its uniform8K
scan was 4.98% slower, so it was rejected. The quality-gated prototype removes
that regression; this is a failure-type rule rather than a dataset-specific gate.
Both versions, their outputs, and the initially failed checker invocation are
retained. The preceding [observation study](vtm-step-observation.md) found cases
where larger-than-predicted steps succeeded, motivating the full-bracket fallback.

The production suite passes 1,407 tests, followed by the unchanged five-phase 8K
producer/mesher/consumer recovery loop. Targeted integration tests cover eligibility,
fresh replay resets, sticky failures, convergence verification, bounded fallback,
original global controls and local rollback. Four isolated memory audits exercise
37 targeted contracts and the full 8K loop with padding and with the allocation
pool disabled plus lifetime checking; all report zero memory errors.

Native traces cover the exercised 16K history path, a complete uniform8K solve,
and the five-phase changing-input loop. Each measured region has one outer graph
launch and one explicit end synchronization, device-to-device copies only, and
no host callback or replay allocation/free. Native and sanitizer output frames
are bound to independently certified geometry. Profiling timings are diagnostic,
not the clean performance measurements above.

Raw evidence is distributed as a release bundle, keeping binary snapshots and
traces out of ordinary Git. The manifest records its download URL and SHA-256.
The bundle restores the original `.work/vtm` files and ignored `results/local`
aliases; extract it from the repository root after verifying the checksum.

Reproduce using the pinned revisions and the commands/environment in the
[immutable manifest](../results/validation/vtm/selective-step-history/manifest.json).
Restore shared proof/input dependencies, use fresh output destinations, run one
GPU job at a time, and run CPU geometry checks after clean timings. Native
validators are described in [benchmarks/cpu](../benchmarks/cpu/README.md).
The [verification result](../results/validation/vtm/selective-step-history/premerge-integrity.json)
binds the files, sources, tests, geometry, graph and memory evidence. Full matched
CPU comparison and remaining robustness/performance investigations remain tracked
in [the implementation goal](https://github.com/alecjacobson/warpower/issues/29).
'''
p=root/'docs/selective-step-history.md';assert not p.exists();p.write_text(text)
