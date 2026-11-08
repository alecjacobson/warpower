"""Append complete production time/quality results, including negative cases."""
import json
from pathlib import Path
w=Path('.work/vtm');r=json.loads((w/'local-backtracking-final-independent-v2.json').read_text());p=Path('docs/local-quality-backtracking.md')
text='''
## Complete-solve measurements

Runtime `6aa4ba6` was compared with baseline `39f8f54` on NVIDIA L40 / Warp 1.19.
Each number is the median of three additional **complete fresh replays**, including
domain/sizing refits, sampling, growth, optimization, recovery, extraction and
final diagnostics. Setup, upload, download and independent validation are outside
these times; first-replay timings are retained separately. All recorded first and
repeated timings have clean CPU quota counters. This is not CPU outperformance
evidence.

All rows below use four quality rounds and seed 2026 unless noted. Enabled runs
use `local_quality_retries=12`. Surface RMS is independently sampled in both
directions and expressed as a percentage of the input bounding-box diagonal;
it is not a Hausdorff bound. The full percentile distributions, volumes,
dihedrals, topology checks and per-feature records accompany each result.

| Example | Time, ms default → enabled | Minimum radius ratio default → enabled | Two-sided RMS, % default → enabled |
|---|---:|---:|---:|
'''
labels={'torus1000':'torus 1k','torus4096':'torus 4k','graded-torus':'graded torus 4k','thin-hand':'dense hand 1k','plate':'plate 1k','cavity':'cavity 1k','disconnected':'disconnected 1k','armadillo-uniform':'uniform Armadillo 4k','armadillo-graded':'graded Armadillo 4k','plate-seed7':'plate 1k, seed 7'}
for name,label in labels.items():
 a,b=[r['cases'][name][k] for k in ['before','enabled']];qa,qb=a['quality'],b['quality']
 text+=f"| {label} | {a['median_ms']:.2f} → {b['median_ms']:.2f} | {qa['radius_ratio_percentiles']['0']:.5f} → {qb['radius_ratio_percentiles']['0']:.5f} | {qa['two_sided_area_weighted_rms_percent_bbox_diagonal']:.5f} → {qb['two_sided_area_weighted_rms_percent_bbox_diagonal']:.5f} |\n"
text+='''
The cavity becomes slower with a worse minimum. Uniform Armadillo becomes faster
but has a worse minimum and slightly worse fit. The plate's alternate seed trades
some surface accuracy for better minimum quality and time. Equal round budgets
also do not imply equal optimization work: local rollback can continue after a
global step would have stalled. These are reasons to keep activation explicit.

### Feature fitting

Corners match exactly in the checked hand and plate outputs. Crease fitting is
approximate, and surface RMS alone does not describe it. Length-weighted sampled
input-crease-to-output-surface RMS changes as follows (model units):

| Example | Crease RMS default → enabled | Worst crease-site deviation default → enabled |
|---|---:|---:|
'''
for name in ['thin-hand','plate','plate-seed7']:
 a,b=[r['cases'][name][k]['features'] for k in ['before','enabled']]
 text+=f"| {labels[name]} | {a['feature_to_boundary']['rms']:.7f} → {b['feature_to_boundary']['rms']:.7f} | {a['crease_site_distance_percentiles']['100']:.7f} → {b['crease_site_distance_percentiles']['100']:.7f} |\n"
text+='''
All three sampled crease RMS values worsen, and seed 7's largest crease-site
deviation rises substantially. These measurements include per-chain records and
maximum sample distances; they do not establish constrained-edge conformity.
The option does not resolve [#149](https://github.com/alecjacobson/warpower/issues/149).

### Time to quality

The torus sweep retains every tested budget: one, two and four rounds.
Two enabled rounds take **444.84 ms**, with minimum ratio **0.11936** and sampled
surface RMS **0.24097%** of the diagonal. Four default rounds take **608.98 ms**,
with minimum **0.09286** and RMS **0.25623%**. That is 27% less replay time at better
measured quality/fit for this example. One enabled round has a worse minimum than
one default round, while four enabled rounds cost more than four default rounds.

![All tested torus round budgets](../results/validation/vtm/local-quality-backtracking/local-backtracking-quality-frontier-v1.png)

### Failure, compatibility and provenance

Both dense-hand seed-7 runs fail before selective optimization with error 64 and
zero public counts. They reach all 1,000 sites, but the material-boundary validity
gate rejects them after three recovery rounds. Both coarse-hand 4/4-quadrature
runs also fail explicitly. These four results are retained and are **not** valid
mesh performance results. The seed failure stopped the first harness; the resumed
harness preserves all earlier outputs and records. It is tracked separately in
[#205](https://github.com/alecjacobson/warpower/issues/205).

Three disabled-option/default pairs preserve counts, topology, roles and all
recorded optimizer work. Torus and plate positions match bitwise; the hand's
largest coordinate difference is 1.63e-19. Median time changes span -0.06% to
+0.79%. Five independent diagram/Delaunay pairs preserve recorded inputs/outputs,
counts, statuses and diagnostics, with median changes from -0.29% to +0.75%.
They do not allocate local-retry state, and their source/defaults are unchanged.

All 27 valid production meshes pass independent CGAL global empty-sphere,
orientation, hull/face checks, NumPy physical measurements, sampled AABB surface
distances, boundary vertex-link checks, exact VTU round-trips and checked boundary
self-intersections. The full suite passes 1,034 tests; the focused suite passes
46, including bounded rejection, stale reasons, captured producer/consumer
failure recovery, active prefixes, grading and exact corner anchors.

The [artifact manifest](../results/validation/vtm/local-quality-backtracking/manifest.json)
records all experimental policies, failure logs, production runs, source hashes,
independent checks, and the lower-round sweep. The private experimental test that
expected exactly three rejections fails because local retries add attempts; the
new production tests validate the explicit combined bound and complete rollback.
An initial private recovery test incorrectly required acceptance with an
insufficient budget; it was corrected to compare recovery with a fresh identical
workspace. The original failed runs remain in the bundle. No midpoint correction
or relaxed quality acceptance was adopted.
'''
assert '## Complete-solve measurements' not in p.read_text();p.write_text(p.read_text()+text)
