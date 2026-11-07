# Representable boundary updates and selective backtracking

An exact-midpoint proposal removes a real boundary-fitting stall through the
normal optimizer acceptance path. It also exposes a broader convergence problem:
raising the current minimum quality can make the selective guard shrink every
site's step, slowing feature fitting substantially. **The proposal is not adopted.**
Production kernels, defaults, independent geometry APIs and README are unchanged.

This investigation uses runtime `6f7efc9f1515a18077053d23f3a306c4a6cdc47a`,
NVIDIA L40 and Warp 1.19. It advances
[#117](https://github.com/alecjacobson/warpower/issues/117) and identifies the
localized-backtracking investigation in
[#202](https://github.com/alecjacobson/warpower/issues/202). It does not establish
comparable-quality CPU outperformance or complete coarse-hand recovery.

## Proposal and lifetime

The existing fitter stores float32 centroid targets. A half-damped update toward
an adjacent target can land exactly halfway between two representable values;
nearest-even conversion can return the old coordinate indefinitely.

The private proposal first computes the ordinary rounded point. A component can
instead take its existing boundary target only when the ordinary result equals
the original coordinate and the float64 requested value is exactly midway
between that coordinate and the target. Finite checks apply. The complete
candidate vector must fit within the original `step_limit * shortest_edge` cap;
otherwise all additional midpoint corrections are discarded. Smaller steps that
do not reach the midpoint keep ordinary rounding. Corners, interior directions,
inactive entries and zero selective directions receive no correction.

Starting boundary targets are frozen once per optimizer step, with original
positions substituted for non-boundary roles. This matters because candidate
evaluation recomputes and mutates the ordinary targets and roles during
backtracking. The frozen buffer remains valid for that step's retries and is
refreshed at the next step. The prototype uses 12 bytes per site for these targets,
plus separate diagnostic flags/counters. Every candidate still goes through
ordinary reconstruction, energy acceptance, selective quality checks and rollback.

The paper uses sampled centroid approximations for surface and crease updates
in §3.5. This probe preserves that objective; it does not add projection or
constrained-edge conformity. [Alliez et al., paper](https://authors.library.caltech.edu/records/c9exj-3ed14/files/ACSYD.pdf?download=1).

## Accepted transactions from fitted snapshots

Each pair starts from the same retained point set and executes ten ordinary
global core iterations, then ordinary extraction. These are continuation tests,
not fresh complete solves or timing measurements.

| Starting snapshot | Baseline minimum ratio | Candidate minimum ratio | Accepted corrected components | Material validity |
|---|---:|---:|---:|---|
| old coarse hand | 0.000000280624 | 0.108303 | 1,067 | invalid in both |
| current dense hand | 0.056169227 | 0.056169218 | 1,251 | valid in both |
| torus 1k | 0.062846807 | 0.062846807 | 0 | valid in both |

All six full-hull meshes pass independent CGAL global empty-sphere, orientation,
face and convex-hull checks. Independent NumPy element measurements agree with
the recorded minima. The old hand retains 19 nonmanifold boundary edges in both
continuations. Removing numerical stalls alone does not fix its material topology.

## Fresh complete solves

Eight fresh gallery pilots compare the baseline and proposal on uniform torus,
dense hand (20/20 quadrature), coarse hand (4/4), and counterbored plate (16/16).
All use 1,000 sites, 256 initial sites, seed 2026 and four quality rounds. The
candidate includes diagnostic atomics and acceptance counters, so these runs are
not production speed evidence. Every outcome and timing record is retained.

The six valid outputs pass independent CGAL, NumPy, AABB sampled-distance and
boundary-link checks, exact VTU round-trips and zero checked boundary
self-intersections. Both coarse-hand runs fail explicitly with error 64 and
invalid output; their states are retained for diagnosis.

| Example | Minimum ratio before → after | Two-sided sampled RMS, % of box diagonal before → after | Result |
|---|---:|---:|---|
| torus | 0.0928581 → 0.0928581 | 0.256233 → 0.256233 | positions and topology bitwise equivalent |
| dense hand | 0.0935229 → 0.0961569 | 0.0344620 → 0.0488730 | valid, worse lower percentiles and fit |
| plate | 0.1027469 → 0.1027468 | 0.0870176 → 0.0870176 | same material connectivity and boundary |

The plate's full-hull cell count falls from 5,458 to 5,049, while its 3,951
material cells and 1,338 boundary faces match. Its largest coordinate change is
1.192e-7. This does not improve the worst-case crease-fitting limitation in #149:
the largest crease-site deviation remains approximately 0.06576.

The hand's length-weighted, sampled input-crease-to-output-surface RMS worsens
from 0.003220 to 0.004447, and its maximum sample distance from 0.04183 to 0.06589.
Its 95th-percentile crease-site deviation improves, but its worst deviation rises
from 0.04706 to 0.08382. These are sampling/assignment diagnostics, not Hausdorff
bounds or proofs of constrained-edge conformity. Full per-chain records are
retained, including exact corner checks. A better minimum alone is insufficient
evidence for adoption.

## Where the hand trajectories diverge

Read-only device observers record complete-solve positions, targets, roles,
physical diagnostics and optimization work after extraction and outer quality
acceptance/rollback. A second version additionally records every selective
proposal's step size, quality minima, guard status and selected-site data.
The observed runs preserve final counts, rounds, steps, moves and roles. Small
coordinate reduction differences relative to the unobserved pilots are recorded
and bounded below 1e-8; bitwise identity is not claimed. No instrumented timing
is used as a performance result.

Immediately before the first selective round, fingertip site 403 has the same
position and boundary target in both runs. It is selected in every recorded
attempt. The current material minima differ because earlier midpoint corrections
remove very poor cells.

| First selective round | Baseline | Midpoint candidate |
|---|---:|---:|
| starting minimum ratio | 0.00000103098 | 0.0492533 |
| first trial's minimum | 0.00000103098 | 0.00453199 |
| accepted iterations | 10 | 10 |
| attempted reconstructions | 10 | 110 |
| accepted step size, each iteration | 0.5 | 0.00048828125 |
| quality rejections | 0 | 100 |

Every rejected candidate in this round reports quality reason bit 4 and a
minimum below its saved threshold. The accepted step is 1,024 times smaller.
The baseline brings site 403 close to its fingertip target; the candidate barely
advances. Later selective/jitter rounds do not recover the lost fit. This
attributes the observed slowdown in fitting to the shared step and minimum
guard, rather than an omitted selection or missing target. It does not prove
that the limiting cells are spatially independent of site 403.

Issue #202 will first identify the limiting cells and proposal regions, then
test localized damping with complete current-input certification and unchanged
acceptance conditions. Global backtracking and rollback remain necessary when
local retries are ineffective or unsupported. Neither a weaker quality guard
nor a surface-projection replacement is justified by this probe.

## Tests, failures and records

The private numerical suite passes 18 cases spanning FMA settings, scales from
`2^-140` to `2^120`, adjacent values, midpoint neighbors, exact/insufficient caps,
smaller steps and nonfinite inputs. Five real transaction tests cover accepted
progress, capped steps, forced geometry rejection/rollback, frozen targets,
active prefixes, selective masks, captured failure and recovery. Forty-four
existing core, active, selective and device-control tests also pass with the
private proposal installed, including forbidden-host-action capture checks.

The first transaction exporter used incorrect boundary-buffer attribute names
and failed after execution. Its source/log are retained; the corrected exporter
changes only those names and explicit module-loading device selection. Three
initial transaction tests also incorrectly required the exact triangulation
chosen by SciPy for a cospherical fixture. They are retained. The revised tests
check actual geometry and use the independent CGAL global certificate; all five
pass. No runtime change was made to satisfy those comparisons.

The [artifact manifest](../results/validation/vtm/representable-boundary-probe/manifest.json)
records sources, commands, valid and invalid meshes, physical/feature checks,
observer arrays, hashes and failed runs. This is an experimental evidence bundle.
No adoption full-suite, standalone performance, native profiler or sanitizer gate
is claimed for the proposal. The production implementation remains the tested
runtime from PR #201.
