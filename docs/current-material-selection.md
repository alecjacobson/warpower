# Reusing an accepted material evaluation

After a successful selective VTM trial, the next optimizer step consumes the
same points, full tetrahedra and material configuration that the trial's quality
check just classified. That step now reuses its keep mask, per-cell quality,
minimum and retained count. The reuse also preserves the current recovery
extraction metadata when material recovery is enabled.

The lifetime is restricted to the optimizer's uninterrupted internal schedule.
A fresh public `run()` resets the flag before anchoring or reconstruction.
Every trial proposal and rollback invalidates it before modifying geometry.
Select-all transitions and explicit `select()` or `evaluate()` calls invalidate
it too. Every trial quality check recomputes; only an error-free, accepted check
marks the result reusable. The following `_accept` updates ODT statistics and
the control gate may stop the loop; neither changes classification inputs or
outputs. Selection still rebuilds the bad-vertex mask, one-ring mask and previous
minimum normally.

All decisions use a preallocated device integer and captured conditions. There
is no host input comparison, synchronization, allocation or readback in the
solve. Producers may change inputs, domain and controls before the next public
run, which resets the flag. The private `_select_current` method is not a general
cache API for arbitrary external writes. Independently callable power-diagram,
regular-triangulation and Delaunay functions retain their original code and
storage. The cache adds one four-byte integer to each selective workspace.

## Complete solves

Fresh-process pairs compare merged baseline `80d6ded503ac` with runtime
`0b67386c59b2` on an NVIDIA L40 and Warp 1.19. Each replay starts a complete
fresh solve, including refits, sampling, progressive growth, optimization,
quality work and extraction. Torus 1k uses five warmed samples; the other
seven cases use three. Setup, compilation and offline validation are excluded.
Separate setup and first-call measurements are retained. All first and timed
CPU-quota windows are clean.

| Example | Before ms | After ms | Reduction |
|---|---:|---:|---:|
| torus1000 | 623.076 | 617.123 | 0.96% |
| torus4096 | 1833.717 | 1814.513 | 1.05% |
| graded-torus | 2103.133 | 2080.223 | 1.09% |
| thin-hand | 1342.760 | 1333.578 | 0.68% |
| cavity | 442.638 | 441.815 | 0.19% |
| disconnected | 498.623 | 498.456 | 0.03% |
| armadillo-uniform | 2998.989 | 2959.402 | 1.32% |
| armadillo-graded | 6582.537 | 5289.101 | 19.65% |

The disconnected case had no reuse opportunities in the preceding observer;
its tiny timing difference is ordinary variation. The graded scan's reduction
is explained by the separate native profiles below, not by a change in solver
work or output quality.

All eight pairs have identical input arrays, parameters, iteration/refinement
work, full/material tetrahedron sets, oriented boundary faces and roles.
Seven point arrays are bitwise identical. Thin-hand differs in one coordinate
by 7.996e-19; the exact values are retained in the
[paired independent checks](../results/validation/vtm/current-selection/current-selection-final-independent.json).
Every output passes independent CGAL global empty-sphere, orientation, face and
convex-hull checks, NumPy quality measurements, CGAL AABB sampled surface distances,
and boundary vertex-link checks. The six gallery VTU files round-trip exactly
and their boundaries have zero checked self-intersection pairs. Sampled surface
distances are not a Hausdorff certificate.

The measured torus workspace grows by exactly four bytes at both 1k and 4k.
Setup and first-call measurements remain separate from replay timings.

## Independent APIs

| Routine/input | Before ms | After ms | Change |
|---|---:|---:|---:|
| delaunay1000 | 9.169 | 9.176 | +0.07% |
| delaunay100000 | 183.206 | 182.009 | -0.65% |
| delaunay-grid | 2.206 | 2.196 | -0.44% |
| delaunay-fitted-hand | 13.702 | 13.463 | -1.74% |
| diagram100000 | 18.856 | 18.220 | -3.37% |

These source-unchanged routines have identical recorded output counts, input
hashes, status and diagnostics. Their timing variation, including the faster
100k diagram result, is not attributed to this optimization. No slowdown above
0.08% was observed in these pairs. The main README keeps its existing Paragram
flagship result.

## Validation and graph lifetime

The full suite passes **851 tests** and the focused selection/control/recovery
suite passes **46 tests**. Seven new tests compare consumed cache results with
forced fresh evaluation, exercise changed device geometry and domain data,
ordinary/material-recovery modes, invalid inputs and recovery, direct calls,
select-all/reset transitions and rejected quality checks. Existing rollback,
active-prefix and control tests also pass.

The initial production test adaptation misspelled the monkeypatch target;
it reported one failed and 45 passed tests. Its source and log are retained
alongside the corrected run. Earlier private test harness failures (unsupported
`float('nan')` in a Warp kernel and an optional routing-report field assumed
present) are likewise preserved. These are not counted as successful checks.

Four focused Compute Sanitizer jobs pass with zero reported errors: seven new
cache/oracle tests and three control/rollback/recovery tests, each with padded
allocations and again with the pool disabled plus API-access/use-after-free
checking. The diagnostic stack is 128 KiB. This is focused memory coverage,
not a claim that the entire 851-test suite ran under the sanitizer; the broader
allocation-order limitation tracked in #147/#155 is unchanged.

The native full-solve trace records 62,741 kernel activities and 38,086 kernel
nodes. The separate eight-phase producer/domain-refit/VTM/consumer graph records
10,054 activities and 1,618 kernel nodes, with 28 cache hits among 34 selection
calls. Its five valid phases match separately recorded forced-fresh positions
and connectivity; three invalid budget/mode/point phases publish zero cells,
and subsequent phases recover. Both captures include first native graph
instantiation/upload, one graph launch and one final audit synchronization.
Neither contains host transfers/callbacks, graph allocation/free nodes or
explicit allocation/free API calls. All 3,652 full-graph and 174 loop-graph copy
nodes are device-to-device. Nsight can omit graph copy activities, so both DOT
node kinds and observed copies were checked.

## Why the graded scan benefits more

Separate first-launch native profiles use the exact graded Armadillo gallery
parameters. They preserve counts `[4096,24667,18746,3618]`, steps `[115,40]` and
moves `[4,0]`. They are structural evidence, separate from the clean warmed
performance pairs above.

| Kernel | Before calls | After calls | Before activity ms | After activity ms |
|---|---:|---:|---:|---:|
| classify | 146 | 110 | 3257.623 | 2450.768 |
| surface_error | 146 | 110 | 1877.307 | 1424.468 |
| accumulate_samples | 466 | 466 | 386.466 | 386.225 |
| check_groups | 219 | 219 | 127.092 | 127.090 |

Removing 36 duplicate complete recovery evaluations saves about 807 ms of
classification activity and 453 ms of boundary-distance activity. Reconstruction
and boundary fitting work stay the same. Ordinary torus selection is cheaper:
the corresponding full trace reduces `classify_retained` calls from 152 to 141.
The new acceptance-flag kernel costs 0.171 ms across 129 calls in that trace;
flag fills/conditional scheduling also add overhead, included in complete timing.

The remaining unconsumed recovery diagnostics are tracked in
[#192](https://github.com/alecjacobson/warpower/issues/192);
this change keeps every fresh extraction and error check intact. CPU
outperformance at comparable quality remains unachieved (#40): the 4k torus
still takes about 1.815 s, and the earlier native CPU baselines also have better
lower-percentile/surface quality. This milestone compares Warp revisions.

The [artifact manifest](../results/validation/vtm/current-selection/manifest.json)
records exact commands, sources, raw metrics, final meshes, reference-tool
hashes, memory logs, native traces and initial failed harnesses. Earlier private
observer/pilot meshes are retained with hashes; their final production
counterparts are published. The authoritative Nsight and DOT archives are
preserved; SQLite is a regenerable export described in the analysis scripts.
The private prototype/observer launchers require their recorded baseline
revision, rather than the new production class.

