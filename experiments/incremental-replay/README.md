# Incremental replay: first infrastructure milestone

This is exploratory evidence for [issue #297](https://github.com/alecjacobson/warpower/issues/297),
using the standalone [replay harness](../../docs/incremental-replay.md). The
production VTM implementation, its complete exact/symbolic certificate, and its
fallback policy remain unchanged. Native RadFoam is a comparison implementation;
it does not supply the same certificate. No Warp frontier candidate or automatic
strategy selector is implemented here.

## Persistent VTM trajectory

The fixed-slot Armadillo capture contains 180 frames of 8,192 float32 sites.
Each backend traversed all 179 transitions with persistent state, one full warm-up,
and three measured trials. All 2,148 measured updates succeeded and matched the
CGAL 6.2.1 tetrahedron sets for both previous and current inputs. No measured
sample reported CPU quota throttling.

| Backend | Median update | 95th percentile | Sum of per-transition medians |
| --- | ---: | ---: | ---: |
| VTM fragment reuse | 3.55 ms | 9.61 ms | 726 ms |
| Native RadFoam incremental | 4.30 ms | 16.87 ms | 803 ms |
| Native RadFoam fresh | 16.64 ms | 35.67 ms | 3,081 ms |
| Warp fresh | 91.59 ms | 145.51 ms | 18,337 ms |

Using a 5% margin, VTM wins 134 transitions, native incremental wins 37, and
eight are near ties. Native wins are concentrated early in this one trajectory.
All measured VTM transitions accepted fragment reuse; the separate corrupted-seed
control below exercises its wall fallback. Selected independent pairs also
reproduce a mixed winner: native wins frame 2, while VTM wins frames 1, 162 and 179.

These are isolated geometry updates. The sum of medians excludes initialization,
offline analysis and validation; it is not whole-optimizer wall time. Native
adapter input guards and internal point-order mapping are included in update
time, so these numbers are not directly interchangeable with older native-only
microbenchmarks. Motion and connectivity churn in the CSV are explanatory
offline measurements, not measured inexpensive routing signals. This single
trajectory does not establish a held-out strategy-selection rule.

## Bonsai bounded adjacency

Four paired transitions from the recorded Bonsai fit cover 9,216 to 253,903
sites. Each pair uses one warm-up and five measured trials. Both Warp variants
return zero cell status and identical directed-adjacency hashes in every measured
sample. This cross-check is **not an independent exact completeness certificate**
and does not turn bounded adjacency into a full-hull tetrahedralization.

| Source fitting step | Sites | Fresh adjacency | Previous-neighbor hints | Ratio |
| --- | ---: | ---: | ---: | ---: |
| 3 | 9,216 | 150.99 ms | 143.19 ms | 1.05× |
| 4,814 | 84,946 | 21.19 ms | 19.31 ms | 1.10× |
| 8,401 | 168,402 | 33.65 ms | 28.86 ms | 1.17× |
| 11,103 | 253,903 | 48.30 ms | 43.89 ms | 1.10× |

The legacy capture lacks persistent IDs through prune/densify boundaries and
omits some refreshes. Export conservatively resets identity at those boundaries;
it does not reconstruct genuine birth/death history. The source fit used the
shared reduced site/ray/resolution recipe documented in the
[RadFoam experiment](https://github.com/alecjacobson/warpower/tree/e37e00f693a26bee20928ba01f64a78d7906ed8a/experiments/radfoam-integration).
This replay neither loads images nor claims a win over the official full fitting
recipe.

## Controlled behavior

The 18-frame corpus covers unchanged/reordered inputs, local/global motion,
rollback, insertion/deletion, duplicate/nonfinite coordinates, exact cospheres,
one-ULP perturbation, thin/planar sets and corrupted retained connectivity.

- Ordinary transitions, count changes, and restoration succeed across all four
  full-hull adapters and match CGAL.
- Duplicate coordinates succeed in both Warp variants and the native incremental
  reference. Native fresh reconstruction reports its duplicate-point exception.
- Warp's cospherical and one-ULP outputs differ from CGAL's connectivity but pass
  the exhaustive exact geometric checker. This does not establish identical
  symbolic tie policies. The native reference reports construction failures.
- Injected invalid tetrahedron indices are remapped safely. VTM rejects the
  retained fragment and accepts its wall fallback, matching CGAL. Arbitrary native
  retained-state injection is explicitly unsupported.
- Nonfinite and planar inputs produce explicit failures or unsupported-state
  reports. An earlier native exact-planar construction stalled for over 90 seconds;
  its interrupted evidence is retained. The native adapter now guards that case,
  and a process watchdog bounds an entire replay run, including native calls.
- Separate four-tetrahedron-capacity runs fail explicitly in both Warp variants
  and publish zero cells. They do not produce usable timing baselines.

## Reproduction and evidence

Use the shared Python environment with NumPy/SciPy, Warp and Torch installed.
Set `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` for the reported
single-worker reference setup. Build the optional native extension using the
linked integration instructions; it is pinned to upstream RadFoam
`3e7b52cf74e37ab2ab5e695f53570f515f537e3d`. The geometry implementation is based on
warpower `730ec233505bcc474c248d0ab237b10042fa029e`.

The commands, input provenance, native/reference hashes and machine information
are in [provenance.json](evidence/provenance.json). Compact results are in
[summary.json](evidence/summary.json), with per-transition data in
[vtm-sequence.csv](evidence/vtm-sequence.csv) and
[vtm-pairs.csv](evidence/vtm-pairs.csv). Regenerate the compact summaries with:

```bash
python experiments/incremental-replay/summarize.py \
  results/local/incremental-replay-297 experiments/incremental-replay/evidence
python -m unittest tests.test_incremental_replay
```

Full successful, failed and interrupted outputs, fixture frames, manifests and
executed source snapshots are retained in persistent ignored local storage.
The [local bundle manifest](evidence/manifest.json) pins those files. The bundle
has been packed and locally verified; it has not been uploaded or independently
download-verified. A Git commit of this milestone does not publish that corpus.

## Remaining issue work

Producer instrumentation must record true optimizer phases and stable IDs through
RadFoam birth/prune events. Further work includes the Warp frontier candidate,
common certification costs, filtering/bookkeeping/growth/certification profiling,
CUDA-captured sequences, allocator/memory/switching costs, broader real scenes and
held-out crossover tests. These remain prerequisites for an adoption or default
policy recommendation.
