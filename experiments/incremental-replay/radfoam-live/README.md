# Real RadFoam producer capture

The Bonsai run records **56 consecutive native geometry inputs and meshes** from
450 actual fitting optimizer steps. All 56 native tetrahedron sets match the
independent CGAL 6.2.1 reference. The complete producer identity chain validates:
seven permutations, 8,943 newly allocated IDs, and 885 pruned IDs. No ID reset,
inferred correspondence, synthetic motion, or skipped geometry call is used.

This is a short workload from pinned upstream `train.py`, with explicit reduced
initialization, site/ray budgets, resolution, error-map selection and schedule.
It is not a full training-quality or performance comparison. Instrumentation
copies inputs and connectivity synchronously and compresses each snapshot.

| Input category | Count |
| --- | ---: |
| Initial native construction | 1 |
| Optimizer geometry refresh | 49 |
| Actual prune/densify refresh | 5 |
| Requested point-freeze full refresh | 1 |
| Native failures/retries | 0 |

The capture starts with 9,216 sites and ends with 17,274. Genuine birth/prune
events occur at optimizer steps 39, 139, 239, 339 and 439. The full refresh at
step 400 retains site identity. Several count-changing inputs immediately follow
an ordinary geometry refresh and have exactly zero displacement for surviving
sites. Later unchanged inputs also arise naturally after position freezing.

The raw corpus occupies **34,496,366 bytes**, below the 150 MiB capture limit.
Model/optimizer checkpoints, PLY, final images and TensorBoard files were not
saved. Actual training losses, held-out evaluations and native geometry still ran.
The instrumented run completed in 46.72 seconds on an NVIDIA L40; this duration
is not a geometry benchmark.

## Producer contracts

`capture_radfoam.py` instruments the actual native constructor and every native
`rebuild` attempt. If upstream retries with perturbed coordinates, that attempted
input is recorded separately; failed native state is not described as the last
successful mesh. This run needed no retries.

Producer hooks track `permute_points`, `densification_postfix` and `prune_points`.
They verify the exact coordinate-row operation before applying the corresponding
ID operation. IDs are assigned before initial native ordering, new sites receive
monotonically increasing IDs, and surviving IDs persist through all full rebuilds.
Optimizer completion counters and the actual `train.py` iteration identify each
geometry refresh. Initialization's adjacency/AABB attachment is an event, not an
invented extra construction input.

The fixture uses the existing replay schema. Native connectivity is stored in
separate checked NPZ sidecars with `tetrahedra` and `site_ids`; this avoids forcing
native retained-cell injection into ordinary replay. For input frame `i`, the
sidecar `connectivity/iiiiii.npz` describes the native result for that input.
Its tetrahedron indices use the sidecar's ID mapping, which can differ from the
producer's input row order. For a pair `(i-1, i)`, use the previous sidecar as the
initial mesh and compare the two original float32 position arrays by stable ID.
For retained IDs, displacement is the exact float64 difference of those float32
coordinates. Newly inserted and deleted IDs remain separate events.

The [audit](audit.json) verifies all sidecar hashes, replays every identity
operation, checks index validity, and compares every observed native mesh to
CGAL. This offline comparison does not add VTM's runtime certificate to RadFoam.
[Selected pairs](pairs.json) identify early, middle, count-changing, reset and
unchanged cases; all 55 consecutive pairs remain available in the corpus.

## Reproduce

The source implementation is committed at
`7218403dba67603bd950f1128fa4d7ff8abafca1`. It uses RadFoam
`3e7b52cf74e37ab2ab5e695f53570f515f537e3d` and the existing headless native build.
[summary.json](summary.json) records the exact configuration, source hashes,
binary hash, losses/evaluations and capture manifest hash.

From the repository root, with Bonsai images and COLMAP data already installed:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python experiments/incremental-replay/capture_radfoam.py \
  --output results/local/radfoam-live-20261111 \
  --data /path/to/radfoam-training \
  --build /path/to/headless-native-build

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m benchmarks.replay.radfoam_capture_audit \
  results/local/radfoam-live-20261111/capture \
  --cgal-binary /path/to/cgal-delaunay \
  --output results/local/radfoam-live-20261111/audit.json

python -m unittest tests.test_incremental_replay tests.test_radfoam_capture
```

Raw files are retained under the persistent worktree's ignored
`results/local/radfoam-live-20261111/`. The [inventory](inventory.json) pins the
local files; no evidence archive has been published. The capture and six new
producer tests passed alongside the existing 13 replay tests.

This is one scene and a shortened fitting schedule. The locally available
garden, bicycle and playroom assets contain static fitted positions only, without
the images/cameras needed for a second real fitting capture. They were not used
to invent an optimization trajectory.
