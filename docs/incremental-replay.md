# Isolated incremental geometry replay

`python -m benchmarks.incremental_replay` records and replays point trajectories
without a VTM optimizer, RadFoam scene, renderer, or image dataset. It is the first
experiment infrastructure for [issue #297](https://github.com/alecjacobson/warpower/issues/297).
It compares existing implementations; it does not add a frontier implementation,
automatic strategy selection, or a production API/default change.

The fixture reader, producer ID registry, exporters, and inspection command only
need NumPy. Offline motion features and the optional floating-point reference use
SciPy. Warp/CUDA and Torch/native RadFoam are imported only when their backend is
selected.

## Fixture identity and events

A fixture directory contains a versioned `manifest.json` and checksummed NPZ frame
payloads. Every frame preserves original float32 coordinates, int64 site IDs, a
stored capacity and active prefix, current-row to previous-row correspondence,
inserted/deleted IDs, an identity epoch, and explicit reset/restoration markers.
Optional retained tetrahedra carry a separate site-ID index mapping. Malformed
retained indices are allowed deliberately for adversarial replay and are never
used as unchecked host array indices.

Replay normalizes active rows by stable site ID before constructing backend
inputs. This prevents storage reordering from masquerading as motion and gives
every backend the same canonical input order. Native RadFoam's internal
permutation is carried through the sequence; subsequent coordinates are mapped
to that order. VTM retains its normal fragments and fallback state. Count changes
reallocate the workspace; same-size identity changes and explicit resets clear
retained state. Those costs are included in the measured update.

A restoration marker names an earlier frame whose active IDs and binary float32
coordinates must match exactly after row normalization. Restoration does not
automatically reset geometry state: the next update must handle the retained
state left by the intervening proposal. IDs allocated after an insertion are
never reused for unrelated sites, including after rollback.

New producers can use `benchmarks.replay.fixture.Writer` and `SiteRegistry`:

```python
registry = SiteRegistry(len(points))
writer = Writer(output, name="scene-fit", provenance={"seed": seed, "revision": revision})
writer.append(points, registry.ids, event="initial", reset=True)

# Apply the same selection/permutation to attributes and points in the producer.
registry.select(kept_rows)
writer.append(pruned_points, registry.ids, event="prune", stage="densification")

registry.insert(number_of_new_sites)
writer.append(grown_points, registry.ids, event="insert", stage="densification")
writer.finalize()
```

Pass `active_count` when a producer has reserved inactive slots. Record actual
proposal/acceptance stage names and restoration targets at the producer; do not
infer those optimizer decisions from maximum displacement alone.

## Existing captures

The VTM exporter handles the fixed-slot 8,192-site captures produced by
`benchmarks.neighbor_clipping_trials.record_trials`. It preserves slot identity
and identifies exact nonconsecutive coordinate restoration. The legacy producer
did not record phase or acceptance labels, so the exporter marks them
unclassified.

```bash
python -m benchmarks.incremental_replay export-vtm \
  --source /path/to/actual-trials/frames.npy \
  --record /path/to/actual-trials/record.json \
  --output results/local/replay/vtm
```

The legacy RadFoam exporter requires the original-order `warpower` or
`warpower-hints` run report. It refuses native/Morton captures because their
permutations were not recorded. Full refreshes, count changes, and recorded
prune/densify boundaries begin a new identity epoch. The exported insertion and
deletion lists at those boundaries represent an identity reset, **not recovered
birth/pruning decisions**. Same-size incremental segments retain identity.
Some original geometry refreshes were not saved, and replay follows the retained
snapshots rather than reconstructing the missing updates.

```bash
python -m benchmarks.incremental_replay export-radfoam \
  --source /path/to/bonsai-warpower-20k/trajectory.npz \
  --metadata /path/to/bonsai-warpower-20k/trajectory.json \
  --report /path/to/bonsai-warpower-20k/report.json \
  --output results/local/replay/radfoam
python -m benchmarks.incremental_replay inspect results/local/replay/radfoam
```

The controlled corpus adds unchanged/reordered inputs, local/global motion,
restoration after a proposal, insertion/deletion, duplicate coordinates,
nonfinite input, exact cospherical integer points, a one-ulp perturbation, thin
and planar sets, and corrupted retained connectivity:

```bash
python -m benchmarks.incremental_replay synthetic \
  --count 256 --seed 2026 --output results/local/replay/control
```

Native full reconstruction stalled on the exact planar control in an initial
smoke run. The optional native adapter now explicitly rejects axis-aligned
planar inputs before entering that call; it also rejects nonfinite input. This
does not claim a native robustness guarantee on other degenerate configurations.
The CLI runs its worker in a separate process with a 900-second total wall-clock
limit, including JIT. `--timeout-seconds` changes that limit; zero disables it.
A timeout terminates the worker, preserves available partial records, and marks
the report incomplete. It is not a per-update latency measurement.

## Separate output contracts

The full-hull group is `vtm-fragment`, `warp-full`, `radfoam-incremental`,
`radfoam-full`, and `scipy-full`. VTM keeps its complete current-coordinate exact
and symbolic certificate and wall/fresh fallbacks. The two native choices are
optional references and do not provide that certificate.

The bounded-adjacency group is `warp-adjacency` and `warp-hints`. These produce
clipped Voronoi adjacency, not full-hull tetrahedralizations. The CLI rejects a
mixed-contract run and refuses to call a full-hull reference a certificate for
bounded adjacency. Zero cell status is not a proof of complete graph correctness.

Native retained-state injection is unsupported and reported explicitly. Fresh
baselines ignore optional previous connectivity; `vtm-fragment` remaps and
recertifies it. `--tetrahedra-capacity` can exercise Warp capacity exhaustion;
the native reference does not expose the corresponding control.

## Replay and validation

Paired mode reconstructs and independently checks the previous input before
each measured transition. Sequence mode traverses every frame for each trial and
retains each backend's legitimate state. Subsampling a sequence is rejected;
use paired mode for selected transitions. Backend order is randomized with a
recorded seed; at least one complete warm-up pass is required.

```bash
python -m benchmarks.incremental_replay run results/local/replay/vtm \
  --mode pairs --transitions 1 2 162 179 \
  --backends vtm-fragment warp-full --warmups 1 --repeats 5 \
  --output results/local/replay/vtm-pairs.json

python -m benchmarks.incremental_replay run results/local/replay/vtm \
  --mode sequence --backends vtm-fragment radfoam-incremental \
  --native-module-dir /path/to/headless/build --native-module radfoam_headless \
  --warmups 1 --repeats 3 --output results/local/replay/vtm-sequence.json

python -m benchmarks.incremental_replay run results/local/replay/radfoam \
  --mode sequence --backends warp-adjacency warp-hints --validate none \
  --output results/local/replay/radfoam-adjacency.json
```

The [RadFoam integration experiment](https://github.com/alecjacobson/warpower/tree/e37e00f693a26bee20928ba01f64a78d7906ed8a/experiments/radfoam-integration)
describes building the optional headless reference module. Replay imports its
native triangulation bindings directly, without importing the RadFoam Python
scene or loading image data.

SciPy validation is the default floating-point reference. For an exact-predicate
reference, build `benchmarks/cpu/delaunay.cpp` as described in
[the CPU reference instructions](../benchmarks/cpu/README.md), then use
`--validate cgal --cgal-binary /path/to/cgal-delaunay`. Both comparisons canonicalize
duplicate ownership and require tetrahedron-set equality. Valid cospherical
alternatives can differ. `--exact-checker /path/to/check-delaunay` invokes the
existing exhaustive CGAL checker on nonmatching outputs and records whether they
are geometrically valid; it does not claim identical symbolic tie policies.

Raw failures, unsupported cases, reference mismatches, prior-input checks,
medians, tail latency, fallback counters, CPU quota observations, offline motion
features, and topology churn are retained. The independently comparable median
requires both prior and current outputs to match the selected reference.
Source hashes, package versions, binary hashes for validators, and random trial
order accompany each report.

Canonicalization and host-to-device staging are outside update timing. Device
copies, internal permutations, workspace allocation on count changes,
construction, synchronization, and state switching are inside. Readback,
independent references, and explanatory nearest-neighbor/topology measurements
are outside. Those offline measurements are not assumed cheap enough for an
automatic routing policy.

## Remaining investigation

This harness does not complete issue #297. New producer instrumentation is still
needed for true stage labels and persistent IDs through RadFoam pruning and
densification. A Warp frontier-repair candidate, common certification comparisons,
separate filtering/bookkeeping/growth/certification timing, captured-sequence
qualification, memory/switching-cost measurements, broader real scenes, and
held-out crossover validation remain necessary before recommending an API or
default change. Keep the current VTM policy while that evidence is collected.

Run the CPU protocol tests without initializing CUDA:

```bash
python -m unittest tests.test_incremental_replay
```

Raw fixtures and replay output belong under ignored `results/local/` and in
checksum-pinned release bundles when published. Follow the repository's
[artifact policy](experiment-artifacts.md); committing the harness does not
publish the existing local corpus.
