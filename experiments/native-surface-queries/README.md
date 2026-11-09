# Native-query qualification

Runtime: `0dc7217e58bfb8ec60b33f1ca756a3b8749519db`.
Main baseline: `afec85de70d6c65b95e2c8d745edef5ecc9ca965`.

- `ablation.json`: every replay sample, configuration, scheduler result, input hash,
  independent mesh/fit check and coordinate/connectivity comparison for five cases
  with both backends. Timing is complete resident GPU replay; this is not a CPU
  comparison. Both backends use trusted-input mode.
- `default-source-audit.json`: all existing device definitions in the three edited
  runtime modules are unchanged; the other 73 existing runtime files are byte
  identical to main. Native code is isolated behind explicit backend selection.
- `tests.json`: the full 1,396-test result and log hash.
- `geometry-controls.json`: alternating paired geometry timings, output/launch
  equivalence, intervals, allocator observations and CPU-quota records.
- `standalone-controls.json`: certified Delaunay and default-mesher controls.
- `memory.json`: passing padded checks, the larger-case timeout/retry, and the
  unresolved native nested lifetime-tracking reports; this is not an all-green
  lifetime claim.
- `capture.json`: the native producer/refit/mesher/consumer trace, including its
  explicit allocation limitation and raw trace hashes.

See [the implementation report](../../docs/native-surface-queries.md) for scope,
reproduction commands and the native-refit limitation. Reproduce the scan input
with `python -m examples.fetch_armadillo --output results/local/armadillo-input`;
substitute that normalized OFF path into the recorded configurations. Build the
independent CGAL checking tools using [their instructions](../../benchmarks/cpu/README.md).

Raw meshes, profiler traces, logs and executed one-off drivers remain on persistent
storage under `results/local/fast-vtm-pivot-v1` and `.work/vtm`. They are not added
to Git. Compacted metadata retains their hashes where used as evidence. Historical
failed trials (normal-sign cavity cancellation and conditional native-refit
allocation) remain recorded separately from successful qualification.

`native-mesh-lifetime-controls.json` and `native_mesh_lifetime_v1.py` isolate
[#269](https://github.com/alecjacobson/warpower/issues/269) without Warpower geometry.
From the repository root:

```bash
PYTHONPATH=. compute-sanitizer --tool memcheck --track-stream-ordered-races all \
  --error-exitcode 99 --padding 32 python \
  experiments/native-surface-queries/native_mesh_lifetime_v1.py --depth 1
```

The recorded environment fails this nested lifetime control. `--depth 0` is the
top-level control; omit lifetime tracking to run the padded control. The diagnostic
script keeps owners alive, synchronizes setup and sets a 32 KiB stack allowance;
it does not alter production runtime settings.
