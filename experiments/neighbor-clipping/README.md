# Neighbor clipping evidence

See [the experiment report](../../docs/neighbor-clipping.md) for the result,
scope and reproduction commands. No runtime or README flagship change is made.

- `sweeps.json`: all 45 configuration results, timing samples, memory counts,
  error/fallback diagnostics, source/input hashes and independent Qhull comparisons.
- `actual-trials.json`: alternating paired comparisons on transitions selected
  from 180 recorded full-budget inputs of a real fresh Armadillo solve.
- `trajectory.json`: the recorded solve configuration/result and frame hash.
- `runtime-source-audit.json`: byte identity of all 77 runtime files against
  `5acfe23c2de194a7ed3380941ea4d1f2e3e12b9e`.
- `qualification.json`: ordinary test and padded/lifetime memcheck results and
  hashes of the retained logs.

Raw frames, logs and superseded pilot results remain under ignored persistent
`results/local/neighbor-clipping-v1` and `results/local/neighbor-clipping-v2`.
No binary artifacts are committed. The initial duplicate-input test compared
Qhull's arbitrary duplicate IDs directly; its corrected check validates the
documented canonical owner against the independent unique-point reference.
The initial recording invocation lacked `PYTHONPATH=.`; the installed benchmark
module entry point avoids that invocation error.

The implementation is an ordinary-Delaunay research prototype, not a new public
weighted meshing or VTM backend. Its negative comparison against the existing
reuse path is the reason production defaults are unchanged. The broader default
performance matrix is not rerun for an addition that changes no runtime source.
