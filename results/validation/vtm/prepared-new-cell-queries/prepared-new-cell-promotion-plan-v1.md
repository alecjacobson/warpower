# Conditional production plan for issue 233

Do not adopt until the private v2 pilot/grid checks establish correctness and a
useful full-solve benefit. Retain v1 and v2 sources/results independently.

If v2 qualifies:

- Add private `_unproved_queries.py` with the exact tested prepare/strict/exact
  kernels. Keep module options (no FMA/fast math) and 32-thread integration launch.
- Its helper accepts existing Sphere/Box/pending/index arrays as required keyword
  arguments, validates their capacities and current segment association, and
  allocates no device arrays. Standalone primitive tests allocate their own
  scratch explicitly. No hidden global registry or certificate/seed parent cycle.
- ProvedCertificate starts with `_prepared_unproved = None`; `_global_spheres`
  dispatches to it only when non-null and `point_parallel` is true, else keeps the
  original implementation. Ordinary certificates and public Delaunay are unchanged.
- SeededFacetCompletion attaches this helper only with StacklessSeedQueries on
  the existing 4097..16384 route and sufficient array capacity. Scratch remains
  consumed seed data, refreshed before every subsequent consumer. Custom capacity
  paths with insufficient scratch use the original checker.
- Preserve all kernel ASTs relative to v2. Adapt all 56 primitive/dispatch tests
  to the production helper, keeping exact old-checker counters/rational oracles.
  Add meaningful ownership/association/capacity guards, including proof-all and
  zero-count transitions. Existing large changing-input tests cover reuse across
  failed and recovered iterations; full suite, memory lifetime, and native trace
  gates still apply.
- Run fresh interleaved production/baseline full solves including 1k/4k unchanged,
  8k progressive/fixed, 16k fixed/graded, reserved16k target4k, retained grid, and
  independent public diagram/Delaunay guards. Independently validate every full
  mesh and canonical/work equality after all clean timing is terminal.
- Native warmed complete solve and changing-input graph loop: only resident
  kernels/conditional/device copies, stable pointers and invalid-state recovery;
  preserve raw traces and DOTs compressed losslessly. Disclose first-launch
  deferred graph instantiation/upload if present.
- Padded and pool-disabled isolated sanitizer jobs exercise current kernels and
  integration/lifetime, including overflow, exact ties and changing counts.
- Publish a source/hash manifest and evidence report; architecture/VTM links,
  README unchanged. PR closes only 233; comparable-quality wide CPU gap and
  unsuccessful Armadillo remain open. Self-merge after all relevant gates pass.
