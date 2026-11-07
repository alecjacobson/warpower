# Next substantive performance check: #37 / #40

After #158 is merged, measure time to comparable quality before treating more
small kernel gains as sufficient progress toward the CPU goal. The current
0.820-second torus still has different boundary error and quality from CGAL's
0.238/0.172-second sequential/four-thread results. Equal iteration budgets would
not make these algorithms comparable, and CGAL's ordinary convergence/freezing
must remain enabled.

Use the existing complete `benchmarks.mesher` runner, unchanged runtime defaults,
same 48x24 float32 input torus and seed 2026, n=1000, three clean resident replays
per exploratory configuration. Record setup/first execution separately and all
invalid/timeout cases. Proposed configurations are `(initial,growth,polish,selective,quality)`:

- Control: `(256,5,20,10,4)`.
- Short: `(256,2,5,3,1)`.
- Intermediate: `(256,3,10,5,2)`.
- No optional quality moves: `(256,3,10,0,0)`.
- Larger initial prefix: `(512,2,5,3,1)`.
- Fixed budget, short: `(1000,0,5,3,1)`.
- Fixed budget, intermediate: `(1000,0,10,5,2)`.
- Fixed budget, full polish: `(1000,0,20,10,4)`.

Save every valid full/material mesh. After GPU timing, independently check full
hulls, material manifold topology, minimum/1st/median radius ratios, signed
volume error, two-sided surface RMS and boundary-site fraction. Keep first and
warm timings and quota failures, rather than using the fastest isolated sample.
Find nondominated quality/time choices; do not silently change defaults or
declare a faster lower-quality mesh a speedup.

Use those observations to select a small CGAL size/facet-distance sweep if needed
for actual comparable-quality targets. Preserve normal CGAL stopping/freezing and
all distinct parallel outputs. Report setup and output counts along with quality.
Repeat meaningful candidates on other domains before promoting any result.

The private #159 extraction-split source is also ready for a short pilot, but has
not run and has no performance claim. Broader cooperative construction #98 is
still needed for scaling beyond the small VTM reuse prefixes; do not infer a
large-input gain from the current 1k optimizations.
