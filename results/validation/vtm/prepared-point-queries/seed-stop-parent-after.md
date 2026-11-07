Implement a feature-complete pure-Warp version of Alliez et al., Variational Tetrahedral Meshing (SIGGRAPH 2005), built on Warpower's independently useful diagram and Delaunay routines.

The numerical pipeline must remain on device and compose in CUDA graph loops. Preserve existing standalone geometry performance. Target at least 10x versus a native CPU implementation on substantial examples at comparable quality; report measured results and limits, never substitute a kernel-only comparison for end-to-end success.

Scope includes ODT optimization, robust Delaunay updates, boundary/feature fitting, graded sizing, initial sampling, progressive refinement, quality improvement, nonconvex domain extraction, graph APIs, examples, and CPU comparisons. Boundary approximation is intentional; exact input-facet conformity is outside the Alliez-style implementation.

Baseline: commit 3091289464af368d2a61f843d169d7f4863d1cfd, NVIDIA L40.
Reference: https://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Alliez05.pdf
CPU reference: https://doc.cgal.org/latest/Mesh_3/index.html

Implementation will proceed in tested PRs, merged by the agent as authorized. Newly discovered work receives additional linked issues. Completion requires the whole pipeline and evidence, not just the optimizer prototype.

Milestones:

- [x] #30 — Preserve standalone geometry APIs and establish VTM performance baselines
- [x] #31 — Implement device ODT reductions and fixed-boundary captured optimization
- [x] #32 — Handle Delaunay degeneracies reliably during optimization
- [x] #33 — Add GPU domain queries and surface/feature quadrature
- [x] #34 — Implement VTM boundary fitting and sharp-feature constraints
- [x] #35 — Implement graded sizing fields and device initial sampling
- [x] #36 — Extract and validate the final nonconvex domain tetrahedral mesh
- [x] #37 — Add captured progressive refinement and worst-element improvement
- [x] #38 — Integrate the complete device-resident VTM workspace and public API
- [x] #39 — Build paper-inspired VTM examples and independent quality validation
- [ ] #40 — Benchmark complete VTM against native CPU meshers at comparable quality



Additional discovered work:

- [x] #42 — Avoid line-search stalls during topology changes
- [ ] #43 — Profile and accelerate trajectory reconstruction
- [x] #44 — Recover intermediate high-valence clipping overflow


- [x] #47 — Remove native refit graph allocation/free nodes
- [x] #48 — Accelerate pure-Warp surface queries

- [x] #50 — Keep nearest-site rebuild scratch out of captured replay

- [x] #52 — Isolate nested-loop instrumentation limits and document validated test configurations

- [x] #54 — Exact device predicates for original-coordinate certification

- [x] #56 — Reduce certified Delaunay recovery and validation overhead in captured loops

- [x] #58 — Accelerate exact Delaunay certification with conservative sphere bounds

- [x] #61 — Recover structured torus boundary Delaunay construction for automatic sizing

- [x] #62 — Accelerate original-coordinate symbolic cell recovery

- [x] #64 — Separate initial-sampling random streams at large site counts
- [x] #65 — Preserve probability-CDF monotonicity across empty proposal cells

- [x] #67 — Device-controlled active prefixes for captured VTM
- [x] #69 — Ranked Voronoi-center insertion batches
- [x] #70 — Selective optimization and tangential boundary jitter
- [x] #71 — Normalized-cell teleportation and worst-tet insertion

- [x] #76 — Bounded recovery for invalid coarse graded material extraction

- [x] #81 — Refresh stale reconstruction preferences on changing VTM inputs

- [x] #84 — Accelerate hull-face certification in complete meshing trajectories

- [ ] #87 — Fast certified candidates with bounded local repair

- [x] #89 — Accelerate certified sphere leaves with cached conservative bounds (#96)

- [x] #91 — Isolate conditional native-sort allocation-tracking reports

- [x] #92 — Resolve imported surface-contact rejection with exact witnesses

- [x] #93 — Resolve the thin-hand solve exceeding ten minutes

- [x] #95 — Reuse successful auxiliary amplitudes as certified candidate-order hints (#99)

- [x] #97 — Investigate a proven local Delaunay certificate with global fallback

- [ ] #98 — Cooperative Warp reconstruction for small/medium VTM batches

- [x] #102 — Faster pure-Warp exact predicate fallbacks for degenerate reconstruction

- [x] #104 — Complete certified Delaunay fragments by device facet extension

- [x] #105 — Parallel global sphere certification for small degenerate clouds

- [x] #108 — Recover useful insertion batches in the thin-hand growth scheduler.

- [x] #109 — Stabilize generated Voronoi centers and deduplicate roundoff-split insertions.

- [x] #110 — Evaluate tiny certified tetrahedron volumes robustly during VTM.

- [ ] #112 Recover a manifold material boundary for the full thin-hand example.

- [x] #114 Prevent unassigned VTM sites from escaping outside the input domain.

- [x] #116 Reuse certified cells across nearby VTM reconstructions.

- [ ] #117 Resolve one-ulp stagnation in fitted float32 boundary coordinates.

- [x] #118 Publish a validated thin-hand example with sufficient crease sampling (#119).

- [x] #120 Choose reconstruction capacity from the growing device active prefix (#123).

- [ ] #122 Reduce duplicated scratch in device-selected VTM prefix constructors.

- [x] #124 Stabilize physical determinants and circumcenters in material extraction.

- [x] #125 Recover the nonmanifold boundary in the full coarse cavity example.

- [x] #127 Carry stable geometry through quality selection and bounded dual volumes.

- [x] #128 Preserve physical face areas in uncertain tetrahedron quality evaluation.

- [x] #129 Complete the unfiltered memory audit of the full disconnected quality trajectory.

- [ ] #130 Reuse exact certificate results when only a subset of VTM sites changes.

- [x] #133 Use explicit small launch blocks for VTM geometric queries.



- [x] #135 Tune VTM reconstruction and boundary-projection launch sizes (#137).

- [x] #136 Reuse exact cell proofs within one unchanged-input facet completion.

- [x] #138 Reuse current hull-support results to skip impossible facet searches.

- [x] #140 — Uniform and graded scanned Armadillo example, provenance and independent validation

- [x] #142 — Diagnose and recover the graded scanned Armadillo material-extraction failure

- [x] #145 — Sharp-feature mechanical example.
- [x] #146 — Stop private hull-support traversal after a conclusive exterior witness.
- [ ] #147 — Investigate stack-dependent sanitizer behavior in mixed extraction graphs.

- [ ] #149 — Improve and quantify worst-case crease fitting on the counterbored plate.

- [x] #152 Spatial radix surface hierarchy and captured refit acceleration (#153).

- [x] #154 Boundary quadrature assignment and scalar-reduction throughput (#156).

- [ ] #155 Reduce mixed-process full-mesher sanitizer synchronization delay.

- [x] #157 Share current-input sphere preparation across cooperative seed-query lanes.

- [ ] #159 Separate physical tetrahedron geometry from surface-query kernels.

- [x] #161 Extend certified fragment reuse to larger VTM reconstructions.

- [ ] #164 Prune current-input facet candidate searches with conservative hierarchy bounds.

- [x] #166 Evaluate analytic roundoff filters for certified VTM predicates.
- [ ] #167 Separate strict facet search from exact device fallback.
- [ ] #169 Stop private seed validation after a certified rejection.
