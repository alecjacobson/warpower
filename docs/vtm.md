# Variational tetrahedral meshing development

The [implementation goal](https://github.com/alecjacobson/warpower/issues/29)
tracks a pure-Warp, device-resident implementation of
[Alliez et al., Variational Tetrahedral Meshing (2005)](https://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Alliez05.pdf).
This is work in progress; the existing diagram and full-hull tetrahedron routines
are not yet a complete nonconvex-domain mesher.

The target includes ODT relocation, boundary and sharp-feature fitting, graded
sizing, initial sampling, refinement, final domain extraction, and GPU convergence
and recovery decisions. Geometry processing must compose in a captured GPU loop;
allocation and compilation are setup operations. Boundary approximation is part
of the method. Exact preservation of every input facet is outside this scope.

## Milestones

| Issue | Deliverable |
|---|---|
| [#30](https://github.com/alecjacobson/warpower/issues/30) | Standalone APIs and performance regression baseline |
| [#31](https://github.com/alecjacobson/warpower/issues/31) | ODT kernels and fixed-boundary device optimizer |
| [#32](https://github.com/alecjacobson/warpower/issues/32) | Robust Delaunay construction through degeneracies |
| [#33](https://github.com/alecjacobson/warpower/issues/33) | Domain queries and quadrature |
| [#34](https://github.com/alecjacobson/warpower/issues/34) | Boundary and sharp-feature fitting |
| [#35](https://github.com/alecjacobson/warpower/issues/35) | Graded sizing and initial sampling |
| [#36](https://github.com/alecjacobson/warpower/issues/36) | Nonconvex domain extraction and validation |
| [#37](https://github.com/alecjacobson/warpower/issues/37) | Progressive refinement and worst-element improvement |
| [#38](https://github.com/alecjacobson/warpower/issues/38) | Complete captured workspace and public API |
| [#39](https://github.com/alecjacobson/warpower/issues/39) | Examples and independent quality validation |
| [#40](https://github.com/alecjacobson/warpower/issues/40) | Native CPU comparisons and final performance evidence |

## Compatibility and performance contract

The baseline is commit `3091289464af368d2a61f843d169d7f4863d1cfd`.
`voronoi_diagram`, `power_diagram`, and `DiagramWorkspace` remain independently
usable. New meshing modules are lazy imports, and their optional computation is
separate from the existing default diagram path. The standalone
`delaunay_tetrahedra` and `regular_tetrahedra` convenience functions return the
full-hull mesh without constructing bounded adjacency. For captured standalone
meshing, use `DiagramWorkspace(adjacency=False, return_tetrahedra=True)`.

Changes to shared geometry code require independent correctness checks and
paired performance comparisons against the baseline, including diagram calls
and tetrahedra-only and combined graph replays. Record raw samples, changed
source hashes, output equivalence, CPU throttling, and persistent GPU memory.
Investigate significant slowdowns; do not offset a regression in one routine by
reporting a speedup in another.

The full-mesher performance target is at least **10x** versus a native CPU
implementation on substantial examples at comparable mesh quality. This is a
goal, not a measured result. CGAL Mesh_3/ODT, including parallel CPU operation
where available, is the planned comparison. Report total time and resident-device
replay separately, alongside boundary error, element-quality distributions,
topology checks, hardware, and thread limits. A Python reference is useful for
correctness but is not sufficient evidence for the CPU speedup claim.

Examples will include a torus, a graded scanned shape, thin features, a
fandisk-like part, a cavity, and disconnected components, with asset provenance
and reproducible generation. The main README remains limited to installation
and the best measured flagship result.

Run the baseline gate from the repository root:

```bash
git worktree add --detach .work/vtm/baseline 3091289464af368d2a61f843d169d7f4863d1cfd
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python -m benchmarks.geometry_regression \
  --baseline-path .work/vtm/baseline \
  --output results/local/geometry-regression.json
```

The harness alternates baseline/current calls with resident inputs, checks full
output equivalence, compares eager launch/allocation traces, and records paired
95% intervals. It exits with failure if an interval lies entirely above a 2%
slowdown or CPU quota throttling is observed. This threshold separates an
investigation trigger from noisy individual samples; all changes remain visible.
Persistent-memory measurements are allocator deltas around workspace creation
and include capture-runtime storage. `--quick` provides small smoke cases and is
not release performance evidence.
