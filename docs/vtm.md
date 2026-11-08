# Variational tetrahedral meshing development

The [implementation goal](https://github.com/alecjacobson/warpower/issues/29)
tracks a pure-Warp, device-resident implementation of
[Alliez et al., Variational Tetrahedral Meshing (2005)](https://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Alliez05.pdf).
The [public mesher](mesher.md) now composes the numerical stages in a captured
nonconvex-domain pipeline. The [current schedule and size study](fixed-site-current.md)
measures existing options through 16,384 sites against native size controls.
The [hierarchical retained-cell queries](hierarchical-seed-queries.md) improve
large complete GPU solves while preserving their coordinates and connectivity.
The [cached exact-query split](cached-exact-seed-queries.md) further improves
these larger solves while retaining complete exact fallback.
[Shared new-cell geometry](prepared-new-cell-queries.md) further reduces completion
work without adding device arrays.
[Bounded tangential topology recovery](tangential-topology-recovery.md) now resolves
the recorded uniform/graded 8K scanned Armadillo extraction failures while keeping
all meshing decisions in the captured device graph.
The wide comparable-quality CPU performance goal remains unmet.

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
certified unweighted meshing, use [`DelaunayWorkspace`](delaunay-recovery.md).
`DiagramWorkspace(adjacency=False, return_tetrahedra=True)` also exposes the
diagram-based tetrahedral output, including weighted regular triangulations.

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

## Current implementation and follow-up work

The [fixed-boundary ODT workspace](odt.md) provides the interior optimizer and
captured failure recovery. See [validation and known limits](vtm-validation.md).
It is not yet the full boundary-fitted mesher or a CPU speedup result.

The [surface-domain layer](domains.md) provides captured refits, geometric queries,
device input validation, and surface/feature quadrature. It supports nonconvex
boundaries, cavities, and disconnected components independently of tetrahedral
meshing. The independent [boundary fitter](boundary.md) now provides evolving
surface/crease membership and exact corner anchors. Its integration with interior
ODT runs in a captured loop, with device backtracking and rollback. The internal
core returns full-hull connectivity; the independent [extractor](extraction.md)
selects a nonconvex domain mesh and measures topology, quality, and approximation
error. Both compose in a captured producer/consumer loop. The optional
[certified Delaunay constructor](delaunay-recovery.md) handles recovered
candidates with exact original-input certification and explicit finite failure.
For capacities of 256–2,048 sites, the VTM core additionally
[recertifies and completes previous Delaunay fragments](delaunay-reuse.md) before
falling back to full reconstruction. Standalone geometry defaults are unchanged.
Larger reserved workspaces with a device active count
[route reconstruction through smaller preallocated prefixes](active-reconstruction.md)
until the sites outgrow them. [Additional 4,096/8,192-site branches](larger-active-prefixes.md)
reduce large-reservation costs while adding persistent storage and setup work. Their [full route reuses certified fragments through
16,384 reserved sites](extended-fragment-reuse.md). Full original-coordinate
certification and the ordinary full-capacity fallback remain mandatory.
The [sampled sizing layer](sizing.md) supplies device Lipschitz extension and
graded interior/boundary updates, pole-based automatic feature-size estimates,
and fixed-budget device initial sampling. The internal
[refinement stage](refinement.md) grows a device active prefix with ranked,
deduplicated Voronoi-center insertions and certified transactional reconstruction.
Optional [selective optimization](selective.md) limits updates to poor retained
tetrahedra and one neighbor ring, with an additional minimum-quality guard.
[Tangential boundary jitter](jitter.md) and [interior teleportation/insertion](interior-improvement.md)
now provide optional quality transactions with complete device rollback.
The [public scheduler](mesher.md) initializes, grows, fits, improves and extracts
the mesh in one device graph. It checks input/output topology invariants, retries
failed insertion batches with a single insertion, repairs coarse invalid material
boundaries with bounded additional fitting, and restores optional quality stages
that invalidate the last good mesh. Broader examples, integrated time-to-quality
comparisons and measured CPU outperformance remain necessary to complete the goal.

Additional findings are tracked in
[#42](https://github.com/alecjacobson/warpower/issues/42) (old-connectivity line-search stalls),
[#43](https://github.com/alecjacobson/warpower/issues/43) (trajectory reconstruction costs), and
[#44](https://github.com/alecjacobson/warpower/issues/44) (intermediate high-valence cell capacity).
Domain work also uncovered
[#47](https://github.com/alecjacobson/warpower/issues/47) (native refit allocations)
and [#48](https://github.com/alecjacobson/warpower/issues/48) (surface query performance).
Boundary work also exposed [#50](https://github.com/alecjacobson/warpower/issues/50)
(native radix-sort capture allocations), resolved by rebuilding in an input-validity
conditional graph whose scratch storage belongs to capture setup.
[#52](https://github.com/alecjacobson/warpower/issues/52) records the explicit
stack allowance needed to memory-check a deeper captured optimizer trajectory;
see the [diagnostic environment](cuda-validation.md).
[#54](https://github.com/alecjacobson/warpower/issues/54) adds the isolated
[exact predicate foundation](exact-predicates.md) for certified numerical
recovery, integrated with candidate construction and whole-mesh certification in
[#32](https://github.com/alecjacobson/warpower/issues/32).
[#56](https://github.com/alecjacobson/warpower/issues/56) removes repeated failed
construction work by reusing successful strategies with fresh current-input
certification; [#58](https://github.com/alecjacobson/warpower/issues/58) tracks
the remaining large-cloud certification cost.
