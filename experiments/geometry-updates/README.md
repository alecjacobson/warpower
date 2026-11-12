# Representative geometry-update benchmark

The replay suite contains **45 real full-hull update pairs** from six VTM instances and a live RadFoam Bonsai fit, plus **three large Bonsai adjacency pairs**. Each full-hull pair includes original float32 coordinates, stable IDs, displacements through correspondence, and an independently constructed CGAL initial mesh. Replaying the bundle requires neither optimizer, training images nor renderer.

Caching oriented hull-face proofs within one geometry update improves complete 4,096-site VTM solves: **987.230 → 914.652 ms (7.35%)** for torus and **881.099 → 824.910 ms (6.38%)** for graded torus. Positions are bitwise identical, full tetrahedron sets and boundary roles match, iteration work is unchanged, and all four final meshes match CGAL. These are three alternating measured complete solves per variant after warm-up; they include sampling, growth, fitting, quality work, extraction and synchronization.

The table reports milliseconds, using the median of each case's pair medians. Five randomized measured trials follow one warm-up. All measured CPU quota windows are unthrottled. Hardware: one NVIDIA L40 (48 GB), Warp 1.19.0, Torch 2.7.1+cu126, CGAL 6.2.1, four-core CPU quota. Setup, canonicalization, upload, readback, hashing and validation are outside update timing. Device copies, synchronization, point reordering, fallback and workspace recreation on count changes are inside it.

| Case | Sites | Pairs | Warp cell rebuild | Native fresh | Native incremental | Fragment before | Fragment optimized |
|---|---:|---:|---:|---:|---:|---:|---:|
| Bonsai, live native fit | 9,216–17,274 | 11 | 286.970 | 22.890 | 6.725 | 20.022 | 19.969 |
| Armadillo | 8,192 | 7 | 107.491 | 16.573 | 4.602 | 4.345 | 4.055 |
| Graded torus | 4,096 | 6 | 32.108 | 13.593 | 5.116 | 3.209 | 2.967 |
| Feature hand (45°) | 1,024 | 6 | 385.125 | failed | failed | 3.131 | 2.824 |
| Thin hand (120°) | 1,024 | 4 | 11.402 | 9.061 | 2.633 | 1.073 | 1.065 |
| Torus, 1K | 1,024 | 6 | 13.035 | 8.299 | 2.743 | 1.191 | 1.173 |
| Torus, 4K | 4,096 | 5 | 34.045 | 13.262 | 3.980 | 3.615 | 3.248 |

The `warp-full` identifier selects `DelaunayWorkspace(cache_strategy=False)`, the cell-based constructor of Delaunay tetrahedra, labeled **Warp cell rebuild**. It is a different driver from both the private VTM wall builder and `voronoi_diagram`, which returns box-clipped neighbor CSR. The 286.970 ms result is not a measurement of the latter or proof of the best available fresh tetrahedron constructor: forced-fresh VTM wall construction still needs a separate comparison. A [follow-up profile](fresh-diagnosis.json) of Bonsai transition 3 (9,216 sites) attributes 282.773 of 286.592 ms of GPU kernel time (98.7%) to Voronoi cell construction. There are no normalized/symbolic recovery attempts or workspace reallocations on that update; the remaining kernels, including validation, total about 3.82 ms. This is one instrumented update, not a new aggregate timing. The general constructor's FP32 candidate path is gated at 32,768 sites, above this entire live Bonsai corpus. Native RadFoam instead seeds tetrahedra and grows an exposed-face frontier directly. The fresh column therefore measures a particular construction algorithm, not Warp language/runtime overhead.

The [claim audit](../benchmark-claims-audit/README.md) also measures `voronoi_diagram` on the same 9,216-site input: 141.241 ms versus 42.150 ms for Paragram and 19.579 ms for native fresh construction. Thus the small-input disadvantage is real even for neighbor construction; the separate tetrahedron driver adds further cost. On the 1.97-million-site checkpoint, Warpower neighbor construction remains faster than Paragram (224.804 versus 421.254 ms). These fresh-call probes have different output/certification contracts and averaging from the update table above. Do not infer a universal ranking. [Issue #300](https://github.com/alecjacobson/warpower/issues/300) tracks the corrections and regression requirements. Removing the FP32 gate alone is not a validated fix: [#270](https://github.com/alecjacobson/warpower/issues/270) records an 8K failed validation attempt. RadFoam and VoroTracing fitting runs are deferred until the geometry work is settled.

Bonsai's optimizer-only subset is **12.922 → 12.903 ms** for Warp versus **5.154 ms** for native incremental; the cache has no meaningful speed effect there. Growth/reset cases are recorded separately in [summary.json](summary.json). Native remains faster on nine of eleven Bonsai pairs. On Armadillo the median favors Warp, but native wins the expensive early repairs; summing these seven pair medians favors native (33.597 versus 44.124 ms). These sums describe selected updates, not training time.

The feature-hand inputs contain exact ties. Native returns `ambiguous triangulation` on all six pairs. The three Warp variants agree with one another and pass CGAL's exhaustive exact orientation, empty-sphere, face-pairing, coverage and hull-support checks; their valid cospherical choice differs from CGAL's triangulation. Across the full matrix, **1,065 of 1,125 measured updates succeed and are independently verified**; the other 60 are these native failures. All 675 measured Warp outputs pass. Previous inputs are also checked before measurement.

Large-cloud numbers below are a separate **bounded Voronoi-adjacency** contract. They use legacy original-order fitting segments with documented identity limitations. Both variants produce identical graph hashes and zero cell-status errors. This establishes agreement, not independent exact geometric certification; it is not a full-Delaunay comparison.

| Sites | Fresh adjacency | Previous-neighbor hints |
|---:|---:|---:|
| 84,946 | 21.119 | 19.852 |
| 168,402 | 34.220 | 28.080 |
| 253,903 | 48.812 | 43.636 |

The accepted change is private to VTM fragment completion. A cache stores cyclically oriented face keys only after the existing global support query proves them. Its contents are cleared at every completion call, including wall fallback, so no proof survives site motion. Reversed faces have distinct keys; hash collisions compare complete keys; saturation loses reuse rather than accepting a face. Cell orientation, topology, coverage, empty-sphere checks, symbolic ties and safe fallback remain enabled. Cache memory is reported per sample (about 0.75 MiB at 4K and 1.5 MiB at 8K).

Profiling points to exposed-face candidate search as the larger remaining cost: 11.7 ms on the profiled Bonsai Warp update versus 1.60 ms in native frontier growth. Those are instrumented component durations, not interchangeable end-to-end measurements. A separate strict-float32 search followed by exact replay of uncertain lanes was correct but slower: **14.428 → 24.547 ms** and **3.252 → 3.582 ms** on two Armadillo pairs. It remains an explicitly named benchmark probe, outside the production update path.

Validation includes **152 focused tests**, six-phase captured input changes, invalid input/recovery, corrupted retained state, exact ties, capacity exhaustion, direct search-lane comparisons and packaging regressions. Ten cache tests pass Compute Sanitizer with 32-byte padding and zero errors; the sanitizer launcher uses a diagnostic 128-KiB CUDA stack. This is padded memcheck coverage, not a new claim about full allocation-order race tracking.

The actual VTM producer captures cover full-budget fitting and quality reconstruction calls, including rejected proposals. They do not label acceptance or capture the entire initialization/growth/rollback history. Torus, graded torus and both hands are procedural examples, not paper assets. Armadillo uses the earlier real 180-frame capture. The native Bonsai producer records 56 consecutive inputs/meshes across 450 optimizer steps, including seven permutations, 8,943 births and 885 deletions; all 56 producer meshes match CGAL. Its five real growth events and point-freeze event are preserved in the full capture. The suite uses one radiance scene and fixed seeds, so it does not establish a universal crossover rule.

Keep strategy selection explicit in the benchmark. A new public `auto` mode is not justified by this corpus; offline motion, spacing and topology-churn features in [pairs.csv](pairs.csv) are not free routing signals. Persistent frontier bookkeeping, broader producer coverage and an independently evaluated automatic policy remain follow-up work under [#297](https://github.com/alecjacobson/warpower/issues/297).

Download the [verified evidence prerelease](https://github.com/alecjacobson/warpower/releases/tag/evidence-geometry-updates-4ceb748), verify it with the committed [manifest](manifest.json), then extract into a new empty persistent directory. The bundle includes selected pairs, complete producer captures, initial meshes, raw samples, profiler traces, source snapshots, optional reference binaries and build provenance. The native binary is platform-specific; [build_radfoam_reference.py](build_radfoam_reference.py) rebuilds the viewer-free reference from RadFoam `3e7b52cf74e37ab2ab5e695f53570f515f537e3d` and atomic_queue `b7f1befe89446752767243c5e5550779a9a5eed2` using `RADFOAM_SOURCE_DIR`, `RADFOAM_BUILD_DIR` and `ATOMIC_QUEUE_INCLUDE_DIR`. Geometry and tracing code are unchanged by that builder.

```bash
python -m benchmarks.artifacts verify --archive geometry-updates.tar.gz \
  --manifest experiments/geometry-updates/manifest.json
# Extract the verified archive into a new empty directory, then:
chmod +x "$GEOMETRY_EVIDENCE/references/cgal-delaunay" "$GEOMETRY_EVIDENCE/references/check-delaunay"
python -m benchmarks.replay.suite --root "$GEOMETRY_EVIDENCE" \
  --output "$GEOMETRY_RESULTS" \
  --native-module-dir "$GEOMETRY_EVIDENCE/references" \
  --cgal-binary "$GEOMETRY_EVIDENCE/references/cgal-delaunay" \
  --exact-checker "$GEOMETRY_EVIDENCE/references/check-delaunay"
```

For Warp-only full-hull runs, add `--backends vtm-fragment vtm-hull-cache warp-full`. `vtm-fragment` disables the new private cache as the before baseline; `vtm-hull-cache` uses the production optimization. [suite.json](suite.json) pins every fixture and transition. The [compact summary](summary.json), [pair-level CSV](pairs.csv), [complete solves](integration.json) and [profile summary](profile-summary.json) accompany the raw bundle.

Published measurements come from the regenerated corpus outside the sparse worktree. An earlier ignored scratch set was removed during sparse-checkout maintenance and is excluded from these results; the bundle records the regeneration. Source code and original RadFoam/Armadillo captures remained intact.
