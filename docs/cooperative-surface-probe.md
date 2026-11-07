# Cooperative surface-query experiments

These private experiments on `305758fac0575cfb4c7ded604df02a52560905ce`
measure surface queries and ordinary retained-cell classification on NVIDIA L40,
Warp 1.19. **No prototype is adopted.** Production kernels, public APIs, storage
and defaults are unchanged. Issues [#197](https://github.com/alecjacobson/warpower/issues/197)
and [#159](https://github.com/alecjacobson/warpower/issues/159) remain open.

The full classifier's final-mesh cost is mostly queries: preparing geometry
takes 0.011–0.017 ms across the four snapshots. The earlier full-classification
split remains [unsuccessful](extraction-split-probe.md). The new experiments
attribute ray versus nearest work and examine the ordinary retained classifier,
which skips or bounds many nearest queries.

## Ray arrangements

The four-triangle leaves can distribute triangle work across two or four lanes.
Each lane follows the same original stackless ray traversal and evaluates its
assigned triangles with the original floating-point expressions. Integer sums
combine crossings; a positive sum of ambiguity flags implements their OR. The
excluded-face correction is applied once. All lanes reach every tile collective,
including inactive lanes in the final block. The original three direction
retries remain conditional on the combined ambiguity flag.

A second arrangement assigns disjoint subtrees to 2–32 lanes. It tests ancestor
boxes before descending, assigns an early leaf to only one lane, and stops each
subtree at its original escape link. This arrangement is slower on the larger
measured batches, although it helps the separate 64-query batch.

There is no triangle-geometry cache. The rejected [#151](https://github.com/alecjacobson/warpower/issues/151)
cache changed near-boundary results under different caller contraction settings.
Integer reductions alone do not establish equivalence of floating-point leaf
decisions, so these prototypes reuse those stronger fixtures.

| Static batch | Queries | Original rays ms | Leaf 2 ms | Leaf 4 ms | Subtree 32 ms | Original nearest ms |
|---|---:|---:|---:|---:|---:|---:|
| Final torus1000 centers | 5,826 | 0.625 | 0.368 | 0.332 | 0.913 | 3.503 |
| Final torus4096 centers | 24,519 | 1.393 | 1.175 | 1.235 | 3.518 | 7.736 |
| Final Armadillo uniform4096 centers | 24,721 | 4.180 | 3.266 | 3.439 | 9.197 | 17.142 |
| Final Armadillo graded4096 centers | 24,667 | 4.617 | 3.467 | 3.901 | 9.906 | 18.501 |
| Random points, 2,304-face torus | 100,000 | 5.377 | 4.234 | 3.972 | 10.118 | 9.834 |
| Random points, 9,216-face torus | 100,000 | 7.536 | 5.898 | 5.569 | 14.069 | 17.583 |
| First 64 graded Armadillo centers | 64 | 1.405 | 0.709 | 0.425 | 0.369 | 5.127 |

These are separate warmed, captured query kernels with fixed prepared inputs,
three warmups and 15 interleaved samples. Every timing window is CPU-quota clean,
and every candidate side matches the scalar reference. Preparation and setup are
excluded. Separate kernel times must not be added to infer complete-solve cost:
compiler context, conditional work and resource use differ in fused callers.
All full-nearest results in these snapshots are finite and nonzero.

## Retained-cell classification

A fused prototype duplicates physical geometry calculation within each lane
group, shares ray work, then lets the group leader execute the original bounded
nearest fallback and write the keep/quality outputs. Uncertain geometry retains
the original exact fallback. This prototype adds no persistent arrays.

A staged prototype computes geometry once per cell, then runs a separate query
kernel. It uses **32 bytes per cell capacity** for current float64 centers/radii,
plus existing keep/quality arrays. Both launches are included below; geometry is
refreshed each time. It does not cache classifications across inputs.

| Final mesh | Original retained ms | Fused 4 lanes ms | Staged scalar ms | Staged 2 lanes ms | Staged 4 lanes ms |
|---|---:|---:|---:|---:|---:|
| torus1000 | 0.873 | 0.586 | 0.828 | 0.596 | 0.578 |
| torus4096 | 1.804 | 1.694 | 1.803 | 1.662 | 1.853 |
| Armadillo uniform4096 | 4.538 | 3.873 | 4.505 | 3.694 | 4.056 |
| Armadillo graded4096 | 4.970 | 4.162 | 4.932 | 3.897 | 4.309 |

The staged measurements have a separate fresh control (0.874, 1.805, 4.540 and
4.973 ms respectively). Both batches use three warmups and 20 interleaved
samples, with clean quota windows and bitwise-equal errors, keep masks and
quality arrays. Staging alone contributes little on these snapshots. Staged
failure semantics and complete solves have **not** been validated; the extra
scratch is not justified by an end-to-end result yet. Register pressure and
occupancy were not measured in this experiment.

Two fresh-process complete torus pairs test only the fused four-lane prototype:

| Complete solve, five warmed replays | Original ms | Prototype ms | Time reduction |
|---|---:|---:|---:|
| torus1000 | 618.107 | 593.933 | 3.91% |
| torus4096 | 1,815.755 | 1,803.988 | 0.65% |

Each replay includes a fresh complete solve with refits, sampling, growth,
optimization, quality work and extraction. First and timed quota windows are
clean. Prototype kernels were preloaded before the setup timer; there is no
setup-speed claim. The small 4k difference needs stronger confirmation before
claiming a useful general gain.

Both pairs preserve positions bitwise, input arrays, roles, full/material tet
sets, oriented boundary faces, parameters and recorded solver work. The two
candidate meshes pass independent CGAL global empty-sphere, original-coordinate
orientation, face pairing and convex-hull support checks, NumPy quality,
CGAL AABB sampled distances, and boundary vertex-link checks. No VTU export or
boundary self-intersection audit was performed for these two pilots.

## Validation and remaining work

- Ray comparisons cover six shapes, scales `2^-60`, `1`, `2^60`, and both caller
  FMA settings: 36 configurations with 17,196 query rows. All eight arrangements
  match raw crossing counts, ambiguity flags and final side decisions, both in
  a diagnostic caller and a side-only caller. Fixtures include random points,
  vertices, edge midpoints, face centers and excluded-face orientation queries.
  Independent solid-angle winding checks the 100 ordinary points per configuration.
- A seven-replay device producer/refit/query graph covers deformation, collapsed
  geometry, NaNs, reflection and recovery. All arrangements match on the four
  valid phases, including the independent winding oracle. Invalid phases check
  the public query's rejection; no material-side meaning is asserted for raw
  queries on an invalid domain. This is a functional capture check, not a native
  allocation/transfer trace audit.
- All 27 existing selective/current-selection tests pass with the fused
  four-lane classifier, including captured input changes and recovery. A further
  53 direct comparisons preserve successful arrays bitwise, errors and inactive
  padding across scaled clouds, exact boundary centers, strict threshold
  neighbors, exact-geometry fallbacks and poisoned/invalid inputs.
- The original tile-shape and positional-launch mistakes, and the initial 27
  adapter failures caused by forwarding `outputs=None`, are retained in the raw
  logs. Corrected runs use the same scientific fixtures; no runtime fix or test
  relaxation was needed.

No full regression-suite run, sanitizer audit, native graph trace, complete scan
comparison or standalone API benchmark was performed for adoption. Production
is unchanged, so this publication does not claim those gates. Before adoption,
complete the relevant correctness, memory, capture, quality and performance
checks and account for setup and any additional storage.

Nearest-query work limits the full extractor's benefit from ray cooperation.
[#198](https://github.com/alecjacobson/warpower/issues/198) tracks conservative
surface-bound filters, beginning with node/leaf work attribution and a proof
against the existing floating-point predicates. The quality-matched wide CPU
outperformance goal remains unmet.

The [manifest](../results/validation/vtm/cooperative-surface-probe/manifest.json)
hashes 60 immutable files: source generators and candidates, successful and
failed logs, raw timings, paired meshes and independent checks. Scripts preserve
their original `.work/vtm` paths and commands as experiment records; native
validators come from [benchmarks/cpu](../benchmarks/cpu/README.md). Input snapshot
paths and hashes refer to the preceding published recovery-classification data.
