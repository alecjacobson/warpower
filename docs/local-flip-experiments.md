# Bounded local Delaunay repair experiments

The two private experiments in [issue #211](https://github.com/alecjacobson/warpower/issues/211)
repair selected moving-site meshes with strictly convex 2→3 and 3→2 flips.
Both make complete VTM slower, so neither is adopted. Production, independently
callable power-diagram and Delaunay functions, and the README flagship remain
unchanged at baseline `74ae277572d0a6e07cfc1442760f45b9178eee87`.

## Opportunity measured on actual trajectories

An observer checks previous connectivity against current sites, then runs the
unchanged production reconstruction. These counts include the first complete
solve and one fresh replay. Observer timings include instrumentation overhead
and are not performance evidence.

| Trajectory / workspace capacity | Calls | Already certified | Sphere violations only, unchanged owners and positive count |
| --- | ---: | ---: | ---: |
| Torus / 1,000 | 280 | 86 | 38 |
| Torus / 4,096 | 338 | 60 | 80 |
| Thin hand, seed 7 / 1,000 | 378 | 152 | 82 |
| Graded Armadillo / 4,096 | 344 | 78 | 102 |

Each larger solve also uses 1,024- and 2,048-capacity workspaces: together they
add 78 calls, ten already certified and no eligible sphere-only calls. Most
remaining calls have inverted cells, unsupported hull faces or missing sites.
Two additional hand calls have sphere-only errors but also lose flat cells and
are excluded from the last column. Seventy of the 86 already-certified torus
1,000 calls have numerically changed sites; this opportunity is not confined to
identical inputs. The numeric change counter treats positive and negative zero
as equal.

Forty representative snapshots retain the preceding points/connectivity, current
points and production reconstruction. All 80 preceding/reconstructed hulls pass
independent CGAL exact orientation, global empty-sphere, face-pairing, site-use
and hull-support checks. Those checks allow ordinary cospherical alternatives;
the device certificate additionally enforces its symbolic lifting policy.
The four observer solves report valid final output. Their final exported meshes
are retained, but the 80-check count refers to the snapshots, not those exports.

## Private implementation

Each proposal uses the existing filtered/exact orientation and symbolic sphere
predicates on original float32 positions. It requires a strict convex five-site
cavity, positive target tetrahedra and strict target sphere tests. A 3→2 proposal
also requires the same third tetrahedron across both remaining edge faces,
establishing the closed three-cell edge link in the indexed complex.

Proposals reserve every old cell with an atomic minimum of proposal IDs. Only a
proposal owning all its old cells survives. Separate buffers receive retained
and replacement cells after bounded prefix scans; neither the input positions
nor input connectivity is modified. Capacity exhaustion, iteration exhaustion,
invalid geometry or a stall produce zero accepted output. The VTM experiment
then invokes the existing reconstruction from its original state.

| Schedule | Before and between flip rounds | Before publishing |
| --- | --- | --- |
| v1 | Full original-coordinate certificate after every change | Last complete certificate must succeed |
| v2 | Positive-cell, coverage and indexed face checks each round; hull support before the first round | Full original-coordinate certificate must succeed |

v2 uses local sphere violations to drive proposals. A lack of proposals does
not establish success: remaining violations cause a stall, and a locally clean
candidate still receives the complete certificate. In particular, the existing
global fallback remains essential for flat hulls and overlapping cube covers
without a strict hull anchor. No caller-supplied trust mask or coordinate
perturbation is introduced.

These are deliberately private experiments, not a completed adoption contract.
The observer measures canonical-owner stability, but the prototype dispatch
does not separately enforce an unchanged-owner checkpoint; its current-input
coverage and final certificate remain mandatory. A future restricted fast path
needs that explicit guard, additional adversarial conflict tests and captured
invalid-input/recovery tests before adoption. Intermediate topology checks are
not claimed to certify arbitrary global complexes.

The strict geometric flip and its obstruction are described in Shewchuk's
[*Delaunay Refinement Mesh Generation*, §2.2](https://www.cs.cmu.edu/afs/cs.cmu.edu/user/jxu/quake/public/papers/delaunay-refinement.pdf).
In three dimensions a sequence of elementary improving flips can stall before
reaching Delaunay connectivity. The bounded fallback is therefore required.

## Correctness trials and the hand obstruction

Each version completes 35 private trials. The cases include both bipyramid
diagonals, exact cospherical choices, insufficient capacity, zero round budget,
inversion, a repeated cell, overlapping cube covers and 24 saved trajectory
inputs. All trials verify unchanged input buffers. Each version accepts twelve
outputs; all twelve pass independent global CGAL checks and retain their input
boundary. Six are saved torus/Armadillo cases: each matches production's
canonical tetrahedra after one or two flip rounds. The remaining saved cases
are rejected or stall; they are retained, not counted as successes.

The two eligible hand snapshots stall. Comparing their unaccepted candidates
with production identifies three four-cell replacements on six vertices. Each
has exactly equal rational volume and the same oriented cavity boundary, with
different interior diagonals whose four endpoints are exactly coplanar. A
strict five-site flip cannot pass through the corresponding flat intermediate
tetrahedron. [Issue #212](https://github.com/alecjacobson/warpower/issues/212)
tracks possible direct 4→4 handling. Production already reconstructs both cases;
this is a limitation of the private repair candidate, not a production failure.

## Complete-solve performance

Measurements use NVIDIA L40, Warp 1.19.0 and NumPy 2.2.6. Each number is the median
of three additional complete fresh resident solves, including sampling, fitting,
growth, four quality rounds, extraction and all failed repair/fallback work.
Setup, compilation, uploads, downloads and independent validation are outside
the replay timings. All first and timed replay windows have zero CPU quota
throttling. Some setup windows are throttled and are retained separately.

| Private schedule | Case | Production (ms) | Experiment (ms) | Change |
| --- | --- | ---: | ---: | ---: |
| v1, certificate each round | Torus, 4,096 | 1,788.460 | 1,860.725 | +4.04% |
| v1, certificate each round | Hand, seed 7, 1,000 | 1,463.198 | 1,912.653 | +30.72% |
| v2, final certificate | Torus, 4,096 | 1,788.280 | 1,842.563 | +3.04% |
| v2, final certificate | Hand, seed 7, 1,000 | 1,446.371 | 1,896.187 | +31.10% |

All eight complete outputs pass independent full-hull checks, material quality
and topology checks, boundary vertex-link checks, exact VTU round trips and
zero boundary self-intersections. Both torus pairs have bitwise-identical points
and identical canonical connectivity. Both hand pairs have identical material
cells, oriented boundary, roles and recorded optimization/refinement work. Tiny
coordinate differences (maximum about 4.4e-18 in v1 and 6.9e-18 in v2) accompany
different exterior full-hull cells in v1; v2's full-hull cell set agrees.
Both v1 alternatives independently validate. This is retained variation, not a
claim of bitwise reproducibility or a traced causal kernel; see
[issue #208](https://github.com/alecjacobson/warpower/issues/208).

An isolated alternating ten-sample comparison on one retained hand input finds
that the existing point-parallel complete certificate takes 1.771 ms versus
2.050 ms for its BVH route. Both accept the same 5,614 cells. Torus inputs select
the local certificate in either setting. This isolated result is not an adopted
change or a complete-mesher speedup, and does not justify the observed regressions.

The result also extends the earlier
[fresh previous-mesh certificate experiment](previous-complete-certificate.md):
successful speculative certification or repair is insufficient when its extra
work costs more than the reconstruction it replaces. Further work should test
sharing the current-input seed checks already performed by fragment reuse,
with strict internal proof provenance, rather than adding another independent
pipeline. No such integration is implemented here.

## Evidence and scope

The [manifest](../results/validation/vtm/local-flip-experiments/manifest.json)
binds private sources, observations, inputs, unsuccessful cases, raw samples,
outputs, independent checks and production source hashes. Reproduction needs
the pinned baseline, the recorded private modules restored under `.work/vtm`,
the referenced shared validators and the existing Armadillo input. Use fresh
output directories: the harnesses refuse overwrite. Export archives are lossless
gzip; native reference binaries can be rebuilt from `benchmarks/cpu`.

The first observer failed compilation because Warp does not support its vector
`!=` expression; the scalar-component revision passed. The first complete-pilot
checker rejected a throttled aggregate window including setup; the revised
checker preserves setup diagnostics and separately verifies the already
unthrottled first and timed replay windows. Both failed invocations are retained.

These unadopted experiments have no new full-suite, memory-sanitizer, native
trace, broad-gallery or standalone timing claim. Public runtime sources are
unchanged. The wide comparable-quality native CPU speedup goal remains unmet.
