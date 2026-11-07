# Existing-segment stackless query probe

The private #177 experiment does not establish a useful complete-solve speedup
over the merged [flat Morton groups](grouped-seed-queries.md). Production keeps
the flat-group implementation. The source and results are retained for later
work; no standalone behavior or README flagship changes.

The prototype reuses the current original-coordinate SiteHierarchy segment tree.
At group base max(1, leaf_base/32), heap nodes bound consecutive 32-point Morton
groups. All 32 query lanes follow the same node sequence: descend left on an
overlapping internal node, otherwise ascend through right-child ancestors and
visit the next sibling. It uses no traversal stack. A limit of 2*group_base−1
visits covers the full binary tree and exhaustion fails closed. Group ranks
outside the point buffer are masked; surviving points retain the merged Ball64
and symbolic checks, including exhaustive checking after rejection.

The segment bounds are conservative, including their finite empty-leaf
sentinels; a populated group's ancestors contain all its points. Invalid sphere
boxes prune nothing. A private setup registry routes the matching point and
Morton-ID buffers to their existing hierarchy, with pointer/capacity assertions.
This registry is experimental setup code; any adoption would pass the segment
explicitly through the private constructor.

## Measurements

Fresh controls use main ea1c00a, with runtime b5d369c, on NVIDIA L40 / Warp 1.19.0.
Each solve includes refit, sampling, growth, fitting, quality work, extraction
and diagnostics. Allocation, compilation, transfers and offline checks are
excluded from replay. All first and measured quota windows are unthrottled.

| Complete fresh torus solve | Flat groups, ms | Stackless, ms | Change |
|---|---:|---:|---:|
| 1,000 sites, five replays | 703.179 | 704.703 | +0.22% |
| 4,096 sites, three replays | 2,400.853 | 2,398.479 | −0.10% |

These small differences do not support a speedup claim. The prototype removes
the two group arrays and their preparation launch, reducing persistent pool
storage by 768 bytes at 1k and 5,376 bytes at 4k. Warm-cache setup takes
1.550→1.508 seconds and 2.932→2.900 seconds; first executions take
944.907→934.826 ms and 3,199.800→3,174.098 ms. These are single setup/first-run
observations, not a repeated startup comparison. In the preceding flat-group
native trace, all 208 group-preparation activities total only 1.106 ms.

Both final meshes are bitwise identical to their controls, including positions,
full/material cell sets, oriented boundary, site roles, inputs and parameters.
All iteration and reconstruction work counters match. Independent CGAL exact
global full-hull checks and NumPy/AABB material quality checks pass, including
single-cycle boundary-vertex links.

## Checks and scope

The private variant passes 69 existing reconstruction tests in 53.75 seconds.
A further 76 cases pass in 39.97 seconds: 24 traversal tests compare every
visited node and point against independent recursive traversal over 69 pruning
patterns; 16 test current segment containment for extreme finite coordinates,
permutations, partial/padded groups and poisoned old bounds; 36 compare captured
query masks with exhaustive and small rational symbolic references. These mask
tests also force invalid sphere boxes to exercise complete traversal.

A separate 36-case comparison against the merged flat-group checker passes in
39.38 seconds. It retains changing coordinates, duplicate owners, shrinking,
empty and restored candidate prefixes, scales 2^-60 through 2^60, and block sizes
32 and 256. Traversal sizes range from one to 4,097 points, including both sides
of group and power-of-two boundaries.

The [manifest](../results/validation/vtm/stackless-seed-probe/manifest.json)
records all prototypes, commands, test logs, timing samples, meshes, independent
checks and source hashes. No broad gallery, adoption sanitizer or native-capture
audit is claimed for this unadopted probe. The production implementation remains
the already validated flat-group path. Larger-capacity reuse is not tested here.

Tracked in [#177](https://github.com/alecjacobson/warpower/issues/177).
The wide CPU outperformance goal remains unmet.
