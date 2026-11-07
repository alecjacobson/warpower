Private pilots on current main 6469ab6 (NVIDIA L40 / Warp 1.19.0), with unchanged production runtime:

| Variant | Sites | Fresh control (ms) | Prototype (ms) |
| --- | ---: | ---: | ---: |
| Point-specific prepared enclosure | 1,000 | 735.419 | 726.861 |
| Point-specific prepared enclosure | 4,096 | 2,827.083 | 2,696.968 |
| Analytic first, prepared enclosure fallback | 1,000 | 734.613 | 723.422 |
| Analytic first, prepared enclosure fallback | 4,096 | 2,825.663 | 2,740.336 |

Complete fresh resident solves, including refit/sampling/growth/fitting/quality/extraction; five measured samples at 1k and three at 4k. All initial and measured quota windows are unthrottled. Every pair has bitwise equal coordinates/roles, identical full/material tetrahedron sets and oriented boundaries, identical inputs/parameters and iteration work. Independent CGAL global full-hull certification and NumPy/AABB quality and boundary-link checks pass for all four.

Both variants pass the 69 existing reconstruction tests. Four new tests check the complete point-expression error enclosure against rational determinants, including full finite float32 bit patterns, subnormals, extremes, translations, ties and thin cells. A further 36 captured producer/filter/consumer tests compare both variants and #169's rejected-cell stop with the original exhaustive checker: changed geometry/canonical ownership, shrinking/empty/restored prefixes, poisoned prior live flags, cube/octahedron ties, thin/random/fitted clouds, scales 2^-60 through 2^60, block sizes 32/256. All masks/errors agree; small cases also match independent rational symbolic masks.

Next: evaluate using the prepared point enclosure for both strict signs (the current point variant only excludes strictly outside queries), separately from #169's stop. These are private experiments; no default change or adoption memory/capture audit is claimed yet. CPU outperformance remains unmet.
