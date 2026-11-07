Integrated runtime f8b5cfa passes all 650 tests, including seven new larger-route/provenance cases. Final independent checks pass for ten complete-solve pairs: bitwise coordinates and roles, identical full/material cell sets and oriented boundaries, identical iteration work and inputs; CGAL global full-hull predicates, NumPy/AABB quality and boundary links pass. The three gallery exports also have exact VTU round trips and zero CGAL surface self-intersections.

Complete fresh solve medians (ms), before → integrated after:
- Torus 2,050: 3287.902 → 1490.729
- Torus 3,072: 4620.510 → 2472.733
- Torus 4,096: 7065.010 → 3787.116
- Graded torus 4,096: 7730.856 → 4237.962
- Armadillo uniform 4,096: 13732.105 → 4635.680
- Armadillo graded 4,096: 15205.459 → 8176.069

The 1k default, 4k reserved capacity with 1k/2k targets, and 8k reserved capacity with a 4k target vary by less than 0.15%. Five standalone diagram/Delaunay full-operation pairs vary by less than 0.6%, with their inputs/counters unchanged. All 26 final timing jobs are unthrottled. Source isolation confirms all Warp kernels/functions and every other production module are unchanged.

The extension costs 34,106,983 additional persistent pool bytes at 4,096 reserved slots, including when a smaller target never enters that route. Storage and separate setup/first execution costs will be published.

PR #163 remains a draft pending the running padded memory/lifetime and native capture audits, followed by evidence publication. These are GPU version improvements; the CPU performance target in #40 remains open.
