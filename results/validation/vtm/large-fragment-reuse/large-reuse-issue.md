The complete 4,096-site torus still takes about 7.07 s on L40 after #158. Its captured trajectory routes 27 reconstructions to the 1,024-site child, 14 to the 2,048-site child, and 170 to the full constructor. Fragment reuse is limited to 2,048 sites because its current seed and facet queries scan every site; that leaves most of the larger trajectory rebuilding from scratch.

Measure whether certified fragment reuse can extend beyond the existing cutoff. Start with a private opt-in of the existing reuse implementation on the full 4,096-site path; this establishes the actual scaling cost before designing a new query kernel. If needed, use the current hierarchy and conservative current-input sphere bounds to prune groups of queries, retaining the exact symbolic predicate for every unresolved point. Consider cooperative traversal over disjoint subtrees. No prior validity or unchanged tet corners alone may substitute for current-input certification.

Acceptance:
- Preserve fresh sampling, canonical duplicate ownership, original-coordinate exact/symbolic policy, full convex-hull output and bounded ordinary-constructor fallback.
- Validate complete actual fitting trajectories at 4k and larger sizes; report all invalid, timeout, rejected and slower variants, setup/storage cost, and work/mesh comparisons. Include graded and imported domains before any default extension.
- Test changing graph inputs, growing/invalid active prefixes, co-spherical and degenerate data, invalid/recovery phases, and capacity failure. All numerical decisions and scratch remain on device and preallocated for capture.
- Extend routing/default thresholds only with measured complete-solve gains, independent geometry checks, capture and memory/lifetime audits, and unchanged standalone diagram/Delaunay performance.

This is a scaling experiment under #29/#40/#43/#87, complementary to cooperative cell construction #98. Scratch duplication #122 remains separate. The current private extraction-classification split (#159) shows no meaningful gain at either 1k or 4k, making larger reconstruction work the next target; no CPU speedup is claimed.
