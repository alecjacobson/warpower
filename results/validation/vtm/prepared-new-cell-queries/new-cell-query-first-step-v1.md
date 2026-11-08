# Issue 233: narrower first experiment (unimplemented)

Before changing point predicates or splitting exact work, test only shared
preparation plus the current segment traversal. Keep the original point-box,
general Ball64 `outside(shape,p,p)` and `symbolic_sphere` functions unchanged.
This avoids introducing a new private query-counter definition in the first step.

One preparation thread writes Sphere/Box for each current unproved cell; proved
cells skip preparation. The query retains the no-early-error behavior and maps
32 lanes onto the current segment's 32-site groups. Every retained query uses
the original point tests. A once-per-cell counter records proved/unproved cells.

Required proof: the current segment encloses its descendant points, and a node
excluded by the existing sphere Box would exclude every descendant through the
same point-box test. Therefore pruning removes zero queries that the original
checker would have counted. The shared Sphere/Box construction must preserve
the same floating-point operations/options; only lane scheduling changes.
All query counts and accumulated error bits should then match exactly.

This first variant still contains the expensive exact routine and may keep 255
registers. Its possible benefit is avoiding 32 identical geometry preparations
and most flat point visits. Measure it before combining independent changes.
Only after this is understood should bounded point predicates or a cached exact
pass be added; those can alter the diagnostic query count and need separate
semantics, tests and timing.

An isolated primitive prototype can establish arithmetic/mask/error equivalence
before solving setup association and scratch ownership in ProvedCertificate.
Do not modify the current runtime until PR234 qualification/publication/merge
finishes. The new-cell-query-design-v1.md ownership and provenance constraints
remain applicable.
