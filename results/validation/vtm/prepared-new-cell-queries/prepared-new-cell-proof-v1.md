# Private issue 233: first-step equivalence and storage ownership

Scope: current finite original sites, valid cell indices, canonical owners and
the rebuilt original-site segment/permutation used by private facet completion.
The primitive comparison additionally exercises flat defining cells to check
invalid-predicate error accumulation. No production adoption is claimed yet.

The original checker constructs identical Sphere/Box data in every query lane.
Preparing those same operations once per current unproved cell stores identical
binary64/binary32 values. Module options still disable fused operations and fast
math. Proved cells are skipped without reading the corresponding geometry.

A segment node encloses its descendant coordinates. Sphere Box exclusion uses
strict componentwise interval separation. If it excludes a node, the identical
point-box test would exclude every descendant; none would increment the old
query count. Unexcluded leaves map consecutive Morton ranks into 32 lanes, with
the same owner/incidence checks and original general Ball64 outside/symbolic
predicates. The changed traversal therefore preserves the queried set and its
counter for each given input cell/proof mask. An invalid sphere Box cannot prune.
The existing successor visit bound covers the entire implicit binary tree; an
unexpected unfinished traversal fails closed.

Unlike the seed filter, neither preparation nor queries stop merely because an
error bit already exists. The checker keeps collecting invalid-predicate bit 2
and nonempty-sphere bit 8. Skipped/checked proof counters increment exactly once
per active cell. Integer additions and bitwise OR do not depend on query order.
Counter definitions are unchanged; this is not a promise that unrelated existing
atomic duplicate-winner scheduling yields identical proof masks in all runs.

The integration probe borrows the seed checker's Sphere/Box arrays only where
its current segment association is available and its capacity covers every
completion cell slot. A different-capacity private caller retains the old path.
The seed filter completes preparation, strict/exact checks and compaction before
completion starts. Its sphere data is dead at that point. Each certificate then
prepares the current unproved cells after proof compaction and before reading
those entries. Every later seed invocation refreshes its own active geometry.
No unchanged-data proof is carried across a call or input update.

Arrays remain strongly referenced by both owners; the adapter owns no new device
arrays and creates no parent-reference cycle. Kernel launches stay in the ordered
captured stream. Each certificate receives its own helper bound to its seed
buffers; there is no global segment registry. Unsupported/small routes keep the
original checker. The public diagram/Delaunay code is untouched.

The 36 primitive cases capture six changing-input/count/proof/error phases and
four modes: current checker, prepared hierarchy, flat checker with sphere-box
pruning disabled, and prepared hierarchy with sphere boxes invalidated. They
compare exact error, query and proof counters; small cases also use independent
rational physical/symbolic oracles. Geometry is poisoned before each prepared
pass, counts include zero, proof flags include all/mixed/none, preexisting errors
are nonzero, and recovery repeats the original phase. Both block sizes, scales
2^-60/1/2^60, ties, duplicates, flat cells, partial groups and 16K points are covered.

Whole-workspace integration, performance, independent geometry, memory lifetime
and native graph audit remain separate gates. The seed buffers are private
scratch, not a public post-completion geometry output.
