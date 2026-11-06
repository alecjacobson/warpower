# Parallel global certificate experiment

The existing local certificate and original-site hierarchy are unchanged. When
local coverage cannot be proved, global checks on buffers of 128–512 sites
currently distribute each tetrahedron's site queries over 32 workers. The
upper limit follows the fixed-buffer grid benchmark: direct point scans improve
the 512-site case but regress the 1,000-site case, which retains the tree.
Larger and smaller buffers retain hierarchical traversal. No new arrays or
per-replay allocations are required.

Each worker scans a disjoint residue class of original site indices. It skips
canonical duplicates and defining vertices, applies the same conservative
circumsphere box and cached polynomial interval to the point, and evaluates the
same exact symbolic sphere predicate for every unresolved sign. Bounds skip only
strictly outside points. Invalid predicates and nonnegative symbolic signs set
the same shared error bits with atomic OR. Therefore the union of workers tests
exactly the same lower-facet condition as the original traversal, independent of
point order. Exact orientations, face incidence, support and site coverage are
still checked by the existing certificate.

For this path, `statistics[0]` counts unresolved symbolic predicate queries after
point bounds, and `statistics[1]` is zero because no tree nodes are traversed.
The legacy tree path counts visited leaf candidates before its final point
filter, so these branch-specific diagnostic counters should not be interpreted
as identical units of work. Output connectivity and error semantics are the
same; these counters are diagnostics rather than correctness evidence.

The runtime change passes all 410 existing tests. Another 28 differential tests
compare the old and new global checks on identical inputs, including very small
and very large float32 scales, subnormal tetrahedron heights, duplicate queries,
interior/exterior witnesses, and sizes around the dispatch threshold and warp
boundaries. Performance, padded memory/lifetime checks, and native replay audits
are in progress; this change is not yet adopted on main.

An initial profile of the separate hand-fragment completion experiment measured
1,083 ms in four global sphere passes versus about 10 ms in everything else.
Private tree partitioning reduced that completion to roughly 211 ms; direct
point partitioning reduced it to 52–58 ms. Both still returned the same complete
2,164-cell mesh in three rounds. This excludes constructing the fragment and
is not a complete VTM or CPU comparison. Current integration is tracked in
[issue #105](https://github.com/alecjacobson/warpower/issues/105).
