# Current-query ordering experiment under issue197 (design only)

The production recovery uniform8K native trace at6178890 attributes678.071ms to
63 selection_cached calls,421.261ms to60 classify_cached calls and659.456ms to60
surface_error calls. These are diagnostic activity totals, not clean timings or
promised savings. Full fresh uniform replay is3.944s; the comparable-quality wide
CPU goal is still unmet. Cache/filter/cooperative experiments are already tracked
in197/198/159; avoid a duplicate issue.

A narrow next hypothesis is spatially ordering independent per-tetrahedron query
threads while preserving the scalar traversal and per-query arithmetic. No
cooperative traversal or approximate acceptance is needed for this probe. Begin
with cached classifiers, whose valid-input geometry and query outputs are written
by unique original cell index; keep reductions, compaction, cache-building and
output order at their original indices. Do not reorder surface_error initially:
it atomically adds quadrature contributions to a shared integral, so changing its
thread grouping changes floating-point reduction order. It requires separate
numerical/trajectory qualification, and no unchanged-bits claim follows.

Private static stages, all paid in the complete timed variant:
1. Prepare finite/validated current centers, using the existing geometry/query
   split prototype only after auditing uncertain-geometry and error handling.
2. Build current center Morton keys plus a bijective original-cell index map,
   with inactive/invalid slots represented safely. No key may cause an unchecked
   index or bypass the original error path. Domain boxes are only sorting bounds,
   not geometric proof. Large/outside/degenerate coordinates must remain safe.
3. Device radix sort in preallocated capture-compatible arrays; include key
   generation and sorting in total timing. Reuse established graph-safe sorting
   setup, not replay allocation or host count decisions.
4. Run exactly the existing scalar query/acceptance arithmetic in index-map order,
   scatter outputs back to original slots, and build the next exact query cache
   with its original semantics. Audit all writes and global error behavior.

First compare unsorted/sorted query-only cost and full preparation+sort+query on
frozen 1K/4K/8K/16K torus/Armadillo meshes, cache-hit/miss/reordered trajectories,
strict thresholds, ties, nonfinite errors and alternate streams. Report actual
query counts/cache states, sort cost, memory/setup and current source hashes.
Per-query output equivalence is necessary but not sufficient for adoption.
A full producer/refit/count/failure/recovery captured trajectory, independent
geometry/quality checks, clean complete solves, memory/lifetime audits and
standalone nonregression are required before any production change. A negative
result remains useful; avoid treating register count as measured occupancy.
