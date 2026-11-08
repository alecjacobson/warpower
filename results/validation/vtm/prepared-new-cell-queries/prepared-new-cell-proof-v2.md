# Private issue 233: strict/exact split with unchanged query counters

This extends the current-input geometry and segment exclusion proof in v1.
It is a private experiment until full integration and performance gates pass.

The strict pass applies the original point-box and general Ball64 `outside`
checks before incrementing the original query counter. Additional strict
physical-sign filters happen only after that increment. The count still means
queries remaining after the original exclusions, not exact kernel invocations.
Proof counters increment once per active cell in this pass alone.

Both additional filters already exist in production seed filtering: the bounded
binary32 determinant and prepared Ball64 `point_value`. A strict physical sign
agrees with the unweighted determinant used first by `symbolic_sphere`; no
symbolic perturbation is needed on a nonzero physical determinant. Unsupported
ranges or overlapping error bounds continue to the original symbolic predicate.
No new floating-point error bound or tolerance is introduced here.

For each cell/lane, cache the first unresolved site index, or -2 when more than
one remains. Reset the entire pending mask array before every checker invocation.
Only flagged lanes read cached indices. Exact dispatch uses the saved query if
unique, or rescans every eligible query when the lane overflowed. No inputs,
owners, cell indices, proofs or hierarchy mutate between the ordered passes.
The rescan does not increment counters again. Malformed scratch/unfinished
traversal fail closed with bit 2. Original exact invalid and rejection bits
remain 2 and 8. Neither strict nor exact kernels stop on an existing error bit;
this is essential for flat cells, already-rejected cells and preexisting errors.
Unsigned lane masks preserve bit 31.

The integration borrows consumed seed Sphere/Box/pending/index arrays after its
strict/exact queries and compaction finish. The original seed invocation resets
its own masks/prepares its geometry every time; certificates likewise reset
masks and prepare current unproved geometry. The segment is read-only during
both consumers. This adds no device arrays. Capacity and current permutation
association are checked at setup; unsupported paths retain the old checker.

Tests retain v1's captured changing-input equivalence checks, with both pending
and index scratch poisoned before each invocation. Additional captured dispatch
fixtures require lane-31 fallback after a definite rejection, all lanes with
multiple cospherical queries, and an overflow whose first exact query accepts
but second rejects. They compare exact error/query/proof counters against the
old checker and rational acceptance, with zero/restored counts and initial
errors 0/64. Small rational checks also cover fully flat cells and exact ties.

Passing tests is evidence in addition to this argument, not a replacement for
current-data ownership, ordering, capacity and arithmetic prerequisites.
