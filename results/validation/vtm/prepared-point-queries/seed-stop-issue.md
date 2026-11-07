After #168, the complete 4k torus native trace attributes 736.293 ms across 208 calls to the private prepared seed checker, ahead of hull support (454.475 ms) and exposed-face search (313.394 ms). Current prepared seed lanes continue scanning after an exact/symbolic test clears their candidate's live flag.

Investigate stopping remaining lanes once the same current-input candidate has been rejected. A retained candidate must still pass every canonical-site test not conservatively excluded by the existing bounds. A discarded candidate needs no further emptiness proof. Keep the original geometry preparation, ownership, symbolic policy, upstream finite-input/count validation and final completeness certificate. The per-run live reset is essential; no previous-input rejection may survive replay.

Start with an isolated private checker that atomically reads the monotone live flag after the cheap sphere-box exclusion, before the expensive interval test. Return only the current candidate lane; this kernel assigns one candidate per thread group and has no strided tail of other candidates to check. Compare polling positions/frequency only if measurements justify it, since extra atomics may outweigh saved work. The ordinary seed checker and standalone APIs stay unchanged during exploration.

Acceptance:
- Prove that retained-cell coverage is exhaustive and current-input flags are race-free/reset correctly. Preserve public fail-closed behavior for invalid input, counts, predicate failures and capacity failures; state the internal finite-input precondition explicitly.
- Compare accepted candidate sets with the original exhaustive checker and independent symbolic/rational references, including late witnesses, cross-lane rejection, exact ties, duplicates, extreme scales, empty/restored prefixes and changed producers.
- Measure clean fresh complete solves and iteration work at 1k/4k, then graded/scanned/feature examples if promising. Preserve failed and rejected probes.
- Require independent full-mesh/quality checks, full tests, memory/capture scopes and standalone performance gates before adoption. No host decisions, new frame-to-frame trust or allocations in replay.

Tracked under #29/#40/#43/#87. #167 remains a separate unimplemented split-filter experiment. No implementation or speed gain is claimed yet; the wide CPU goal remains unmet.
