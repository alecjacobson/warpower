# Issue228 integration follow-up after the successful private v2

The private v2 completes graded/uniform8K plus the previously valid graded4K
control. All three independently validate. Two tangent rings have actual binary32
displacement bounded by10/20% of the shortest full-mesh incident edge,16directions
each. Graded accepts attempt1; uniform22. The strict global fitting rule stays
unchanged; the separate quality transaction retains the existing1% fit/volume
bounds, strict deficit improvement, nondecreasing minimum quality, expected input
topology, every original material reference and all original feature roles.

Issue239 isolates a later existing proof bottleneck and is being qualified on a
separate branch before this recovery is promoted. Main does not yet contain this
recovery. Do not count experimental proof benchmark rows as a merged recovery.

Proposed production scoping to review and test:
- Stop globally monkeypatching ExteriorBandRecovery. Pass a private witness buffer
  only to the full mesher's main extraction, not to standalone extractors or every
  quality workspace. No standalone diagram/Delaunay source or allocation changes.
- Clear that buffer at the beginning of an actual band attempt, inside its existing
  conditional branch, then capture a pinched role1 vertex before a rejected trial
  rolls back. Successful/no-op extraction need not pay another witness kernel.
- Eligibility must additionally require the main band's CURRENT rejected flag.
  Its existing begin kernel resets that flag on every run. This prevents use of
  an earlier witness when a later extraction skipped the band attempt. Each actual
  attempt resets the witness before any possible rollback. Fresh solve resets the
  recovery's published status; audit invalid-domain/target skips for stale state.
- Keep bounded current degree/twin traversal and deterministic minimum eligible
  witness. Never hardcode the observed sites6598 or3406.
- Candidate extraction keeps containment and ordinary band recovery but omits its
  optional proven-exterior search. Final acceptance still requires a valid full
  Delaunay certificate and all topology/quality/reference/role guards.
- Exhaustion restores the original points/active count and reconstructs, evaluates
  and refits to refresh all private indices/certificates. Restoring public cell
  rows alone is not enough. Audit core selection invalidation explicitly.
- Avoid copying an unused old-cell checkpoint in the new transaction if its
  rollback always reconstructs; any scratch reuse/deletion needs a concrete
  lifetime proof. Do not change ordinary QualityMove storage on speculation.

Required before adoption: captured forced-exhaustion/rollback then successful
recovery; invalid-target/domain and subsequent valid frame; nondefault stream,
stable buffers and no capture/replay allocation/readback; independent full/material
mesh checks; native and padded/pool-disabled memory audits; clean successful-path
and standalone performance plus explicit setup/persistent-memory costs. A finite
32proposal search can still fail: retain explicit failure/zero-output semantics.
