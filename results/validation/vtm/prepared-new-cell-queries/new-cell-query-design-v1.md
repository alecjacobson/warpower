# Issue 233: next hypothesis, not implemented

Current `ProvedCertificate._global_spheres` uses `check_unproved_spheres` on
point-parallel private completion. Each of 32 lanes constructs identical Sphere
and Box values, then scans original site IDs with the general Ball64 box test.
The private split-only profile measured about 832 ms over 554 calls; that is
historical diagnostic attribution, not a projected improvement.

Potential implementation: prepare current unproved cell geometry once, traverse
the existing shared current segment in 32-site groups, use the existing bounded
and prepared point predicates, and preserve full exact/symbolic fallback. A
current-input single-query cache can handle rare exact fallback, but should be
measured independently after the seed-filter milestone.

Important semantic differences prevent blind reuse of the seed checker:

- This certificate does not stop on an existing error bit. It must continue to
  collect invalid-predicate bit 2 and invalid-sphere bit 8; it does not publish
  live masks. Preserve this behavior, including already-rejected cells.
- `proved` comes from the winning cell source through deduplication and
  compaction. It cannot be inferred from a hash slot, old index or caller input.
  Current unproved cells require complete proof. The skip/check counters each
  increment once per cell, not once per lane.
- The private query counter currently counts calls after its existing box
  exclusions. Different certified filters may change this diagnostic. Either
  preserve its definition or explicitly define/test the new count. Do not
  silently treat it as a number of sites visited or all predicate operations.
- Segment permutation/bounds must be the current original-site data already
  rebuilt by the caller, with an explicit setup association and no registry.
- Setup-only allocation and a stable captured sequence are required. The seed
  filter's sphere, box, pending and index arrays may be reusable scratch after
  its query has completed, but capacity/ownership/lifetime must be proved first.
  This is a possible way to avoid another large persistent allocation, not an
  authorization to borrow arrays without checking every completion path.
- Keep standalone constructors and small private routes unchanged until their
  own clean measurements justify a change.

Test oracles should include per-cell rational physical/symbolic signs, exact
error-bit accumulation with preexisting errors, mixed proved/unproved cells,
compaction and duplicate winners, exact ties/overflow, invalid boxes, extreme
coordinates, changed device inputs and failed/recovered captured transactions.
Independent full-hull/material/surface checks, full solves and default API
controls remain required; a faster kernel alone does not establish an end-to-end
win or the wide comparable-quality CPU goal.
