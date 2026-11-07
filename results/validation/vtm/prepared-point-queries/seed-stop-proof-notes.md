# Preconditions and proof obligation for #169

The current reuse entry point runs _validation.validate over every original
point/weight in _validate_reuse, then capture_if gates _attempt_reuse. Nonfinite
producers never reach the hierarchy or seed queries. CertifiedSeeds.seed_keys
checks indices/duplicate owners/zero volume; unique_pool rechecks exact
orientation, orients every emitted cell positively and propagates failure.
Count/capacity failures and per-run live resets precede the prepared checker.

For finite original float32 coordinates and the finite hash/unit symbolic
weights, the existing 1,536-bit integer predicates have documented intermediate
bounds below 1,400 bits (docs/integer-predicates.md). Physical predicates are
therefore representable. For a positive-volume defining tetrahedron, the final
symbolic cofactor sequence cannot have every orientation zero. Thus a query on
valid internal inputs has an exact valid sign; this is not an assumption about
arbitrary malformed direct calls to the private checker.

Each live flag is reset by the existing reset_live kernel immediately before
the query pass. During that pass it changes monotonically from 1 to 0, solely
after a current-input rejection or an observed predicate failure. The prototype
uses an atomic zero-add to read the same atomic flag, after the cheap sphere-box
exclusion. Returning ends only this candidate lane; there is no tail of other
candidate cells assigned to that thread. Any candidate retained with live=1 has
had every canonical non-defining query checked or conservatively excluded.
Stopping a candidate already rejected cannot turn an unproved cell into a
retained one. Every observed predicate failure still sets the batch error before
clearing the live flag; final compaction/completion retains its failure policy.

The earlier split-checker experiment deliberately preserved exhaustive query
error handling in its second pass. This experiment instead relies explicitly on
the validated finite-input/positive-cell invariants above and stops rejected
cells. It must not be exposed as a generic unchecked-predicate API or advertised
as validating unsupported private inputs. Public invalid-input/recovery tests
remain required, as do direct current-input mask comparisons before adoption.

No production source has changed yet. Extra atomic traffic may regress valid
cells; clean complete-solve pilots decide whether broader work is worthwhile.
