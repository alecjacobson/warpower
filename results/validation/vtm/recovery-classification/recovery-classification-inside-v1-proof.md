# Private inside-only classification probe (#194)

Baseline: merged f338eb5 (PR #195). No production modification.

The private recovery extractor's consumers require centers, radii, physical
volumes/qualities, keep masks and errors. Exterior-band pruning only tests
whether signed distance is nonpositive. Containment does not consume distance.
Public full extraction, quality moves and final output retain actual distances.
The proposed private distance buffer has only a sign contract: shortcut
interiors write -1. Its numeric magnitude is explicitly unavailable.

A center is eligible only if max(abs(center)) < 1e140 and the identical
closest_triangle computation on the first current sorted leaf gives a finite
squared distance below the original (1e150)^2 bound. Current validated boxes
have finite float32 bounds, so every initial squared box distance is below
4e280, strictly below the full query's initial bound. Before any finite hit,
the original traversal therefore cannot prune the witnessed leaf. It reaches
that leaf or finds another finite hit first; strict comparison cannot replace
a finite hit with a nonfinite distance. The existing traversal stack bound
still applies. This is a floating-arithmetic success witness, not a nearest-
distance estimate. Domain producers must refit the tree as usual.

Only proved centers for which the existing material_side returns exactly 1
omit nearest traversal. The original query must then succeed. If its distance
is zero, the original forces side=1; otherwise it computes the identical
material_side=1. Both paths keep the cell regardless of tolerance and publish
nonpositive sign for exterior recovery. Geometry/volume/quality arithmetic is
unchanged. No threshold approximation or bounded outside query is introduced.

All other centers retain the complete nearest query, including huge/nonfinite
centers, unavailable witnesses, outside/ambiguous sides, zero tolerance and
strict d/R ties. A side already computed on a proved center can be reused if
the full distance is positive; exact zero still forces side=1. This avoids a
second ray query and preserves the original error predicate.

Pending: direct full-kernel oracle cases, caller/query attribution, changed-
input capture, full-solve pilot and independent geometry validation. Adoption
would additionally require public/standalone nonregression, full tests, native
capture and memory audits. No speed or correctness result is asserted here.
