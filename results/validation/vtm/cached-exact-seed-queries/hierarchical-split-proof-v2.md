# Private issue 175: one-query fallback cache

This experiment changes only scheduling/storage of the current stackless seed
queries. Production remains at PR 231 while tests and timing are pending.

The strict pass visits the same groups and checks the same owner, incident-site,
point-box, bounded binary32 predicate and prepared Ball64 predicate as production.
A certified positive value rejects the cell; a certified negative value continues.
It never skips a query merely because another lane has rejected the cell.

For each lane, the first query unresolved by both strict filters is recorded.
A second unresolved query replaces that index with -2; subsequent unresolved
queries preserve -2. A pending bit is published if and only if at least one
unresolved query was visited. The ordered exact kernel reads only flagged lanes.
A nonnegative saved index invokes the unchanged symbolic predicate on that query;
-2 performs the original complete lane traversal and fallback policy. Other
negative indices or out-of-range indices fail closed with error bit 16.

All owner, incidence, sphere, box and index evidence refers to the same input
within this invocation. Kernels in the captured stream execute in order, and
inputs cannot change between these passes. Each invocation clears the entire
pending array. Every lane that publishes a pending bit first writes its saved
index; inactive/unflagged entries are never read. Therefore the saved array need
not be cleared, and stale bytes from earlier invocations cannot authorize a query.
Tests poison that storage and change active counts and input coordinates.

The cache is allocated at setup as 32 int32 entries per candidate cell, with no
runtime host work or variable queue. At 131072 candidate cells this adds 16 MiB.
It is a bounded cache, not a cap on exact queries: overflow takes the full rescan.

The overflow regression fixture uses a unit-cube cosphere and identity permutation.
A first unresolved query in lane 4 has exact symbolic negative sign and keeps the
cell. A second query in that lane (index 36) has positive sign and rejects it.
The independent Fraction oracle confirms both outcomes. Thus dropping the second
query would change the accepted mask. Separate cases cover unsigned lane 31,
all 32 overflow lanes, rejected cells, zero/restored counts, both block sizes,
invalid sphere boxes, extreme scales and captured device producers/consumers.

No additional arithmetic error bound or predicate assumption is introduced.
Performance benefit, full integration, native capture audit and memory safety
remain separate adoption gates. The previous split-only experiment's 7.126%
structured-grid regression remains part of the record even if this revision helps.
