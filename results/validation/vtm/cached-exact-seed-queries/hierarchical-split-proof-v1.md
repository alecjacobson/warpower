# Private hierarchical strict/exact split: semantic argument

This layout changes scheduling of existing predicates, not their arithmetic or
accepted signs. The production combined traversal remains the control. Both
passes use its current sphere boxes, prepared Ball64 data, segment tree, Morton
permutation, owners and bounded visit budget. These inputs do not change between
the ordered passes in a stream.

The strict pass executes the same traversal and duplicate/incident/point-box
exclusions. A strict bounded float32 or prepared Ball64 sign has exactly the
original effect on the cell's live flag. An unresolved point sets a local
uncertainty flag. At traversal completion each such lane atomically ORs its bit
into a per-cell uint32 mask, including unsigned lane31. Traversal-budget errors
retain the original fail-closed behavior. The pass never stops just because
another query has already rejected the cell.

The exact pass returns immediately only for a lane with no unresolved query.
Every flagged lane repeats the original traversal, strict signs and exact /
symbolic fallback on unchanged inputs. Repeating strict atomic-and rejection is
idempotent. Every unresolved original query therefore receives the same exact
policy, even if another query already rejected the cell. An error can still
short-circuit other work as in the original shared-error contract; valid meshes
cannot hide an unresolved sign by having a zero live flag.

The entire pending array resets before preparation on every invocation,
including inactive capacity and zero-count inputs. There is no variable queue,
compaction, capacity estimate or host-dependent dispatch. Both kernels retain
fixed launch dimensions and existing current-input proof semantics.

Evidence so far: 60 captured mask/dispatch cases pass, including conservative
invalid boxes, changed/permuted/unrelated inputs, ties, duplicates, scales,
empty/restored prefixes, poisoned masks, no fallback, all fallback and lane31
after prior rejection. 72 existing behavior tests pass; five production
class-name dispatch assertions are explicitly inapplicable to the private
replacement and excluded. Production remains unchanged. Complete timing and
independent saved-mesh validation determine whether this layout is useful.
