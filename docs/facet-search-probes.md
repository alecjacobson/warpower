# Facet-search pruning probes

Two conservative pruning variants preserve the tested meshes but make complete
VTM solves slower. Neither is adopted. Production remains at the
[certified fragment-reuse implementation](large-fragment-reuse.md).

The current 4,096-site torus trace attributes 1,273.260 ms across 326 launches to
exposed-face candidate search, about 32% of its 3,986.300 ms kernel activity sum.
That made search a useful target; these diagnostic durations are separate from
clean complete-solve timings.

## Variants and complete solves

The first private implementation partitions the current canonical-site hierarchy
into 32 disjoint subtrees. Conservative face-halfspace bounds prune ineligible
nodes. After finding an incumbent opposite site, its current sphere box and
interval sphere bounds prune nodes that cannot improve the winner. Unresolved
points keep the original exact orientation and symbolic sphere tests; traversal
overflow fails closed. The second variant keeps the original strided scan and
uses only the current incumbent's conservative sphere box.

The existing [ordered lifted-plane argument](facet-completion.md#why-the-face-search-is-ordered)
explains why a strictly excluded query cannot improve a later, better incumbent.
No uncertain or tied query is pruned, and all completion/certification requirements
stay unchanged. Both variants are private to the reuse-owned completion.

Fresh before/after pairs use NVIDIA L40 / Warp 1.19.0, baseline `5f3e6d7`.
Each complete replay includes domain refit, fresh sampling, growth, fitting,
quality work and extraction. The 1k cases have five measured replays; the 4k
cases have three. All initial and measured CPU quota windows are unthrottled.
Private query compilation occurs before timing; no setup-speed claim is made.

| Variant | Torus sites | Control (ms) | Prototype (ms) | Change |
| --- | ---: | ---: | ---: | ---: |
| Hierarchy and sphere bounds | 1,000 | 822.840 | 1,253.627 | +52.4% |
| Hierarchy and sphere bounds | 4,096 | 3,784.700 | 5,660.397 | +49.6% |
| Strided scan and sphere box | 1,000 | 824.010 | 945.173 | +14.7% |
| Strided scan and sphere box | 4,096 | 3,784.984 | 4,228.748 | +11.7% |

Each pair preserves bitwise coordinates and roles, full/material tetrahedron
sets, oriented boundaries, inputs, parameters and iteration work. Independent
CGAL global full-hull predicates and NumPy/AABB quality and boundary-link checks
pass for all four prototype outputs.

Both prototypes add no explicit workspace arrays, and the measured persistent
pool allocations match their controls. All three search kernels use 255
registers per thread. Local storage is 15,808 bytes/thread for the original,
17,280 for the hierarchy variant and 15,872 for the box-only scan. These properties
do not isolate the regression's cause: constructing/updating bounds, evaluating
them, traversal and changed instruction scheduling all contribute costs.

## Validation and evidence

Each variant passes the same 51 existing proof/reuse/support tests and 36
additional direct-query cases. The latter cover six captured input phases,
changed duplicate ownership, empty/shrinking/restored face prefixes, padded rows,
cube/octahedron ties, thin cells, random and fitted clouds, scales from 2^-60
through 2^60, and 32/256-thread launches. Final proposals match exhaustive scan;
small cases additionally check sampled winners with independent rational
predicates. Subtree and strided partitions need not produce the same intermediate
lane winners.

A first box-only compilation failed when the persistent storage quota was full.
Its log is retained. Only regenerable Warp test compiler caches were removed,
with a recorded inventory; sources, Git history and unique results remained on
persistent storage. Recompilation and the subsequent tests passed.

No adoption memory/capture audit or broader scanned-domain timing is claimed for
these rejected variants. Production code, tests, benchmark/example code,
standalone geometry APIs and the README are unchanged.

The [manifest](../results/validation/vtm/facet-pruning-probes/manifest.json)
records source hashes, commands, raw samples, private kernels, test logs,
independent comparisons, compiler properties and the storage-failure record.
[Hierarchy checks](../results/validation/vtm/facet-pruning-probes/facet-tree-pilot-independent.json)
and [box-only checks](../results/validation/vtm/facet-pruning-probes/facet-linear-pilot-independent.json)
contain all measured results. Reproduction scripts target the recorded baseline
and require the existing offline reference tools.

Tracked in [#164](https://github.com/alecjacobson/warpower/issues/164). These results
do not advance the [native CPU speed target](cpu-mesher-baseline.md); they prevent
a slower path from becoming the default.
