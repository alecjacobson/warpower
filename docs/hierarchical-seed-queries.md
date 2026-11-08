# Hierarchical checking of retained VTM cells

The subsequent [cached exact-query split](cached-exact-seed-queries.md) builds on
this traversal with separate strict/exact kernels and bounded fallback storage.
The measurements in this report describe the earlier implementation.

For VTM capacities 4,097–16,384, the retained-cell filter now traverses the
existing current segment hierarchy in groups of 32 sites. It can exclude a
whole subtree before inspecting any of its groups. The point predicates,
symbolic tie policy and final complete-mesh certificate are unchanged. The
independently callable power-diagram and Delaunay paths, and VTM capacities
through 4,096, keep their existing query implementations.

## Complete resident solves

Production `1ecaf47` is compared with main `5998b13` on NVIDIA L40 / Warp
1.19.0. All six pairs are newly interleaved, with three measured fresh complete
replays each. The scope includes domain refit, sampling, fitting, refinement,
quality stages and extraction. Compilation, setup, transfers and independent
validation are excluded. All initial and measured timing windows have zero CPU
quota throttling. Raw samples, setup and persistent pool measurements are retained.

| Sites and schedule | Before (ms) | After (ms) | Change |
| --- | ---: | ---: | ---: |
| Uniform torus 16,384, fixed 32 | 7,118.401 | 6,266.752 | -11.96% |
| Uniform torus 8,192, progressive | 3,668.416 | 3,540.830 | -3.48% |
| Torus target 4,096, reserved 16,384 | 7,554.318 | 7,394.130 | -2.12% |
| Graded torus 16,384, progressive | 8,846.710 | 8,134.063 | -8.06% |
| Uniform torus 4,096, progressive | 1,508.598 | 1,510.477 | +0.12% |
| Uniform torus 1,000, progressive | 513.240 | 513.579 | +0.07% |

All twelve outputs pass independent exact full-hull/orientation/empty-sphere
checks, material quality/topology, single-cycle boundary vertex links, and
boundary self-intersection checks. Every pair has bitwise-identical coordinates,
matching canonical full/material tetrahedra, oriented boundary faces, roles and
recorded work. Graded-gallery VTU exports round-trip exactly. See the
[complete comparison](../results/validation/vtm/hierarchical-seed-queries/large-stackless-final-v1-independent.json).

The earlier private six-pair pilot likewise preserved all coordinates,
connectivity and work. It improved the 16k fixed-site solve by 12.1% and the graded
16k solve by 8.2%. Its setup-only registry is absent from production: the existing
hierarchy is passed explicitly when the workspace is constructed. Traversal
kernels have identical ASTs to that measured prototype.

## Coverage and conservative bounds

Let L be the existing segment tree's power-of-two leaf base and G=max(1,L/32).
At depth log2(G), nodes G through 2G−1 cover consecutive blocks of 32 Morton
ranks. Lanes 0–31 inspect those ranks, masking padding beyond the point count.
The existing bottom-up min/max bounds enclose every current descendant point.
The caller rebuilds this segment and the shared permutation before querying.

A depth-first traversal descends to the left child or skips a completed subtree
by climbing right-child ancestors and selecting the next sibling. Thus it visits
each required group exactly once, in increasing rank order, without a per-lane
stack. At most 2G−1 nodes can be visited. An unfinished traversal at that budget
fails closed with an error and clears the cell's live flag.

Only the existing conservative sphere-box exclusion skips a node. An unresolved
or invalid sphere box forces traversal. Each surviving lane keeps exactly the
existing duplicate-owner, incident-site, point-box, bounded float32, prepared
Ball64 and exact/symbolic checks. A rejected cell continues checking its other
queries, preserving diagnostics. No proof survives a change of input.

The production path borrows arrays already owned and rebuilt by the site
hierarchy. It removes the separate group-box arrays and their preparation launch
on the selected routes; it adds no query stack, hierarchy build, host readback,
allocation or conditional graph node during replay. The existing arithmetic
filters and exact Delaunay tests remain in force.

## Native cost and persistent storage

A separate warmed full 16k fixed-site trace records the following kernel
activity. These diagnostic times are not clean complete-solve samples.

| Kernel family | Calls | Activity (ms) |
| --- | ---: | ---: |
| check_stackless | 227 | 2,187.224 |
| retained_cached | 194 | 928.538 |
| check_unproved_spheres | 554 | 817.653 |
| search_unproved_faces | 327 | 718.285 |
| record_face_support | 554 | 442.696 |

The query kernel still uses 255 registers per thread. This observation alone
does not identify a performance cause.

| Reserved sites | Before pool bytes | After pool bytes | Saved bytes |
| --- | ---: | ---: | ---: |
| 8,192 | 2,036,017,404 | 2,036,011,260 | 6,144 |
| 16,384 | 2,520,839,227 | 2,520,826,939 | 12,288 |

These are persistent workspace allocator-pool allocations, not process-wide
peaks. Small-route storage is unchanged.

## Independent API performance and validation

| Independent operation | Before (ms) | After (ms) | Change |
| --- | ---: | ---: | ---: |
| delaunay1000 | 9.171 | 9.165 | -0.06% |
| delaunay100000 | 183.367 | 183.853 | +0.27% |
| delaunay-grid | 2.205 | 2.194 | -0.49% |
| delaunay-fitted-hand | 13.435 | 13.436 | +0.00% |
| diagram100000 | 18.251 | 18.187 | -0.35% |

These newly interleaved measurements cover nine full Delaunay replays and thirty
power-diagram runs. Inputs, counts, status and diagnostics match, with clean quota
windows. All other 139 existing runtime/benchmark/example files are byte-identical
to the control; the two modified existing runtime modules change setup selection
only. Public defaults and README remain unchanged.

The complete suite passes **1,227 tests**, including 106 new primitive cases.
Recursive CPU coverage oracles test partial groups, root pruning, mixed pruning,
empty/full traversal and repeated captured launches. Independent containment
checks cover extreme finite/subnormal coordinates and current permutations.
Captured query comparisons cover ties, duplicates, thin and random clouds,
synthetic torus samples through 16,384 sites, scales 2^-60/1/2^60, empty prefixes,
changing positions and poisoned sphere boxes. Small cases also use exact rational
physical/symbolic oracles. Fifteen existing integration cases exercise route
selection, captured producers/consumers, invalid inputs, fallback, exhaustion and
recovery. The private pilot additionally checks saved real meshing outputs.

Twelve selected primitive/integration cases each pass padded and pool-disabled
lifetime memory checks in isolated processes: **24 jobs, zero reported errors**.
The earlier combined-process sanitizer issue remains tracked separately in
[#155](https://github.com/alecjacobson/warpower/issues/155).

Native audits cover a warmed full 16k solve and a single device-controlled
five-phase loop: 4,097 sites, a deformed 8,192-site case, invalid target 8,193,
a collapsed domain, and recovery. Invalid phases expose zero counts and zero
consumed volume. The three valid snapshots pass independent full/material/boundary
checks; first and recovered outputs agree. Buffers remain stable. Native nodes
contain kernels, memsets, device-to-device copies and conditional control, with
no host callbacks, host copies or explicit allocation/free calls. Each profile
has one final external synchronization. The first loop launch includes deferred
executable-graph instantiation/upload before its single graph launch; initial
setup is not claimed to require zero host work or driver storage. Native timings
are diagnostic, not additional clean benchmark samples.

## Remaining work

This improves the GPU implementation. The wide comparable-quality native CPU
speedup goal remains open; counts, quality and timing scopes must be matched
before such a claim. The [current CPU study](fixed-site-current.md) records those
limitations. Unchanged Armadillo extraction failures remain in
[#228](https://github.com/alecjacobson/warpower/issues/228); no extraction guard is
weakened. Larger active-prefix routing and scratch sharing remain separate
[#230](https://github.com/alecjacobson/warpower/issues/230) and
[#122](https://github.com/alecjacobson/warpower/issues/122) work.

The [provenance manifest](../results/validation/vtm/hierarchical-seed-queries/manifest.json)
links sources, command lines, complete samples, independent checks, native traces
and memory logs. Tracked in [#177](https://github.com/alecjacobson/warpower/issues/177).
