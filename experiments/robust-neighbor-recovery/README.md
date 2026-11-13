# Original-coordinate predicate recovery

The integrated change corrects all seven known wrong neighbor pairs on the original
Bonsai, Garden, Bicycle and Playroom inputs. Independent exact checks also establish
that the four changed pairs in the million-site synthetic comparisons are corrections.
The L40 validation passes the existing per-case 2% regression gate across all 74
historical inputs, the live Bonsai case, and the ten-case geometry gate.

The 74-case geometric-mean speedup over `655412e` is
**1.097×**. This is a comparison with our
previous builder. Small positive changes are retained in the table and raw data;
passing the confidence-interval gate does not mean every case has zero overhead.

The [retry-escalation follow-up](../retry-escalation/README.md) completes the
precision/storage separation in ordinary and captured execution. In the version
measured below, captured FP32 retries lacked the numerical guard and cooperative
FP64 capacity failures could lose their precision history. The follow-up preserves
these historical measurements and publishes a separate comparison against them.

## Implementation

The ordinary FP32 clipper records a suspicious side test and a separate kernel checks
it against the original site coordinates and weights. Interior tests use filtered
FP64 and exact integer predicates; unresolved box-plane tests request recovery.
A bounded queue uses 32 cooperating threads per difficult cell to rebuild with FP64.
Larger or unsuccessful jobs stay in the fallback pipeline. Capacity failures grow
FP32 storage separately from precision failures. Captured `DiagramWorkspace`
execution uses the same checks.

Packed FP32 Cartesian positions, a 96-entry initial traversal queue, live facet hints,
and cooperative clipping/node rejection offset the cost of the new checks.
The cancellation screen is a heuristic, and cell rebuilding still uses floating-point
geometry. **This is not a global exactness certificate for every output edge.**
See [architecture](../../docs/architecture.md) and [status/diagnostic semantics](../../docs/api.md).

## Performance

L40, resident identical inputs, three warmups, 20 alternating paired synchronized
wall-clock repetitions including construction allocations; live Bonsai uses 30.
Warp allocator retention is 32 GiB. All accepted timing windows are free of CPU
throttling. Input hashes match the historical matrix. Success statuses are checked
before and after timing; that check alone is not a geometric certificate.

| Input | Previous (ms) | Updated (ms) | Paired change | 95% interval |
|---|---:|---:|---:|---:|
| uniform-100000-power | 13.939 | 14.145 | +1.48% | [+0.87%, +2.09%] |
| cluster5-100000-voronoi | 29.523 | 20.809 | -29.50% | [-30.45%, -28.54%] |
| cluster10-100000-voronoi | 39.442 | 22.755 | -42.30% | [-42.85%, -41.75%] |
| live-bonsai-9216-voronoi | 142.262 | 83.761 | -41.12% | [-42.83%, -39.41%] |
| bonsai-1968522-voronoi | 240.724 | 225.052 | -6.51% | [-6.82%, -6.19%] |
| uniform-15000000-voronoi | 1805.235 | 1731.522 | -4.08% | [-4.26%, -3.90%] |

The gate rejects a paired 95% Student-t interval wholly above +2%; an aggregate win
cannot override a failing case. All 74 rows and the captured adjacency/tetrahedra
measurements are in [validation.json](validation.json). Tetrahedra match the baseline
exactly in the geometry gate. Its eight directed adjacency changes are independently
audited below.

Both complete VTM solves preserve bitwise point positions, vertex roles and
tetrahedron sets, with the existing hull-proof cache enabled in both versions.
These use three warmups and ten alternating pairs, excluding setup and downloads.

| Complete 4K VTM solve | Previous median (ms) | Updated median (ms) | Paired change | 95% interval |
|---|---:|---:|---:|---:|
| torus | 921.225 | 916.966 | -0.50% | [-0.61%, -0.38%] |
| graded-torus | 824.702 | 821.103 | -0.45% | [-0.55%, -0.36%] |

## Correctness and tests

The scene audit checks every directed difference from validated CGAL references and
every change from our fresh baseline using exact rational positive-area facet LPs.
Shared unchanged pairs remain outside the audit. Box clipping and zero-area dual
faces explain the other CGAL differences.

| Input | Directed pairs audited | Previous errors | Updated errors |
|---|---:|---:|---:|
| live-bonsai (9,216 sites) | 172 | 0 | 0 |
| bonsai (1,968,522 sites) | 368 | 4 | 0 |
| garden (4,064,598 sites) | 1,182 | 2 | 0 |
| bicycle (4,199,873 sites) | 1,566 | 2 | 0 |
| playroom (3,181,453 sites) | 474 | 6 | 0 |

The synthetic audit checks all eight changed directions (four pairs): six directions
on one million unweighted sites and two on one million weighted sites. Positive
facets have rational witnesses checked against every original halfspace; negative
facets have exact LP certificates on subsets of those constraints. All eight changes
are correct. The auditor passes 140 directed real-fixture controls and three additional
weighted/constraint-generation controls.

**168 tests pass:** 126 existing tests and 42 new regressions. The seven reduced real
fixtures are exercised through ordinary calls, one-cell chunks, forced capacity
retries, CSR hints, and captured execution. Forced cooperative tests cover weighted
cells, remapped rows, queue overflow, and duplicate ownership. CUDA memcheck and
initcheck pass all 42 new tests; memcheck, initcheck, racecheck and synccheck pass the
five focused cooperative cases with zero reported errors or hazards.

The initial sanitizer failure is retained in the evidence. One part was a test
copying unused output capacity, now restricted to the valid prefix. A separate
CUDA 12.6 conditional-graph memcpy warning reproduces on the baseline and in a
minimal ordered producer/consumer example. An equivalent Warp copy kernel removes
that warning while keeping device initialization checking enabled.

## Evidence and reproduction

The [final evidence archive](https://github.com/alecjacobson/warpower/releases/download/evidence-geometry-updates-4ceb748/robust-neighbor-recovery-final-20261112.tar.gz) includes the source patch, all per-repetition
samples, input/source hashes, status checks, exact queries and witnesses, test logs,
sanitizer logs and reproduction runners. SHA-256:
`261be3aaca9f513f41a754cfa566890ce2c88d8d26f2faafcdee8ca75f8be58d`. A fresh download verified all members and applying the patch
to a fresh `655412e` source copy reconstructed every recorded library source hash.
See [final-manifest.json](final-manifest.json) and
[final-remote-verification.json](final-remote-verification.json).

The eager 74-case matrix and exact scene audits use frozen v2 sources. Final v3
changes only the captured count copy in `_graph.py`; all eager implementation
modules are byte-identical. Captured timing, VTM integration, the 168-test suite
and broad sanitizer checks use v3. Both variants and their linkage are preserved.

Run `python experiments/robust-neighbor-recovery/exact_reference.py` to independently
recompute the 35 small-fixture cells with rational arithmetic. The runtime remains
pure Warp; native CGAL helpers are offline validators only.

The [earlier screen](screen-20261112.md) records the rejected ~3% small-case regression
and its immutable evidence. The current results supersede that candidate, while
preserving the historical record. Follow [PR #299](https://github.com/alecjacobson/warpower/pull/299)
and [issue #300](https://github.com/alecjacobson/warpower/issues/300).
