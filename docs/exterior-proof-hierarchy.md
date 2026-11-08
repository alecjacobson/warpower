# Hierarchical exact exterior proofs

Whole-tetrahedron exterior recovery now uses the current surface hierarchy for
plane-separation candidates and rational-centroid rays. Exact leaf predicates,
transactional reference/volume/topology guards and the failure reporting contract remain in
place. This addresses [#239](https://github.com/alecjacobson/warpower/issues/239).
The tangential recovery used to expose the large uniform Armadillo workload
remains a separate private experiment under
[#228](https://github.com/alecjacobson/warpower/issues/228).

## Conservative bounds

Surface nodes contain exact binary32 min/max coordinates after domain refit.
Plane separation skips a node only when its box is disjoint from the tetrahedron
box; overlapping leaf triangles still use the original filtered/exact orientation
predicates. This changes traversal, not the plane-separation condition.

For the centroid ray, error balls bound `4*box - sum(tet vertices)` without
rounding the rational centroid. Direction signs orient each entry/exit slab to
positive denominators. A node is excluded only when an axis's upper exit bound is
provably behind the origin, or an entry exceeds an exit after cross multiplication. No
division is required. The fixed directions have finite nonzero integer components;
finite binary32 coordinates keep the bound expressions and their errors in normal
binary64 range. Uncertain comparisons visit the node. Every visited triangle uses
the original filtered/exact dyadic determinant predicate. Unresolved crossings
remain inconclusive. The caller requires a valid domain and current refitted bounds.

The exhaustive reference can report ambiguity for an infinite triangle plane even
when its finite triangle cannot meet the ray. Certified box exclusion may skip
that disjoint coplanarity. Thus arbitrary degenerate ray diagnostics are not
promised to match the reference bit for bit. An explicit regression covers this
case. No approximate centroid or unproved exterior deletion is accepted. The
original exhaustive implementation remains available internally as a test oracle.

No persistent arrays or graph allocations are added. Standalone diagram and
Delaunay implementation files are unchanged, as are 255 existing tracked files,
including README. New code is confined to the private proof module and its caller;
the new tests cover the proof rather than changing default mesher policy.

## Complete replay measurements

NVIDIA L40, Warp 1.19.0. Baseline `6f878c7`, runtime `431958f`. Each table entry is the
median of three fresh complete replays after a first replay, including refit,
initialization, growth/fitting, quality rounds and extraction. Setup/export and
independent validation are excluded; all first/timed CPU quota records are clean.
The completed clean 16K pair is retained from the first batch. An initial-replay
CPU quota event on the first 1K current run stopped that batch; both 1K runs were
repeated, and the original records are retained. Its timed quota was clean, but
the stricter first-and-timed gate rejected it. The two `-recovery` rows use the
identical private tangential recovery on both
revisions and are explicitly experimental. They are not a claim that ordinary
main already recovers those inputs. All other rows use the ordinary production API.
The scanned Armadillo is an additional benchmark, not a model shown in the
Alliez paper. Input attribution: Stanford University Computer Graphics Laboratory,
via the pinned libigl tutorial-data distribution. The
[asset provenance and terms](../results/validation/vtm/exterior-proof-hierarchy/scanned-armadillo-input/provenance.json)
remain separate from the code license.

| Case | Baseline ms | Hierarchy ms | Ratio |
| --- | ---: | ---: | ---: |
| torus16384-fixed32 | 4887.275 | 4891.797 | 0.999× |
| torus1000 | 512.324 | 513.959 | 0.997× |
| thin-hand-seed7 | 1293.570 | 1248.843 | 1.036× |
| cavity | 357.175 | 358.312 | 0.997× |
| disconnected | 487.054 | 487.947 | 0.998× |
| plate | 920.186 | 921.520 | 0.999× |
| armadillo-graded4096 | 2582.186 | 2584.227 | 0.999× |
| armadillo-graded8192-recovery | 4834.963 | 4317.795 | 1.120× |
| armadillo-uniform8192-recovery | 34433.035 | 3956.162 | 8.704× |

All 18 saved before/after meshes are independently checked for complete full hull,
positive orientation/global empty spheres, material quality/topology, single-cycle
boundary vertex links and exact boundary self-intersections. VTU exports round-trip.
Six case pairs have bitwise-identical coordinates. Thin-hand differs in two scalar
coordinates by at most 4.405e-18; cavity and disconnected each differ in one by
2.328e-10 and 4.657e-10 respectively. Canonical full/material cells, oriented
boundaries, roles and scheduler work match for every pair. The minimum radius
ratio is unchanged; the largest per-cell ratio difference is 1.103e-9. No coordinate
bitwise-equivalence claim is made for those three pairs. The first checker completed
all 18 independent validations before failing its unnecessary bitwise-only assertion;
the final analysis retains that failure, reuses the completed geometric checks and
records every changed value. Geometry/work comparisons and all inputs are retained in the
[independent report](../results/validation/vtm/exterior-proof-hierarchy/exterior-bvh-final-v3-independent.json).

## Independent APIs

| Standalone case | Baseline ms | Hierarchy ms | Change |
| --- | ---: | ---: | ---: |
| delaunay1000 | 9.2003 | 9.1908 | -0.10% |
| delaunay100000 | 181.9136 | 182.9933 | +0.59% |
| delaunay-grid | 2.2208 | 2.1938 | -1.21% |
| delaunay-fitted-hand | 13.4104 | 13.4008 | -0.07% |
| diagram100000 | 18.1014 | 18.1236 | +0.12% |

These implementations and inputs/diagnostics are unchanged. Small timing differences
are measured variability, not an intended change to their execution paths.

## Capture and memory qualification

All 1,317 tests pass, including 25 new proof tests. An exact Fraction oracle checks
1,344 boxes across scales 2^-140 through 2^125, including intervals around rounded
centroids. Additional comparisons cover analytic cavities/gaps/crossing cells,
nonfinite/nonpositive tets, odd leaf counts and nondefault-stream refits.
Eight isolated sanitizer jobs pass with zero errors: padded allocations and
pool-disabled lifetime checks for new bounds, a recorded real Armadillo proof,
extraction failure/empty/disable/recovery, and the complete mesher loop. Each job
has a 600s deadline and 131,072-byte diagnostic stack; periodic 60s Python stack
prints are diagnostics, not failures.

Native traces cover a warmed experimental uniform 8K solve and a production
thin-hand loop with changed seed, invalid target and exact recovery on a nondefault
stream. Each has one external graph launch and a final synchronization. All
captured work consists of device kernels/conditionals/memsets/DtoD copies, with no
host transfer, callback or explicit allocation/free in the profiled work.
The loop's three valid frames independently validate, invalid public counts and
consumed volume are zero, output pointers remain stable, and restored geometry
matches the first frame exactly. First-launch graph instantiation/upload, when
reported, is setup outside device work and is retained in the raw audit.

In instrumented uniform 8K traces, topology recovery occupies
453.453 → 456.425 ms, while later quality rounds occupy
31,833.372 → 1,844.496 ms.
These spans explain the bottleneck; they are not clean benchmark samples.

The [manifest](../results/validation/vtm/exterior-proof-hierarchy/manifest.json)
pins exact source, earlier exploratory pilots, final commands and meshes, raw native
traces, validators and memory logs. Restore recorded `.work/vtm` paths/shared
references and pinned revisions to reproduce; adapt the absolute workspace root.
Drivers refuse overwrites. Use the recorded environment and run one GPU process
at a time; keep CPU geometry validation out of clean timing runs. Wide
comparable-quality CPU outperformance and the remaining VTM recovery work remain
open. README retains its earned Paragram flagship result.
