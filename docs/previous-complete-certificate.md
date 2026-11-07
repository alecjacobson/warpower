# Fresh certification of a previous complete mesh

The two private [#182](https://github.com/alecjacobson/warpower/issues/182)
prototypes remain unadopted. Trying a complete certificate before fragment
filtering slows full solves; rejecting invalid candidates earlier removes most
of that cost but gives no useful improvement. Production retains its existing
fragment path. Standalone diagram/Delaunay code and the README result are unchanged.

## Candidate and proof

Each attempt refreshes the current original-site hierarchy and canonical owners,
then copies the previous complete tetrahedron set into the existing candidate
buffer. v1 invokes the ordinary `DelaunayCertificate` on the current coordinates.
It requires positive orientation, valid incidence, canonical-site coverage,
supporting hull faces and the existing local/global symbolic sphere proof.
This is a new proof of the current mesh, not reuse of an old geometric proof.

Success publishes the existing certificate's output and face metadata. The
compacted output is also copied back into its candidate prefix, so discarded flat
rows cannot leave inconsistent face indices. Fragment counters are reset on this
branch. Failure continues through the unchanged fragment-completion path and,
if necessary, the ordinary constructor. All replay decisions and copies are
device operations. The private adapter adds three diagnostic integers per reuse
workspace and recaptures the graph at setup; setup timings are not adoption claims.

v2 moves canonical coverage ahead of the expensive checks and stops after any
conclusive failure. After valid topology, it tests paired-face sphere queries
before hull support. These queries are necessary for any global certificate too;
passing them alone does not authorize publication. A successful candidate still
needs hull support and a strict anchor, or the original global sphere check when
no anchor exists. Small candidates keep the global check. Ordinary construction
and fragment completion retain their original schedules and predicates.

## Complete solves

Fresh controls and private candidates use main cc0ad8f (runtime 4489ff4), Warp
1.19.0 and NVIDIA L40. Every initial and measured quota window is unthrottled.
Times cover complete resident-device meshing, including sampling, growth,
fitting, quality stages, extraction and diagnostics. The 1k case uses five timed
replays and the 4k case three; setup, compilation and offline checks are separate.

| Variant | Sites | Control, ms | Candidate, ms | Change |
|---|---:|---:|---:|---:|
| v1: ordinary complete certificate | 1,000 | 667.510 | 711.953 | 6.66% slower |
| v1: ordinary complete certificate | 4,096 | 2,030.261 | 2,122.643 | 4.55% slower |
| v2: staged rejection | 1,000 | 667.707 | 674.409 | 1.00% slower |
| v2: staged rejection | 4,096 | 2,030.214 | 2,027.990 | 0.11% faster |

The last difference does not establish a useful speedup. No additional timing
repetitions or broader adoption benchmarks are justified by these pilots.
All four candidate meshes have bitwise-identical positions and matching full and
material cell sets, oriented boundaries, roles, input data, parameters and work
counters. They pass independent CGAL exact global full-hull checks and NumPy/AABB
material checks, including single-cycle boundary-vertex links.

## Observation and correctness

An initial observer always ran the original fragment path after the speculative
certificate, regardless of its answer. Its three full graph executions give:

| Reserved solve size | Reuse workspace | Attempts | Accepted |
|---|---:|---:|---:|
| 1,000 | 1,000 | 420 | 129 (30.71%) |
| 4,096 | 1,024 | 78 | 9 (11.54%) |
| 4,096 | 2,048 | 39 | 6 (15.38%) |
| 4,096 | 4,096 | 507 | 90 (17.75%) |

Across the 4k solve this is 105 of 624 attempts (16.83%). Most rejected candidates
contain an inverted old cell; other failures involve sphere violations, hull
support and missing canonical sites. The observer's positions, cell sets, roles
and work counters match the previously validated production outputs. Its timings
are diagnostic overhead, not performance evidence.

Each variant passes 69 existing reconstruction tests. Two private copies of large
route tests also record the new whole-certificate branch, replacing only their
old expectation that fragment proofs must be consumed on an unchanged topology.
v1 passes 15 additional cases and v2 passes 23. Captured producer/consumer loops
cover uniform scaling, one-ULP motion, unrelated input, duplicate merging,
canonical-owner promotion, shrinking prefixes, NaN input and recovery. Other
tests corrupt counts and cell rows, remove cells, introduce inverted/non-Delaunay
cells, and insert harmless flat rows before a valid mesh. They check publication
metadata and replay recovery at scales from 2^-60 to 2^60. Host allocation,
readback and synchronization are forbidden during the captured loop test.

The eight extra v2 cases compare acceptance with both the original certificate
and its global-only route. They include random meshes, malformed buffers,
missing sites, bipyramid diagonals, symbolic octahedron ties, coplanar cube hulls
and overlapping covers whose boundary diagonals differ. The latter still require
the global fallback; local adjacency alone is insufficient.

The [artifact manifest](../results/validation/vtm/previous-complete-certificate/manifest.json)
retains both source variants, the observer, tests, commands, raw samples, meshes
and independent checks. It also retains an initial offline-check launch that
omitted `PYTHONPATH` and failed before validation; the corrected launch passed.
For reproduction, materialize the archived Python modules under their original
names in `.work/vtm` at the recorded baseline and restore the referenced shared
validator. Workspace-specific paths identify the measured run.

There is no new full-suite, broad-gallery, memory-sanitizer, native-capture or
standalone timing claim for these unadopted prototypes. Selective dispatch would
need new evidence that it predicts a useful attempt without costing the savings.
The wide CPU outperformance goal remains unmet.
