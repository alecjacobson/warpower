# Proven exterior recovery

The thickened-domain extraction rule can retain wholly exterior cells between
nearby features. Such cells can leave a four-face boundary edge or a vertex with
two disconnected link cycles. A cell incident to a crease cannot be removed by
the older feature-protecting exterior-band policy. This fallback requires a
geometric proof before considering removal.

`ExtractionWorkspace(..., proven_exterior_recovery=True)` enables a single
transaction after the other enabled recovery policies. The independent default
is `False`, with no recovery storage or kernels. `proven_exterior_enabled` may
be a contiguous device `int32[1]` controlling that transaction; bind it at setup.
The extractor never moves points or changes the supplied full-hull connectivity.

The full mesher enables the fallback by default when `exterior_recovery` is
enabled. It first completes ordinary fitting and recovery. Only a remaining
failure enables one extra global fitting iteration, followed by proof recovery.
`proven_exterior_recovery=False` disables both that iteration and proof recovery.
An explicit `True` also works independently of the older exterior-band option.

Current traversal uses [conservative surface-hierarchy bounds](exterior-proof-hierarchy.md)
for both plane candidates and exact rational-centroid rays. The report distinguishes
safe removal of disjoint coplanar-ray ambiguity from exact leaf predicate decisions
and records the newer performance/capture/memory qualification. The measurements
below remain the original recovery-policy study.

## Whole-cell proof

The input must be a valid, closed, consistently oriented triangle domain and the
input tetrahedra a nonoverlapping positive complex. All proof predicates use the
original binary32 coordinates:

1. Require a finite positively oriented tetrahedron.
2. For every triangle whose axis-aligned box overlaps the tetrahedron's box,
   require its plane to leave all four vertices on one side, allowing zeros.
   A straddled plane makes the proof inconclusive, even when its triangle might
   miss the tetrahedron.
3. Classify the **exact rational centroid**, without rounding its coordinates.
   An unambiguous winding count of zero proves that the tetrahedron's open
   interior is exterior. Three fixed ray directions handle boundary hits;
   unresolved ambiguity or arithmetic failure remains inconclusive.

Plane separation prevents any surface triangle from crossing the connected open
tetrahedron interior. A positive tetrahedron cannot have all four vertices in a
triangle's plane, so its centroid is a strict witness for every overlapping
plane. Triangles with disjoint boxes cannot cross the cell or contain its
centroid. The winding query therefore certifies the entire open interior, not
just an exterior circumcenter. Contact along the cell boundary is allowed.

For a triangle vertex `q`, ray determinants use
`A(q) = (q-p0)+(q-p1)+(q-p2)+(q-p3)`, exactly four times the displacement from the
centroid. Scaling those displacements preserves signs and barycentric decisions.
Existing roundoff-bounded `Ball` operations enclose five determinant signs;
only certified signs bypass the dyadic-integer fallback. A zero is certified
only with zero error. Fast math and contraction are disabled.

Finite binary32 inputs convert exactly to binary64. Relevant nonzero degree-three
value products are at least `2^-447`; the smallest required error products are
at least `2^-603`, above binary64 underflow. Components are bounded by eight times
the largest input magnitude, with determinant values and error bounds below
`2^400`. The interval operations' positive padding covers rounding of their
error bounds. Uncertain signs use the existing wide integer arithmetic on the
same expression, retaining its failure and ambiguous-ray handling.

## Transaction and fitting safeguards

The transaction runs only after fatal input checks pass and the topology
diagnostic is an edge contact (`2`) or a vertex contact (`4`). It snapshots the
selection and volume. Only selected cells with positive circumcenter distance
and a whole-cell proof are candidates. All changes are rejected together if they
lose a referenced vertex, leave a topology failure, or change volume by more
than one percent. Rejection restores and recompacts the complete original
selection. A proved cell may be incident to crease or corner sites, but those
sites must remain referenced. Exact input feature-edge conformity is not claimed.

The final fitting step can resolve a boundary coordinate stalled exactly halfway
between adjacent representable values. It chooses the frozen boundary target
only when the requested coordinate is an exact midpoint that rounded back to the
old value, and the resulting vector obeys the existing displacement cap. It
does not snap arbitrary small movements. Interior and corner targets remain
unchanged. The normal full-hull certification and energy/backtracking transaction
decides whether to accept the step. Ordinary growth, fitting, and selective
proposals retain their kernels. Optional local quality retries remain available
in their normal stages.

Once recovery succeeds, selective, jitter, and interior-stage guards evaluate
the same recovered material subcomplex. A fresh solve resets the enable flag and
midpoint counters. Everything after setup uses preallocated device buffers and
conditional graph nodes. Standalone diagram and Delaunay APIs do not construct
this layer.

## Diagnostics and examples

The independent extractor exposes device scalars `proven_removed_count` and
`proven_exterior_rejected` when enabled. They describe its most recent extraction,
not a cumulative count across a meshing solve. The gallery JSON additionally
records whether the final fallback was enabled and the proposed/accepted midpoint
axis and site counts. Child scratch can describe a rejected quality trial;
the parent `valid`, `error`, and output counts govern consumption.

```bash
python -m examples.vtm_gallery --case thin-hand --n 1000 --seed 7 \
  --repeats 3 --output results/local/proven-hand
python -m examples.vtm_gallery --case thin-hand --n 1000 --seed 7 \
  --no-proven-exterior-recovery --output results/local/original-hand-policy
```

This is a bounded recovery, not a guarantee of a conforming mesh for arbitrary
sampling. Coarse hand quadrature remains a known failure. The overall native CPU
speedup and feature-conformity goals remain open.

## Measured recovery and compatibility

On the NVIDIA L40 with Warp 1.19.0, runtime `fe3e469` is compared with baseline
`a34910d`. The final grid contains 27 complete gallery runs and ten standalone
runs. Each gallery timing is the median of three additional **fresh complete
solves**, including domain refit, sampling, growth, fitting, quality stages and
extraction. Setup, upload, file I/O and independent CPU validation are separate.
All first and timed replay windows are free of CPU quota throttling.

| Hand configuration | Baseline | New valid solve (seconds) | Minimum radius ratio | Two-sided boundary RMS |
|---|---|---:|---:|---:|
| Seed 2, default quality retries | Invalid | 2.8964 | 0.14157 | 0.00023042 |
| Seed 7, default quality retries | Invalid | 1.8930 | 0.11098 | 0.00023094 |
| Seed 7, four local quality retries | Invalid | 1.7491 | 0.12887 | 0.00023094 |
| Coarse 4/4 quadrature, seed 2026 | Invalid | Invalid | — | — |

The repaired cases use 1,000 sites, 256 initial sites, 20/20 quadrature and four
quality rounds. Failed runs publish error 64 and zero counts and are excluded
from mesh-speed comparisons. The explicit opt-out preserves seed 7's original
failure. Local quality retries remain opt-in; one result does not justify a
universal default change. These are successful GPU solve times, not speedups
over invalid results or native CPU meshes.

| Previously successful full solve | Baseline ms | New ms | Change |
|---|---:|---:|---:|
| Hand seed 0 | 1207.492 | 1209.975 | +0.21% |
| Hand seed 1 | 1051.392 | 1044.948 | -0.61% |
| Hand seed 42 | 876.176 | 877.798 | +0.19% |
| Hand seed 2026 | 1332.783 | 1341.703 | +0.67% |
| Counterbored plate | 1073.610 | 1064.964 | -0.81% |
| Cavity, 4/4 quadrature | 1937.713 | 1940.228 | +0.13% |
| Disconnected, 4/4 quadrature | 1926.105 | 1928.679 | +0.13% |
| Torus, 4,096 sites | 1784.795 | 1790.595 | +0.32% |
| Graded Armadillo, 4,096 sites | 3509.307 | 3499.993 | -0.27% |

All nine retain counts, roles and work. Eight retain canonical full/material
connectivity and oriented boundary triangles. Hand seed 1 has a documented
[baseline reproducibility variation](https://github.com/alecjacobson/warpower/issues/208):
tiny near-zero coordinates change eight full-hull cells, six material cells and
two boundary triangles. The same alternate triangulations occur between two
identical-parameter baseline runs; the new result matches the earlier baseline
connectivity. Common-cell qualities, reported quality percentiles and sampled
fit agree. The six changed cells have different qualities, so the complete
sorted quality arrays are not identical. The strict initial checker failure and
separate audit are retained, rather than describing all nine pairs as identical.

| Standalone routine | Baseline ms | New ms | Change |
|---|---:|---:|---:|
| Delaunay, 1,000 sites | 9.1669 | 9.1801 | +0.14% |
| Delaunay, 100,000 sites | 183.1606 | 184.0866 | +0.51% |
| Delaunay, 4³ grid | 2.2003 | 2.1992 | -0.05% |
| Delaunay, fitted hand | 13.4039 | 13.4052 | +0.01% |
| Diagram, 100,000 unweighted sites | 18.8220 | 18.8744 | +0.28% |

The standalone Delaunay cases use nine repeats; the diagram case uses 30 after
five warmups. Recorded inputs, counts, status and diagnostics agree in all five
pairs. The largest observed slowdown is 0.51%, with no material regression in
this grid. The ordinary diagram and Delaunay runtime paths are unchanged.

All 21 valid exported meshes pass independent full-hull/empty-sphere and positive
orientation checks, material incidence and vertex-link checks, physical quality,
two-sided sampled distance, exact VTU round trips and boundary self-intersection
checks. Hand and plate outputs also include sampled crease/corner measurements.
These tests do not establish exact constrained-edge conformity. The complete
unit suite passes 1,084 tests. Eight isolated Compute Sanitizer jobs pass with
zero errors: 33 primitive/oracle tests, eight extraction transaction tests, one
complete recovered solve with local retries, and one four-phase captured loop,
each in padded and pool-disabled lifetime modes. Instrumentation uses the
existing 131,072-byte diagnostic stack allowance; ordinary benchmarks do not. Sources, failed attempts, snapshots, commands and
checks are linked by the
[evidence manifest](../results/validation/vtm/proven-exterior-recovery/manifest.json).
Native first-launch Nsight traces cover a complete recovered solve and the
four-phase producer/mesher/consumer loop, with four local quality retries enabled.
Both contain only device-to-device graph copies, no host callback or
allocation/free nodes, and no allocation/free/copy API calls during the profiled
range. First-launch graph instantiation/upload and one external final
synchronization are recorded. The loop reports errors `[0,1,0,0]`, validity
`[1,0,1,1]`, and recovery enable flags `[1,0,0,1]`; invalid input produces zero
counts and the final frame recovers. Nsight can omit individual graph-copy
activities, so the DOT copy directions are independently checked.

The README's existing Paragram flagship is unchanged.
