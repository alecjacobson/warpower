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
