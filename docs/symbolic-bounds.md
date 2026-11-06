# Conservative geometry in symbolic construction

The symbolic Delaunay fallback keeps its topology as triples of incident plane
IDs and uses the original indexed sites and symbolic lifting rule. Nearby BVH
nodes are visited first, using squared point-to-box distance solely to choose
visit order. The additional geometric filters described here accept only strict
signs; uncertain cases and exact ties retain the existing filtered/exact
predicates. Every resulting mesh still passes the full original-coordinate
Delaunay certificate.

## Cached vertex intervals

A vertex is the intersection of three site or clipping-box planes. The existing
halfspace code supplies float64 balls enclosing the exact, unweighted plane
coefficients. Cramer's rule and the [outward quotient bound](sphere-bounds.md)
enclose the Cartesian intersection. Negative determinant orientations are
normalized by exact negation. If the determinant interval contains zero, a
quotient is nonfinite, or a coordinate cannot be bounded, that coordinate keeps
the complete initial clipping-box interval. A valid clipped vertex's limit lies
inside that box, so interval intersection with the box is safe.

Each coordinate interval `[lo, hi]` is cached as a midpoint and an error radius.
The radius includes the half-width, `16*eps64*(abs(lo)+abs(hi))`, positive-arithmetic
padding, and `16*min_normal64`. These allowances enclose midpoint/radius rounding
and underflow; they are arithmetic error bounds, not geometric tolerances.
Initial box corners are exactly representable promoted float32 values, with zero
error. Subsequent vertices compute their intervals only when they are created.

The cache uses the existing `Scratch64.positions` and `cartesian` arrays as
values and errors during symbolic construction. Every vertex swap moves both
arrays. Plane-table compaction changes IDs without moving the corresponding
geometric vertex. Each constructor initializes its own active scratch before
use, so sharing these arrays with the ordinary floating-point constructor does
not retain stale geometry or add allocations.

For each candidate plane, its unweighted coefficients are computed once. A
cached vertex interval evaluates the linear halfspace expression with Ball
arithmetic. A finite, strictly nonzero result determines inside/outside without
recomputing an in-sphere determinant. An absolute normal-number allowance covers
underflow in these interval products and sums. Otherwise the original site or
site/box symbolic predicate resolves the classification. In particular, a
numerically small value never becomes a tie, and a zero sign never bypasses
symbolic tie-breaking.

## Why unweighted limits suffice

The symbolic weights perturb halfspace constants, not normals. A valid vertex
with nonzero normal determinant is therefore a continuous function of those
infinitesimals and tends to the unweighted Cramer intersection. A strictly
positive or negative halfspace value at that limit has the same sign for all
sufficiently small positive infinitesimals. Only a zero limiting sign requires
higher symbolic coefficients. There are finitely many vertices and predicates
in a bounded construction attempt, so all strict decisions admit a common
sufficiently small perturbation. No finite perturbation is applied to output
positions.

## Directional cell and node rejection

The union of the cached vertex intervals encloses the limiting convex cell.
The cell bounds are updated after each plane that actually removes vertices;
an unchanged vertex count alone would not detect a changed polyhedron.
Let `x` range over the cell box relative to its owner, and let `delta` range over
a BVH node's site positions relative to the same owner. A site cannot clip the
cell if the following strict inequality holds throughout those intervals:

```
2*x dot delta - |delta|^2 < 0.
```

For one coordinate, with `x in [l,h]` and `delta in [a,b]`, the maximum is the
larger of the two values obtained by taking `x=l` and `x=h` and setting
`delta=clamp(x,a,b)`. This follows because the expression is linear in `x`, and
for fixed `x` it is a concave quadratic in `delta`. Summing three coordinate
maxima bounds the complete expression. The implementation encloses input
subtractions, endpoint arithmetic, quadratic evaluation, and the final sum;
only a finite, strictly negative upper bound rejects a node or individual site.
Large offsets, poorly conditioned vertices and nonfinite arithmetic can weaken
or disable rejection, but cannot silently choose a predicate tie.

All relevant kernels disable fast math and fused multiply-add. Interval code
explicitly includes absolute underflow allowances rather than assuming that
rational Cartesian centers obey the normal-product assumptions of the original
float32 polynomial filters.

## Validation

`tests/test_symbolic_bounds.py` uses exact rational arithmetic to check mixed
site/box Cramer intersections across scales and translations, nearly flat
configurations, the directional box maximum, touching/one-ULP cases, and cached
plane signs. Exact repeated-site and cospherical ties must defer to the original
predicate. Independent symbolic-construction tests also cover complete geometry,
changing-input capture, capacity failures, clipping-box growth, invalid input
and recovery. Performance and full-suite validation of this developing
optimization must accompany its eventual integration; local interval tests alone
are not sufficient.
