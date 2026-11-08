# Rational-centroid ray filter

This private candidate changes how signs are evaluated. It leaves the exact
geometric acceptance rule and its conservative failures unchanged.

For a tetrahedron with original float32 vertices `p0,...,p3` and triangle vertex
`q`, form the exact expression `A(q) = (q-p0)+(q-p1)+(q-p2)+(q-p3)`. This is four
times the displacement from the exact rational centroid. No centroid is rounded
to either float32 or float64. The three determinant numerators use these
displacements and the original direction. Scaling all displacements by four
changes parameter magnitudes but preserves every sign and barycentric decision.

The filter evaluates that expression with the existing `Ball` operations in
`warpower/_filtered_predicates.py`. Each original float32 value converts exactly
to binary64. Induction over its bounded additions, subtractions and products
encloses the exact `n,u,v,w` and `u+v+w`. Only when all five signs are certified
does the filter apply the original ray event rule. Any uncertain bound invokes
the original integer arithmetic for that triangle, including its ambiguous-ray
and arithmetic-failure behavior. A certified zero is accepted only when its
error is exactly zero; otherwise zero remains uncertain.

The calling module disables fast math and contraction. For finite binary32
inputs, nonzero degree-three value products have magnitude at least `2^-447`.
The smallest error-product terms needed by these operations are no smaller than
`2^-603`, well above the binary64 normal threshold. There is no division or
centroid averaging in the expression. At the other extreme, a centroid-relative
component is bounded by eight times the largest input magnitude; determinant
values and propagated errors remain below `2^400`. Thus these expressions do
not underflow or overflow binary64 under the stated input assumptions. The
positive pads in the existing interval operations cover rounding when forming
the error bounds themselves.

Plane-separation orientation calls use the existing certified orientation
filter, with the same exact fallback. A positive tetrahedron cannot have all four
vertices on one triangle plane. If every overlapping triangle plane leaves all
four vertices on one side, its exact centroid is strictly off each such plane.
Triangles with disjoint bounding boxes cannot contain the centroid or cross the
tetrahedron interior. An unambiguous centroid winding of zero therefore proves
the open tetrahedron interior belongs to the material exterior. Failed plane or
ray proofs remain inconclusive. Domain validation and finite positive input
tetrahedra remain prerequisites.

The companion tests check analytic geometry, tiny/large scales, ray ambiguity,
and exact rational bounds/events on hand and random finite-bit-pattern cases.
Those tests support implementation correctness; the interval argument above
defines why the fast signs are safe. Full-solve validation and performance gates
are separate.
