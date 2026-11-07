# Private analytic-filter experiment (#166)

This probe changes only the original strided search of the Reuse-owned facet
completion. No production module or independent API has changed. It returns a
strict sign only when abs(det) exceeds a finite analytic bound. All other cases
use the original Ball64/integer orientation or complete symbolic sphere policy.
The physical and weighted tie policies remain intact; uncertainty is never zero.

The determinant and permanent expressions are adapted from Jonathan Richard
Shewchuk's public-domain May 18, 1996 predicates.c, linked from
https://www.cs.cmu.edu/~quake/robust.html. The original reference is retained with
SHA256 f8662c3f407d1c1c5dcd4dd49ea8b8ddd801a71827d65734206ddc3741792029.
Its first-stage coefficients are (7+56u)u and (16+224u)u, u=2^-53. We use exactly
representable powers of two 2^-49 and 2^-47, respectively. These exceed even the
same expressions evaluated with the larger conservative u=2^-52. Multiplication
by these powers is exact in the established normal range. Keep the reference's
operation grouping, disable fast math and fused arithmetic, and promote original
float32 operands to float64 before subtraction.

Sign conventions: for orientation use reference arguments (b,c,d,a), giving
det(b-a,c-a,d-a). For sphere negate the reference grouped determinant. These
permutations/negation do not change the absolute error bounds.

Every finite float32 coordinate has magnitude below 2^128 and belongs to the
lattice with spacing 2^-149. Float64 coordinate differences are below 2^129
(or conservatively at most that bound after rounding). A rounded difference
retains this lattice: when representable spacing is finer, the exact lattice
value is representable; when coarser, rounding picks a coarser lattice point.
Products of degree k retain a lattice no finer than 2^(-149k); rounded sums of
terms of the same degree preserve it. Thus any nonzero degree-five intermediate
is at least 2^-745. The smallest nonzero sphere error bound is at least 2^-792,
and the orientation equivalent is at least 2^-496. All are normal float64.

A product of two differences is at most 2^258. Each two-product minor is bounded
conservatively by 2^260, and each three-term triple determinant or permanent by
2^392, allowing ample rounding slack. A three-square lift is bounded by 2^261.
Each lift times triple is then at most 2^654, and the four-term determinant or
permanent is at most 2^658, again allowing rounding slack. The simpler uniform
bound 2^660 safely covers these expressions and is far below float64 overflow.
No arithmetic is performed in float32 before promotion; this includes subnormal
input coordinates. Nonfinite direct test inputs cannot pass the strict gate.

Validation must independently compare the complete enclosure, not just sampled
accepted signs, to rational original-coordinate determinants. Captured query
tests compare complete symbolic winners and exercise changes to coordinates,
duplicate owners, and active prefixes. These numerical checks support, but do
not replace, the published bound and the range/compiler assumptions above.
