# Certified analytic filters for VTM face search

The VTM reuse constructor's strided exposed-face search uses analytic float64
determinant bounds before the existing interval and exact predicates. A filter
returns only a certified strict nonzero sign. An uncertain or nonfinite result
falls through to the original Ball64, dyadic-integer and symbolic policy.
Uncertainty is never interpreted as a tie. Standalone diagram and Delaunay
kernels retain their existing predicates.

The expressions are adapted from Jonathan Richard Shewchuk's public-domain
May 18, 1996 [predicates.c](https://www.cs.cmu.edu/afs/cs/project/quake/public/code/predicates.c),
linked from his [robust predicates page](https://www.cs.cmu.edu/~quake/robust.html).
The source retains attribution. This is a pure-Warp implementation of the first
filter, with the existing pure-Warp fallback; it does not call the C reference.

## Bounds and conventions

Let P be the nonnegative permanent computed with the reference's grouping and
u the float64 unit roundoff. Shewchuk's first-stage orientation and sphere
coefficients are (7 + 56u)u and (16 + 224u)u, respectively. Warpower uses these
larger, exactly representable powers of two:

| Query | Error bound | Accepted sign |
|---|---|---|
| Orientation | 2^-49 P | sign(det), only if abs(det) exceeds the bound |
| Unweighted sphere | 2^-47 P | sign(det), only if abs(det) exceeds the bound |

Both constants exceed the reference coefficients even with the conservative
choice u = 2^-52, twice the actual binary64 unit roundoff. Scaling by these powers
of two is exact in the normal range established below. The computed determinant
and error bound must both be finite. Equality with the bound falls back.

Orientation uses the reference arguments (b, c, d, a), giving Warpower's
det(b-a, c-a, d-a) convention. Sphere negates the reference's grouped determinant.
Neither transformation changes the absolute error bound. The physical sphere
test has no weights; exact physical ties still use the original hash-weight and
ascending-site-ID symbolic policy, including its final exact cofactors.

Original float32 operands are promoted to float64 **before subtraction**.
Calling kernels disable fast math and fused arithmetic. Operation grouping is
part of the bound, not a formatting preference. The consumer uses only the
predicate's valid flag and sign; the new rounded estimate need not have the old
interval expression's bit pattern.

## Finite-input range

Every finite float32 coordinate has magnitude below 2^128 and lies on the lattice
with spacing 2^-149. A rounded float64 coordinate difference is at most 2^129
and preserves that lattice: finer representable spacing can represent the exact
lattice value, while coarser spacing rounds to a coarser lattice point.

Products of degree k preserve a lattice no finer than 2^(-149k). Rounded sums of
terms of the same degree preserve it as well. Therefore a nonzero degree-five
intermediate is at least 2^-745. The smallest nonzero sphere error bound is at
least 2^-792; the orientation bound is at least 2^-496. These are normal float64
numbers, including when original coordinates are float32 subnormals.

For an upper bound, a product of two differences is at most 2^258. Conservative
bounds of 2^260 for a two-product minor, 2^392 for a three-term triple expression,
and 2^261 for a three-square lift allow rounding slack. A lift-times-triple term
is at most 2^654; four-term sums fit below 2^658 with further slack. The uniform
bound 2^660 safely covers the intermediates and is far below float64 overflow.
Thus neither underflow nor overflow invalidates the published roundoff argument
for finite original float32 inputs. Direct nonfinite tests must fail the strict
gate; normal runtime validation also rejects nonfinite geometry.

## Independent checks

Tests compare complete error enclosures and accepted signs against rational
original-coordinate determinants evaluated by an independent Bareiss algorithm.
They cover random finite float32 bit patterns, subnormals and maximum magnitudes,
near-singular integer shears, translations, permutations, exact ties, one-ulp
changes, NaN and infinity, at 32- and 256-thread launch sizes.

Captured query tests compare every strided lane and the final proposals against
the preceding Ball64/exact search. Six producer phases change coordinates,
duplicate ownership and active face prefixes, including empty and restored
prefixes. Cube, octahedron, thin, random and fitted 4k clouds span scales from
2^-60 through 2^60. Small clouds also compare sampled complete symbolic winners
against rational predicates. These checks support the error analysis; they do
not replace its grouping, compiler and input-range assumptions.

Tracked in [#166](https://github.com/alecjacobson/warpower/issues/166).
