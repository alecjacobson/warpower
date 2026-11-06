# Conservative circumsphere boxes

The exact Delaunay certificate first rejects BVH nodes disjoint from an outward
float32 box of the tetrahedron's original-coordinate circumsphere. Nodes that
survive still use the existing float64 polynomial interval test and exact leaf
predicates. An uncertain or nonfinite box disables this new rejection test.
Only strict separation permits rejection; touching boxes and exact ties survive.
No output, failure code, certificate tolerance, or construction policy changes.

## Bound construction

The existing `sphere()` computes float64 center-numerator and determinant balls
using original float32 coordinates. Let the exact numerator and positive
denominator lie in `a +/- ea` and `b +/- eb`, with `b > eb >= 0`.
For the rounded quotient `q = fl(a/b)`,

```
|A/B - q| <= (ea + |q|*eb + |a - b*q|) / (b - eb).
```

For normal division, `|a - b*q| <= eps64*b*|q|`, where `eps64 = 2^-52`
is twice unit roundoff. The implementation rounds the positive denominator
bound downward by `1 - 4*eps64` and an absolute normal-number allowance.
It pads the positive numerator and its division by `1 + 16*eps64` each.
The numerator expression has two products in its longest product term and two
additions: even with every rounding downward, its relative loss is smaller
than this allowance. The second allowance covers division and its own rounding.
Absolute `16*min_normal64` pads in numerator, denominator and quotient cover
underflow rounding; a nonpositive denominator or nonfinite result disables the
box. For the finite float32 polynomial coefficients used here, nonzero degree
three/four terms and their error estimates are normal float64 values anyway.

Multiplying the quotient balls by one half gives an enclosure of the relative
circumcenter. Sum three squared balls for the radius squared. The positive upper
endpoint receives `1 + 4*eps64` and `16*min_normal64` padding, followed by a
similarly padded square root. The absolute pad covers subnormal squared terms
that the polynomial Ball primitives do not otherwise promise to enclose. A
nonfinite radius disables the box. Adding the original first vertex gives a
world-center ball for each axis. Endpoints receive an additional
`16*eps64*(abs(center) + center_error + radius) + 16*min_normal64` outward
allowance, larger than the roundoff of their additions/subtractions and the
allowance computation itself.

Before conversion to float32, each endpoint moves outward by
`4*eps32*abs(endpoint) + 4*min_normal32`, where `eps32 = 2^-23`.
This covers endpoint-padding arithmetic, correctly rounded conversion and
possible subnormal flushing. Nonfinite converted endpoints disable the box.
BVH bounds enclose original float32 site positions, so strict disjointness of
these boxes proves a node contains no point on or inside the exact sphere.
All relevant Warp kernels disable fast math and fused multiply-add.

Large offsets can make the world-coordinate float32 box loose. Nearly flat
hull tetrahedra can have enormous circumspheres. Retaining the original
polynomial test on surviving nodes is essential: its approximate supporting
halfspace can prune cases where a sphere's axis-aligned box covers the entire
point cloud. This is a conservative pretest, not a replacement predicate.

## Independent checks

`tests/test_sphere_bounds.py` constructs exact rational Cramer centers and squared
radii for finite float32 tetrahedra spanning coordinate exponents -140 through
125, including large translations and nearly flat configurations. For every
reported finite box, exact rational arithmetic verifies for each axis that
`center - lower >= 0`, `upper - center >= 0`, and both squared distances are at
least the exact radius squared. This avoids an approximate square-root oracle.
Coplanar, coincident, negative-orientation and nonfinite inputs exercise the
fallback. These tests support the arithmetic derivation; randomized testing
alone is not a proof of conservativeness.
