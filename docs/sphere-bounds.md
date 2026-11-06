# Conservative circumsphere boxes

The exact Delaunay certificate first rejects BVH nodes disjoint from an outward
float32 box of the tetrahedron's original-coordinate circumsphere. Nodes that
survive still use the existing float64 polynomial interval test. At leaves, the
same polynomial interval test first rejects individual points using the cached
cofactors; uncertain points retain the symbolic/exact predicate. An uncertain or nonfinite box disables this new rejection test.
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

## Cached-cofactor leaf points

A BVH leaf can overlap the sphere even when most of its sites are outside.
For each surviving site, `outside(shape, point, point)` evaluates the existing
polynomial interval with both endpoints equal. It reuses the center numerators
and positive orientation determinant already computed for the tetrahedron.
For exact relative coordinates `x,y,z`, the expression is
`x*Nx + y*Ny + z*Nz - D*(x*x + y*y + z*z)`. The existing Ball operations enclose
its exact value; rejection requires a finite upper endpoint strictly below zero.
Such a point has a strictly negative ordinary sphere determinant, so symbolic
tie breaking cannot change its classification. Every uncertain result still
uses the original symbolic/exact predicate.

This introduces no division, new tolerance, scratch storage or kernel launch.
The certificate still checks every global lower facet, hull support, face
incidence and canonical-site coverage. The leaf-query statistic counts points
before this filter, preserving its previous meaning.

`tests/test_sphere_points.py` compares the filter against exact rational centers
and squared radii on 2,395 queries, spanning exponents -140 through 120, nearly
flat tetrahedra, translations, vertex ties and neighboring float32 values.
Separate cases retain all 88 cospherical cube queries and an exactly outside
point with a determinant too small for the filter to resolve. In that last case,
the filter must defer rather than infer a sign from a tolerance.

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

## Measured effect on certified reconstruction

The [before/after samples](../results/validation/vtm/sphere-box-performance.json)
use L40, Warp 1.19.0 and identical inputs. Full/base/certificate timing windows
have no CPU quota throttling. The 100,000-site case has five samples per stage;
the smaller cases have ten. Medians in milliseconds:

| Input | Complete before | Complete after | Certificate before | Certificate after |
|---|---:|---:|---:|---:|
| random100k (100000 sites) | 1116.913 | 765.726 | 646.830 | 290.187 |
| random1000 (1000 sites) | 14.661 | 12.514 | 5.760 | 3.589 |
| cube (8 sites) | 0.525 | 0.529 | 0.152 | 0.154 |
| grid (64 sites) | 2.701 | 2.343 | 1.056 | 0.716 |
| corner-cloud (200 sites) | 9.085 | 8.450 | 3.625 | 3.018 |
| torus-boundary (128 sites) | 2426.028 | 2415.974 | 112.574 | 102.359 |

For 100,000 random sites, certification is **2.23x faster** and complete certified
construction is **1.46x faster**. Ordinary construction remains 473.02 ms.
Certificate node visits fall from 383,864,890 to 46,061,402; leaf point tests fall
from 29,300,464 to 8,793,024. All 671,908 tetrahedra match both the saved independent
[native CGAL reference](../results/validation/vtm/sphere-box-100k-cgal.json) and the
preceding exact-certified result. Tiny meshes are dominated by fixed overhead;
no universal per-case speedup is claimed. Structured symbolic construction remains
expensive, tracked separately in issue #62. This is a certified-Delaunay comparison,
not the complete variational mesher's CPU speedup target.

```bash
python -m benchmarks.delaunay_recovery --case random --n 100000 --repeats 5 \
  --output results/local/sphere-box-100k.json
python -m pytest -q tests/test_sphere_bounds.py
```

The complete [365-test regression suite](../results/validation/vtm/sphere-box-full-tests.txt)
passes. Five [padded pooled-memory cases](../results/validation/vtm/sphere-box-memcheck.txt)
cover the interval bounds, invalid certificates and one-ULP perturbations; one
[captured changing-input lifetime case](../results/validation/vtm/sphere-box-lifetime.txt)
uses synchronous diagnostic allocations and stream-ordered race tracking. All
six finish with zero sanitizer errors, using the documented 32 KiB diagnostic
stack without changing production settings. The [native trace](../results/validation/vtm/sphere-box-device-audit.json)
records 321 kernel activities across three replays; all 21 graph copies are
device-to-device, with no host/callback or allocation/free nodes. The
[isolation record](../results/validation/vtm/sphere-box-isolation.json) checks all
other runtime modules against the preceding main revision.
