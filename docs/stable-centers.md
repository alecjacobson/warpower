# Consistent generated Voronoi centers

Refinement inserts the float32 position of each tetrahedron's circumcenter.
Different tetrahedra on exactly the same sphere must therefore produce the
same insertion position. Ordinary float64 evaluation does not guarantee that:
three actual hand cells have an exact center with `z=0`, but produce residuals
around `1e-17`. Rounding those residuals to float32 preserves them, creating
distinct sites and nearly flat cells. Canonicalizing each tetrahedron's vertex
order alone cannot make different tetrahedra agree.

The private `_stable_center.circumcenter` now certifies the rounding of generated
centers. Original input sites are unchanged. A conservative filter proves when
the old approximate center is already correctly rounded to float32; otherwise
bounded integer arithmetic evaluates the rational center and tests exact
rounding midpoints. The existing duplicate-site machinery then merges centers
that actually coincide. This is not an approximate distance-based merge.

## Arithmetic and range

Let `a,b,c` be the three original-coordinate differences from `p0`, and
`D=2 det(a,b,c)`. Coordinate `j` of the center is `N_j/D`, where
`N_j = p0_j D + |a|² (b×c)_j + |b|² (c×a)_j + |c|² (a×b)_j`.
The filter evaluates binary64 approximations and corresponding absolute-product
permanents. With unit roundoff `u=2^-53`, a `64*u` margin conservatively covers
coordinate differences, products, sums, and the rounded permanents. The longest
expanded degree-four term has fewer than 20 rounding operations along its path;
FMA can only remove roundings. Every nonzero intermediate product and error bound
is normal and finite in binary64 for finite float32 sites.

The denominator sign must first be certain. For each exact binary64 midpoint
`m` adjacent to the proposed float32 coordinate, the filter bounds `N-D*m`
using the numerator error, `abs(m)` times the denominator error, and the
roundoff of the final multiply/subtract. Strict opposite signs at the lower
and upper midpoints prove the proposed rounding. Uncertain comparisons and
exact ties use the integer fallback. The old approximate center/radius function
is isolated from this filter to preserve its arithmetic on the fast path.

The fallback scales original float32 coordinates into integers with common
exponent `base`, where `-149 <= base <= 0`. Its denominator has degree three and
its numerator degree four; the physical ratio is `N/D * 2**base`. A sign-normalized
denominator permits comparisons of positive magnitudes. To test a midpoint, it
compares a shifted `2*N` against `D*(2*C ± step)` in one common integer scale.
Equality is accepted only for the even float32 significand. At most one adjacent
float32 correction is needed after the binary64 ratio estimate; the corrected
interval is checked again, and any failure remains explicit.

A conservative magnitude bound for the degree-four numerator is
`2**(522-4*base)`. Aligning a midpoint adds at most `base+150` bits, so even that
comparison stays below 1,120 bits. The existing 48-word (1,536-bit) integer
representation has room for every intermediate; explicit capacity checks remain.
Physical numerator/denominator estimates and their ratio are normal and finite
in binary64 (the ratio lies conservatively between `2**-987` and `2**968` when
nonzero). Their few-unit-roundoff estimate error is far below one float32 ulp.

Zero is canonicalized. Subnormal rounding and nearest-even ties are handled
explicitly. A center beyond the finite float32 rounding range is invalid for
insertion; truly coplanar or nonfinite input is also invalid. Radius and priority
remain binary64 geometric estimates, not exact rational outputs. This change
does not make an approximate material query exact or guarantee manifold
extraction.

The exact fallback estimates the radius from the local rational offset, before
adding the origin. Subtracting a global binary64 center from a large translated
origin loses precision even when the insertion position is rounded correctly.
A translated-cell regression test caught a `6.6e-11` relative squared-radius
error in the first version; the local-offset calculation addresses that loss.

## Scope and validation

Only refinement uses the new helper. Independently callable power-diagram,
Delaunay, ODT, extraction, and fixed-site VTM routines retain their code paths.
All arithmetic runs in Warp on device; no host predicates, counts, allocations,
or numerical fallbacks occur during replay.

Tests compare with independent Python `Fraction` circumcenters and exact
nearest-even rounding. They cover six real cospherical hand parents, every
vertex permutation and three scales, 96 finite float32 bit-pattern clouds,
512 ordinary random tetrahedra, midpoint ties, subnormal and maximum scales,
unrepresentable centers, coplanarity, and nonfinite input. A separate FMA-enabled
caller checks the filter under both compilation policies. A captured
producer/consumer loop switches between exact and fast paths, rejects invalid
input, and recovers. The initial 18 focused helper/refinement tests pass. Full
examples, performance and instrumented validation are still in progress before merge.
