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
All arithmetic and fallbacks run in Warp on device. There are no host geometry
decisions, count readbacks, or workspace allocations during replay.

Tests compare with independent Python `Fraction` circumcenters and exact
nearest-even rounding. They cover six real cospherical hand parents, every
vertex permutation and three scales, 96 finite float32 bit-pattern clouds,
512 ordinary random tetrahedra, midpoint ties, subnormal and maximum scales,
unrepresentable centers, coplanarity, and nonfinite input. A separate FMA-enabled
caller checks the filter under both compilation policies. A captured
producer/consumer loop switches between exact and fast paths, rejects invalid
input, and recovers. All 19 focused helper/refinement tests and all 482 suite
tests pass (277.91 s for the complete suite).

Eight arithmetic/capture tests pass padded CUDA memory checking in 2.20 s with
zero errors. The four-phase resident center loop also passes full allocation/
lifetime checking with pooling disabled. Native tracing records 12 GPU kernels,
four graph launches and one final synchronization; its three-node graph contains
only kernels, with no copies, host callbacks, or allocation/free nodes. Diagnostic
memory checking uses the documented 32 KiB stack.

Final L40 medians use three captured replays, with no CPU quota throttling:

| Complete 1,000-site solve | Before, ms | After, ms |
|---|---:|---:|
| Uniform torus | 3,207.373 | 3,204.864 |
| Graded torus | 5,895.893 | 5,894.622 |

Sites, roles, and oriented full/material/boundary connectivity match bit for bit
after sorting rows, for both examples. Graded per-cell radius ratios also match
after aligning them with their cells; raw row order can differ. The uniform
baseline reuses PR111's measured runtime, byte-identical to main `095595c`; the
graded before/after pair was run afresh. Fifty-one other existing runtime modules
and README remain byte-for-byte unchanged, including independent geometry paths.

The final thin-hand attempt reaches all 1,000 sites with no rejected growth
batches in 11.428 s, compared with the previous 41.257 s attempt. These are first
complete attempts with zero CPU throttling, not successful material meshes or
a CPU speedup claim. The final 5,655-cell full mesh still yields a nonmanifold
material boundary (20 edges of incidence four), and the public result correctly
fails with error 64 and zero counts. A diagnostic sweep over 15 extraction
tolerances from 0 to 0.9 finds no valid boundary, so changing that scalar alone
is not adopted as recovery. This remains [#112](https://github.com/alecjacobson/warpower/issues/112).

The raw suite/memory/native results, before/after measurements, output/source
audits, and invalid-hand diagnostics are saved under
[`results/validation/vtm/stable-center-*`](../results/validation/vtm/).
Prototype timing records and the rejected global-radius calculation are retained
separately from the final results. The overall CPU performance goal remains open.
