Prepared point-query enclosure

For a positive defining tetrahedron, the existing preparation computes Ball64
enclosures for its orientation determinant D and sphere cofactors nx, ny, nz,
with original float32 corner a as origin. For query p, evaluate

    (p-a) dot (nx,ny,nz) - D * ((px-ax)^2 + (py-ay)^2 + (pz-az)^2).

This polynomial is the original-coordinate unweighted in-sphere determinant in
Warpower's convention: positive is inside for a positive defining cell. The
original general box query evaluates this polynomial with interval(lo, hi).
When lo == hi == p, the point itself has no input uncertainty: promotion from
float32 to float64 is exact. Using fp.difference(p[k], a[k]) therefore removes
the unnecessary generic midpoint/radius operations and their error padding.
It does not drop the rounding error of subtraction or the uncertainty of any
prepared cofactor. All subsequent Ball64 plus/minus/times/lift operations retain
the existing bounds and operation grouping. Calling kernels disable fast_math
and fused arithmetic, as required by the original Ball64 proof.

A finite value/error pair with abs(value) > error certifies a strict physical
sign. Negative skips this query; positive clears the candidate's live flag.
Equality, a zero enclosure, uncertainty, and nonfinite results fall through to
the original complete symbolic_sphere. Thus a physical tie retains the original
hash-weight then ascending-site-ID infinitesimal policy. No tolerance becomes a
tie. Existing finite-input checks, canonical ownership, positive orientation,
per-invocation preparation, live resets, full certification and fallback remain.

The adopted candidate continues scanning after rejection, retaining the old
exhaustive error-query behavior. #169's optional stop was evaluated separately
but is not required by this arithmetic change. No new arrays or launches are
needed, and only VTM's private PreparedSeedQueries path changes.

The finite float32 coordinate range fits the existing Ball64 degree-five proof:
differences are bounded by 2^129 and preserve the 2^-149 coordinate lattice;
nonzero degree-five geometric products are at least 2^-745. Magnitudes and
roundoff intermediates stay within normal binary64 range under the existing
padded Ball arithmetic assumptions. Tests validate the full value +/- error
enclosure against independent rational Bareiss determinants, including random
finite bit patterns, min/max/subnormal coordinates, thin cells, shifts, exact
ties and one-ulp changes. These tests support, rather than replace, that proof.

Captured mask tests compare the complete live/error outputs with the previous
exhaustive checker at both 32- and 256-thread blocks across six device-produced
frames and two graph replays. They change coordinates, duplicate ownership and
candidate prefixes, including empty and restored prefixes; all candidates are
canonical positive-volume cells as required by this private interface. Small
clouds additionally use independent rational physical and symbolic predicates.
