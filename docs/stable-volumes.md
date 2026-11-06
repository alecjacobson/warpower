# Robust volumes for ODT accumulation

An exactly certified positive tetrahedron can have a negative determinant when
its nearly cancelling products are evaluated in floating point. Two cells in a
576-site thin-hand refinement trial exhibit this: their exact determinants are
`6.138181545497544e-19` and `1.6789178734399065e-27`, while the former FMA-enabled
GPU expression returns `-5.2470370879720105e-20` and `-4.250052371710322e-21`.
Consequently a valid Delaunay insertion batch was rolled back during ODT
accumulation. The original coordinates and independent integer references are
saved in `tests/data/tiny-positive-tets.{npz,json}`.

Uniform and graded accumulation now filter the usual binary64 determinant.
For the rounded edge vectors `a,b,c`, let `P` be the sum of the absolute values
of the six determinant products. A computed determinant whose magnitude exceeds
`64*u*P` is retained, where `u=2^-53`. Otherwise the existing exact integer
orientation routine evaluates the original float32 coordinates and supplies
the physical binary64 determinant estimate. A genuine inverted or zero-volume
cell still fails; nonfinite input still fails. No positions are moved and no
absolute value is substituted for the signed determinant.

## Filter and compilation

Each coordinate difference has relative error at most `u`. Each expanded
determinant term has at most three difference roundings, two product roundings,
and five summation roundings along its path, so the usual `gamma_10` bound
against the exact absolute-product sum applies. Grouping cross products first
only reduces this path length; an FMA removes a rounding. The computed permanent
is itself within a `gamma_10` factor of that sum. The chosen `64*u` margin is
larger than `gamma_10/(1-gamma_10)` and hence conservatively covers both.
For finite float32 coordinates, nonzero differences/products and the bound
stay normal in binary64, and their maximum magnitudes remain finite. Thus the
relative-error argument has no hidden underflow or overflow case.

The exact orientation fallback uses bounded integer arithmetic and does not
depend on the caller's FMA policy. This matters because the interval predicates
require `fuse_fp=False`, whereas the existing ODT kernels permit FMA.
The filter helper is deliberately not inlined: an inlined permanent can cause
the compiler to share products with the original determinant and alter FMA
contraction. A separate original-expression kernel verifies bit-for-bit
preservation on well-conditioned random cells. The first inlined prototype
failed that stronger test and was rejected.

Only the uniform and graded ODT accumulation kernels use this helper. The
independent power-diagram and Delaunay implementations have no new code path,
buffer, or dispatch. This does not fix refinement-center rounding (#109),
promise good quality for nearly flat tetrahedra, or establish full thin-hand
meshing success. Those require separate end-to-end checks.

## Validation

Focused tests compare all vertex permutations of the four saved cells at three
scales with Python integer determinants, including the smallest subnormal and
largest finite float32 axis tetrahedra. They also exercise uniform/graded
physical volume and mass, genuine inversion, and a captured producer/consumer
loop with nonfinite input followed by recovery. Full-suite, memory, native graph,
and complete mesher performance results will be recorded before merge.
